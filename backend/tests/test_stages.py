import importlib.util
from fractions import Fraction
import av
import numpy as np
import pytest
from iris_pipeline.accuracy import edit_distance, metrics, normalize
from iris_pipeline.config import ROOT
from iris_pipeline.media import frames
from iris_pipeline.schemas import JobOptions
from iris_pipeline.stages.detection import quality
from iris_pipeline.stages.enhancement import Enhancer
from iris_pipeline.stages.recognition import parts
from iris_pipeline.workers.paddle_worker import decode, preprocess


@pytest.mark.parametrize("clear", [True, False])
def test_enhancement_matches_reference(clear):
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (35, 95, 3), dtype=np.uint8) if clear else np.full((35,95,3), 100, dtype=np.uint8)
    enhancer = Enhancer(ROOT, "cpu")
    sharp, edges = enhancer.reference.character_metrics(image, "cpu")
    expected = enhancer.reference.enhance_crop(image, char_sharpness=sharp, edge_strength=edges, device="cpu")
    result, metadata = enhancer.enhance(image, JobOptions())
    np.testing.assert_array_equal(result, expected)
    assert metadata["method"] == ("contrast_and_sharpening" if clear else "enlargement_only")
    # Extractor metrics and quality score remain faithful to the supplied script.
    spec = importlib.util.spec_from_file_location("extract_reference", ROOT/"extract_license_plates_gpu_ocr.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    measured = quality(image,.9,"cpu")
    assert measured["char_sharpness"] == pytest.approx(sharp, abs=.02)
    assert measured["edge_strength"] == pytest.approx(edges, abs=.001)
    assert measured["quality_score"] == pytest.approx(reference.quality_score(.9,95,35,measured["contrast"],sharp,edges,measured["edge_density"]), abs=.00001)


def video(path, timestamps=None):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 100, 40, "yuv420p"
        stream.time_base = Fraction(1,1000)
        stream.codec_context.time_base = Fraction(1,1000)
        times = timestamps if timestamps is not None else [round(i*1000/30) for i in range(60)]
        for timestamp in times:
            frame = av.VideoFrame.from_ndarray(np.full((40,100,3), 50, dtype=np.uint8), format="bgr24")
            frame.pts, frame.time_base = timestamp, Fraction(1,1000)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    return path


def test_timestamp_sampling_and_vfr(tmp_path):
    path = video(tmp_path/"regular.mp4")
    samples = list(frames(path,5))
    assert len(samples) == 10
    assert [sample[0] for sample in samples] == list(range(0,60,6))
    assert [sample[1] for sample in samples] == pytest.approx([i/5 for i in range(10)], abs=.002)
    vfr = video(tmp_path/"variable.mp4", [0,30,70,250,280,600,610,950])
    selected = list(frames(vfr,5))
    assert [round(row[1],2) for row in selected] == [0,.25,.6,.95]
    assert list(frames(path,5,lambda:True)) == []


def test_preprocessing_ctc_and_arabic_order():
    tensor = preprocess(np.full((24,80,3),255,dtype=np.uint8),[3,48,320])
    assert tensor.shape == (1,3,48,320)
    np.testing.assert_array_equal(tensor[0,:,:,:160],1)
    np.testing.assert_array_equal(tensor[0,:,:,160:],0)
    output = np.zeros((1,7,4),dtype=np.float32)
    for i,j in enumerate([1,1,0,1,2,2,3]):output[0,i,j] = .9
    result = decode(output,['1','ب','س'])
    assert result['text'] == '11بس'
    assert result['confidence'] == pytest.approx(.9)
    assert parts('١٢٣بسم')['digits'] == '123'
    assert parts('123بسم')['arabic_letters'] == 'مسب'


def test_benchmark_known_errors():
    samples = [{'truth':'١٢٣ بس', 'raw':{'raw_text':'123بس'}}, {'truth':'123بسم', 'raw':{'raw_text':'123بسن'}}]
    result = metrics(samples,'raw')
    assert result['exact_match_accuracy'] == .5
    assert result['character_errors'] == 1
    assert result['character_error_rate'] == pytest.approx(1/11)
    assert edit_distance('abc','ac') == 1
    assert normalize('١٢٣ أإآىة') == '123ااايه'
