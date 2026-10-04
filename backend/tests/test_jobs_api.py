import io
import json
import threading
import zipfile
import av
import numpy as np
from iris_pipeline.media import write_png
from conftest import finished
from test_stages import video


def create(client, image, **options):
    response = client.post('/api/jobs/paths',json={'paths':[str(image)],'options':options})
    assert response.status_code == 201, response.text
    return response.json()['id']


def test_full_job_keeps_poor_crops_and_reruns_preserve_history(client,image):
    job_id = create(client,image)
    assert finished(client,job_id)['status'] == 'completed'
    items = client.get(f'/api/jobs/{job_id}/detections').json()
    assert items['total'] == 2
    item = items['items'][0]
    assert item['needs_review'] and 'small' in item['review_flags']
    assert item['raw_text'] == '123بسم'
    assert client.get(f"/api/jobs/{job_id}/artifacts/{item['original']}").content.startswith(b'\x89PNG')
    detection_id = item['id']
    response = client.post(f'/api/detections/{detection_id}/correction',json={'text':'456 س م','note':'operator verified'})
    assert response.status_code == 200, response.text
    assert response.json()['needs_review'] is False
    calls = client.app.state.runtime.calls
    for stage in ['ocr','enhance']:
        assert client.post(f'/api/jobs/{job_id}/rerun',json={'stage':stage}).status_code == 200
        assert finished(client,job_id)['status'] == 'completed'
    assert client.app.state.runtime.calls == calls
    history = client.get(f'/api/detections/{detection_id}/history').json()
    assert len(history['predictions']) == 3 and len(history['enhancements']) == 2
    assert history['corrections'][0]['note'] == 'operator verified'
    assert client.get(f'/api/jobs/{job_id}/detections?review=true').json()['total'] == 1
    assert client.get(f'/api/jobs/{job_id}/detections?limit=1&offset=1').json()['items'][0]['id'] != detection_id
    assert client.get(f'/api/jobs/{job_id}/detections?max_confidence=0.5').json()['total'] == 0
    assert client.get(f'/api/jobs/{job_id}/events').text.startswith('data: ')
    complete = client.get(f'/api/jobs/{job_id}/export/json').json()
    assert len(complete['runs']) == 3 and len(complete['detections']) == 2
    assert '456 س م' in client.get(f'/api/jobs/{job_id}/export/csv').text
    content = client.get(f'/api/jobs/{job_id}/export/crops').content
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert 'manifest.jsonl' in archive.namelist()
        assert len([name for name in archive.namelist() if name.endswith('.png')]) == 6
    assert client.get(f'/api/jobs/{job_id}/artifacts/%2e%2e%2f%2e%2e%2fpipeline.sqlite3').status_code == 404


def test_sampling_multiple_plates_and_annotated_video_timing(client,tmp_path):
    path = video(tmp_path/'video.mp4')
    job_id = create(client,path,stop_after='detect')
    assert finished(client,job_id)['status'] == 'completed'
    items = client.get(f'/api/jobs/{job_id}/detections?limit=100').json()
    assert items['total'] == 20
    assert [d['frame_index'] for d in items['items']] == [i for i in range(0,60,6) for _ in range(2)]
    response = client.get(f'/api/jobs/{job_id}/export/annotated')
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        annotated = tmp_path/'annotated.mp4'
        annotated.write_bytes(archive.read('s00000.mp4'))
    with av.open(str(path)) as original, av.open(str(annotated)) as annotated:
        before, after = list(original.decode(video=0)), list(annotated.decode(video=0))
        assert len(before) == len(after) == 60
        assert [float(f.time) for f in before] == [float(f.time) for f in after]
    assert client.post(f'/api/jobs/{job_id}/rerun',json={'stage':'ocr'}).status_code == 400


def test_annotated_variable_frame_timestamps(client,tmp_path):
    path = video(tmp_path/'vfr.mp4',[0,30,70,250,280,600,610,950])
    job_id = create(client,path,stop_after='detect')
    assert finished(client,job_id)['status'] == 'completed'
    response = client.get(f'/api/jobs/{job_id}/export/annotated')
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        result = tmp_path/'result.mp4'
        result.write_bytes(archive.read('s00000.mp4'))
    with av.open(str(path)) as original, av.open(str(result)) as rendered:
        assert [float(f.time) for f in original.decode(video=0)] == [float(f.time) for f in rendered.decode(video=0)]


def test_upload_crop_folder_recursive_and_duplicate_names(client,image,tmp_path):
    other = tmp_path/'sub'/'plate.png'
    write_png(other,np.zeros((30,100,3),dtype=np.uint8))
    response = client.post('/api/jobs/paths',json={'paths':[str(tmp_path)],'recursive':True,'options':{'mode':'crop','stop_after':'enhance'}})
    job_id = response.json()['id']
    assert finished(client,job_id)['counts']['detections'] == 2
    sources = client.get(f'/api/jobs/{job_id}').json()['sources']
    assert len({s['stored'] for s in sources}) == 2
    assert client.app.state.runtime.calls == 0
    response = client.post('/api/jobs/upload',files=[('files',('folder/plate.png',image.read_bytes(),'image/png'))],data={'options':json.dumps({'mode':'crop'})})
    assert response.status_code == 201
    assert finished(client,response.json()['id'])['status'] == 'completed'
    assert client.post('/api/jobs/upload',files=[('files',('../plate.png',image.read_bytes(),'image/png'))]).status_code == 400
    assert client.post('/api/jobs/paths',json={'paths':[str(image)],'options':{'mode':'crop','stop_after':'detect'}}).status_code == 422


def test_failed_ocr_retry_and_cancellation(client,image):
    runtime = client.app.state.runtime
    runtime.recognizer.fail = True
    job_id = create(client,image)
    assert finished(client,job_id)['status'] == 'failed'
    calls = runtime.calls
    assert client.post(f'/api/jobs/{job_id}/retry').status_code == 200
    assert finished(client,job_id)['status'] == 'completed'
    assert runtime.calls == calls
    runtime.wait = threading.Event()
    second = create(client,image,stop_after='detect')
    queued = create(client,image,stop_after='detect')
    assert client.post(f'/api/jobs/{second}/cancel').status_code == 200
    assert client.post(f'/api/jobs/{queued}/cancel').status_code == 200
    runtime.wait.set()
    assert finished(client,second)['status'] == 'cancelled'
    assert finished(client,queued)['status'] == 'cancelled'


def test_accuracy_manifest_and_settings(client,image,tmp_path):
    manifest = tmp_path/'labels.json'
    manifest.write_text(json.dumps([{'path':image.name,'truth':'123بسم'}]),encoding='utf-8')
    response = client.post('/api/benchmarks',json={'manifest_path':str(manifest)})
    assert response.status_code == 201, response.text
    job = finished(client,response.json()['id'])
    assert job['status'] == 'completed'
    assert job['benchmark']['original']['exact_match_accuracy'] == 1
    assert job['benchmark']['enhanced']['exact_match_accuracy'] == 1
    response = client.put('/api/settings',json={'defaults':{'sample_fps':2},'presets':{'night':{'sample_fps':1}}})
    assert response.status_code == 200
    assert client.get('/api/settings').json()['presets']['night']['sample_fps'] == 1
    assert client.post('/api/benchmarks',json={'manifest_path':str(tmp_path/'missing.json')}).status_code == 400
    manifest.write_text('[{"truth":"123"}]',encoding='utf-8')
    assert client.post('/api/benchmarks',json={'manifest_path':str(manifest)}).status_code == 400
