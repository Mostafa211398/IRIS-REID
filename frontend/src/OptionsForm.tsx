import type {Options} from './api';
export default function OptionsForm({value, change, advanced = false}: {value: Options; change: (value: Options)=>void; advanced?: boolean}) {
  const set = <K extends keyof Options>(key: K, next: Options[K]) => change({...value, [key]: next});
  return <div className="options-form">
    <div className="field-row three">
      <label>Image mode<select value={value.mode} onChange={e=>change({...value, mode: e.target.value as Options['mode'], stop_after: e.target.value === 'crop' && value.stop_after === 'detect' ? 'enhance' : value.stop_after})}><option value="vehicle">Full vehicle images / video</option><option value="crop">Already-cropped plates</option></select></label>
      <label>Process through<select value={value.stop_after} onChange={e=>set('stop_after', e.target.value as Options['stop_after'])}>{value.mode === 'vehicle' && <option value="detect">Detection only</option>}<option value="enhance">Enhancement</option><option value="ocr">Full pipeline · OCR</option></select></label>
      <label>Video samples per second<input type="number" min="0.1" max="60" step="0.1" value={value.sample_fps} onChange={e=>set('sample_fps', Number(e.target.value))}/></label>
    </div>
    <div className="field-row three"><label>Detector confidence<input type="number" min="0.01" max="1" step="0.01" value={value.confidence} onChange={e=>set('confidence', Number(e.target.value))}/></label><label>Enlargement factor<select value={value.upscale} onChange={e=>set('upscale', Number(e.target.value))}>{[1,2,3,4,5,6].map(n=><option key={n} value={n}>{n}× bicubic</option>)}</select></label><label>Review below OCR confidence<input type="number" min="0" max="1" step="0.01" value={value.review_confidence} onChange={e=>set('review_confidence', Number(e.target.value))}/></label></div>
    <label className="check"><input type="checkbox" checked={value.detail_enhancement} onChange={e=>set('detail_enhancement', e.target.checked)}/>Conditional local contrast and sharpening</label>
    {advanced && <div className="field-row four"><label>Detector IoU<input type="number" min="0.01" max="1" step="0.01" value={value.iou} onChange={e=>set('iou', Number(e.target.value))}/></label><label>Inference image size<input type="number" min="320" max="2048" step="32" value={value.image_size} onChange={e=>set('image_size', Number(e.target.value))}/></label><label>Crop padding (pixels)<input type="number" min="0" max="100" value={value.padding} onChange={e=>set('padding', Number(e.target.value))}/></label><label>Review below crop quality<input type="number" min="0" max="1" step="0.01" value={value.review_quality} onChange={e=>set('review_quality', Number(e.target.value))}/></label></div>}
  </div>;
}
