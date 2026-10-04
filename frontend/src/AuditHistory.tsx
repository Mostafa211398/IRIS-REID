import {artifact, percent} from './api';

export type Audit = {
  predictions: {raw_text: string; ocr_confidence: number; created: string; input: string}[];
  enhancements: {created: string; method: string; upscale: number; path: string}[];
  corrections: {created: string; text: string; note: string}[];
};

export default function AuditHistory({audit, jobId}: {audit: Audit; jobId: string}) {
  const date = (value: string) => new Date(value).toLocaleString();
  return <div className="audit-history">
    <h3>Prediction history</h3>
    {audit.predictions.map((reading,index)=><div className="audit-row" key={index}>
      <span><bdi dir="auto">{reading.raw_text || '(empty reading)'}</bdi><small>{reading.input.startsWith('enhanced/') ? 'Enhanced crop' : 'Original crop'} · {date(reading.created)}</small></span>
      <strong>{percent(reading.ocr_confidence)}</strong>
    </div>)}
    {!audit.predictions.length && <p className="subtle">Recognition has not run yet.</p>}
    <h3>Enhancement history</h3>
    {audit.enhancements.map((version,index)=><div className="audit-row" key={index}>
      <span>{version.upscale}× · {version.method === 'enlargement_only' ? 'Enlargement only' : 'Local contrast and sharpening'}<small>{date(version.created)}</small></span>
      <a className="source-link" href={artifact(jobId,version.path)} target="_blank" rel="noreferrer">View crop</a>
    </div>)}
    <h3>Operator corrections</h3>
    {audit.corrections.map((correction,index)=><div className="audit-row" key={index}>
      <span><bdi dir="auto">{correction.text || '(empty correction)'}</bdi><small>{date(correction.created)}{correction.note && ` · ${correction.note}`}</small></span>
    </div>)}
    {!audit.corrections.length && <p className="subtle">No operator corrections.</p>}
  </div>;
}
