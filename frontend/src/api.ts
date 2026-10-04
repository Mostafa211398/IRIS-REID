export type Options = {mode: 'vehicle'|'crop'; stop_after: 'detect'|'enhance'|'ocr'; sample_fps: number;
  confidence: number; iou: number; image_size: number; padding: number; upscale: number; detail_enhancement: boolean;
  review_confidence: number; review_quality: number};
export const defaults: Options = {mode: 'vehicle', stop_after: 'ocr', sample_fps: 5, confidence: .4, iou: .45,
  image_size: 1280, padding: 1, upscale: 3, detail_enhancement: true, review_confidence: .8, review_quality: .36};
export type Stats = {samples: number; exact_match_accuracy: number; character_error_rate: number; normalized_edit_similarity: number};
export type Prediction = {raw_text: string; ocr_confidence: number};
export type Run = {id: string; stage: string; status: string; timings: Record<string,number>; duration?: number; detections_per_second?: number; created: string};
export type Job = {id: string; created: string; status: string; stage: string; kind: string; options: Options;
  sources: {id: string; name: string}[]; counts: {detections: number}; error?: string; progress: {done?: number; total?: number; sampled_frames?: number; source?: string};
  benchmark?: {original: Stats; enhanced: Stats}; runs?: Run[]};
export type Detection = {id: string; job_id: string; source_id: string; source_name: string; frame_index: number; timestamp: number;
  bbox: number[]; detector_confidence: number|null; quality: Record<string,number>; original: string; enhanced: string|null; source_frame: string;
  raw_text: string|null; corrected_text?: string; digits?: string; arabic_letters?: string; ocr_confidence: number|null;
  needs_review: boolean; review_flags: string[]; enhancement?: {method: string; upscale: number};
  comparison?: {truth: string; original: Prediction; enhanced: Prediction}};
export type Health = {state: string; device: string; torch?: string; gpu?: {name: string; vram_bytes: number; free_bytes: number; compute_capability: string};
  stages: Record<string,{ready: boolean; error?: string; model_inference?: boolean}>};
export async function api<T>(path: string, body?: unknown, method = 'POST'): Promise<T> {
  const response = await fetch('/api'+path, body === undefined ? {} : {method, headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)});
  if (!response.ok) {const error = await response.json().catch(() => ({})); throw new Error(typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail || response.statusText));}
  return response.json();
}
export function artifact(job: string, relative: string) {return `/api/jobs/${job}/artifacts/${relative.split('/').map(encodeURIComponent).join('/')}`;}
export const active = (job?: Job|null) => !!job && ['queued','running','cancelling'].includes(job.status);
export const percent = (value: number|null|undefined) => value == null ? '—' : `${(value*100).toFixed(1)}%`;
