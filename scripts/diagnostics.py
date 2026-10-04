import json
from iris_pipeline.config import settings
from iris_pipeline.runtime import Runtime

runtime = Runtime(settings)
try:
    result = runtime.diagnostics()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (settings.root / ".runtime" / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if result["state"] == "ready" else 1)
finally:
    runtime.close()
