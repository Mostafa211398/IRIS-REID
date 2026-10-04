import json
import sqlite3
import threading
from datetime import datetime, timezone
from contextlib import contextmanager


def now():
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, created TEXT, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS detections(id TEXT PRIMARY KEY, job_id TEXT, source_id TEXT,
                    review INTEGER, confidence REAL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS detections_job ON detections(job_id, source_id, review, confidence);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, job_id TEXT, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, detection_id TEXT,
                    run_id TEXT, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS predictions_detection ON predictions(detection_id);
                CREATE TABLE IF NOT EXISTS enhancements(id INTEGER PRIMARY KEY, detection_id TEXT,
                    run_id TEXT, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS corrections(id INTEGER PRIMARY KEY, detection_id TEXT, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def job(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def save_job(self, job):
        with self.lock, self.connect() as db:
            db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?)", (job["id"], job["created"], json.dumps(job)))

    def update_job(self, job_id, **changes):
        with self.lock:
            job = self.job(job_id)
            job.update(changes, updated=now())
            self.save_job(job)
            return job

    def jobs(self, offset=0, limit=50):
        with self.connect() as db:
            total = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            rows = db.execute("SELECT payload FROM jobs ORDER BY created DESC LIMIT ? OFFSET ?", (limit, offset))
            return {"total": total, "items": [json.loads(row[0]) for row in rows]}

    def save_detection(self, item):
        with self.lock, self.connect() as db:
            db.execute("INSERT OR REPLACE INTO detections VALUES(?,?,?,?,?,?)", (
                item["id"], item["job_id"], item["source_id"], int(item["needs_review"]),
                item.get("ocr_confidence"), json.dumps(item)))

    def detection(self, detection_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM detections WHERE id=?", (detection_id,)).fetchone()
        if row is None:
            raise KeyError(detection_id)
        return json.loads(row[0])

    def detections(self, job_id, offset=0, limit=50, source=None, review=None, max_confidence=None):
        where, args = ["job_id=?"], [job_id]
        if source:
            where.append("source_id=?")
            args.append(source)
        if review is not None:
            where.append("review=?")
            args.append(int(review))
        if max_confidence is not None:
            where.append("(confidence IS NULL OR confidence<=?)")
            args.append(max_confidence)
        clause = " AND ".join(where)
        with self.connect() as db:
            count = db.execute("SELECT COUNT(*) FROM detections WHERE " + clause, args).fetchone()[0]
            rows = db.execute("SELECT payload FROM detections WHERE " + clause + " ORDER BY rowid LIMIT ? OFFSET ?", [*args, limit, offset])
            return {"total": count, "items": [json.loads(row[0]) for row in rows]}

    def iter_detections(self, job_id):
        offset = 0
        while True:
            rows = self.detections(job_id, offset, 100)["items"]
            if not rows:
                return
            yield from rows
            offset += len(rows)

    def record(self, table, detection_id, run_id, payload):
        if table not in {"predictions", "enhancements"}:
            raise ValueError(table)
        with self.lock, self.connect() as db:
            db.execute(f"INSERT INTO {table}(detection_id,run_id,payload) VALUES(?,?,?)", (detection_id, run_id, json.dumps(payload)))

    def audit(self, detection_id):
        with self.connect() as db:
            return {table: [json.loads(row[0]) for row in db.execute(f"SELECT payload FROM {table} WHERE detection_id=? ORDER BY id", (detection_id,))]
                    for table in ("predictions", "enhancements", "corrections")}

    def correct(self, detection_id, text, note):
        with self.lock, self.connect() as db:
            item = self.detection(detection_id)
            payload = {"text": text, "note": note, "previous": item.get("corrected_text"), "created": now()}
            db.execute("INSERT INTO corrections(detection_id,payload) VALUES(?,?)", (detection_id, json.dumps(payload)))
            item.update(corrected_text=text, needs_review=False, reviewed=True)
            db.execute("UPDATE detections SET review=0,payload=? WHERE id=?", (json.dumps(item), detection_id))
            return item

    def save_run(self, run):
        with self.lock, self.connect() as db:
            db.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?)", (run["id"], run["job_id"], json.dumps(run)))

    def runs(self, job_id):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM runs WHERE job_id=? ORDER BY rowid", (job_id,))]

    def preference(self, key, value=None):
        with self.lock, self.connect() as db:
            if value is not None:
                db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, json.dumps(value)))
            row = db.execute("SELECT payload FROM settings WHERE id=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None
