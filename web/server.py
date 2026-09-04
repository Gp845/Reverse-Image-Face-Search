#!/usr/bin/env python3
"""
Web UI for the pipeline.

The pipeline runs on a worker thread and streams its log to the browser over
Server-Sent Events, because a full run takes about 75 seconds -- far too long
for a request to sit blocking, and the log is most of what makes the result
believable. Nothing here reimplements the pipeline: it calls pipeline.core.run,
the same function the CLI calls.

Deliberately for local use. There is no auth, jobs live in memory, and the
process holds a funded private key, so do not expose this to a network you do
not control. See the deployment note in the README.
"""

import io
import json
import os
import sys
import threading
import time
import uuid
from contextlib import redirect_stdout

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import chain                                     # noqa: E402
from pipeline.core import PipelineError, run                   # noqa: E402
from pipeline.encode import (DEFAULT_CONF, DEFAULT_THRESHOLD,   # noqa: E402
                             MIN_FACE_PX)

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UPLOADS = os.path.join(ROOT, "out", "uploads")

# A deployed instance is a face-identification endpoint that spends a funded key
# and a metered search quota. ACCESS_TOKEN gates every route that costs
# something. Binding to anything other than loopback without one is refused
# outright rather than left to whoever finds the URL.
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "").strip()
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))

app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
JOBS = {}
JOBS_LOCK = threading.Lock()


def _token_ok():
    if not ACCESS_TOKEN:
        return True
    supplied = (request.headers.get("X-Access-Token")
                or request.args.get("token")
                or request.form.get("token") or "")
    # compare_digest keeps the check constant-time; a token is a shared secret.
    import hmac
    return hmac.compare_digest(supplied, ACCESS_TOKEN)


@app.before_request
def _gate():
    if request.path.startswith("/api/") and not _token_ok():
        return jsonify(error="unauthorized: missing or bad access token"), 401
    return None


def _running_jobs():
    return sum(1 for j in JOBS.values() if j["status"] == "running")


class LogSink(io.TextIOBase):
    """Collects emitted lines into the job's log.

    Doubles as a stdout replacement: the search backends print progress
    directly (upload attempts, per-engine errors), and that detail is worth
    showing rather than swallowing.

    The log list is the single source of truth; readers follow it by index.
    An earlier version also pushed each line onto a queue that /api/log drained
    after replaying the backlog, which showed every line written before the
    browser connected twice.
    """

    def __init__(self, job):
        self.job = job

    def write(self, text):
        if text and text.strip():
            for line in text.rstrip("\n").split("\n"):
                self.job["log"].append(line)
        return len(text or "")

    def flush(self):
        return None


def _worker(job_id, image_path, opts):
    job = JOBS[job_id]
    sink = LogSink(job)
    try:
        with redirect_stdout(sink):
            result = run(image_path, emit=sink.write, out=job["out"], **opts)
        job["result"] = result
        job["status"] = "done"
    except PipelineError as exc:
        job["error"] = str(exc)
        job["status"] = "error"
        sink.write(f"\nSTOPPED: {exc}")
    except Exception as exc:                      # noqa: BLE001 - surfaced to UI
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["status"] = "error"
        sink.write(f"\nFAILED: {job['error']}")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/run")
def api_run():
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return jsonify(error="no image uploaded"), 400
    with JOBS_LOCK:
        if _running_jobs() >= MAX_CONCURRENT_JOBS:
            return jsonify(error=f"busy: {MAX_CONCURRENT_JOBS} runs already in "
                                 "flight, try again shortly"), 429

    job_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(ROOT, "out", "web", job_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(UPLOADS, exist_ok=True)

    ext = os.path.splitext(upload.filename)[1].lower() or ".jpg"
    image_path = os.path.join(UPLOADS, f"{job_id}{ext}")
    upload.save(image_path)

    form = request.form
    opts = {
        "backend": form.get("backend", "all"),
        "conf": float(form.get("conf", DEFAULT_CONF)),
        "threshold": float(form.get("threshold", DEFAULT_THRESHOLD)),
        "limit": int(form.get("limit", 25)),
        # Capped server-side: each face is a full set of engine queries against
        # a metered quota, and a deployed instance should not let one upload
        # spend it all.
        "faces": max(1, min(int(form.get("faces", 1)), 5)),
        "min_face": max(0, int(form.get("min_face", MIN_FACE_PX))),
        "social_only": form.get("social_only") == "true",
        "no_chain": form.get("no_chain") == "true",
        "rpc": form.get("rpc") or os.getenv("RPC_URL", "memory"),
    }

    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "status": "running", "log": [],
                        "out": out_dir, "image": image_path,
                        "result": None, "error": None}

    threading.Thread(target=_worker, args=(job_id, image_path, opts),
                     daemon=True).start()
    return jsonify(job_id=job_id)


@app.get("/api/log/<job_id>")
def api_log(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404

    def stream():
        # Follow the log by index. A late or reconnecting browser replays the
        # backlog and then continues live, and several browsers can watch the
        # same job, because each holds only its own cursor.
        sent = 0
        while True:
            log = job["log"]
            if sent < len(log):
                yield f"data: {json.dumps(log[sent])}\n\n"
                sent += 1
                continue
            if job["status"] != "running":
                yield "event: end\ndata: {}\n\n"
                return
            time.sleep(0.15)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.get("/api/result/<job_id>")
def api_result(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404
    return jsonify(status=job["status"], error=job["error"],
                   result=job["result"])


@app.get("/api/crop/<job_id>")
@app.get("/api/crop/<job_id>/<int:face>")
def api_crop(job_id, face=None):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404
    # A multi-face run names its crops probe_faceN_preview.jpg and writes no
    # plain probe_preview.jpg, so the unnumbered route has to fall back to the
    # first face or it 404s for the whole run. Padded previews come before the
    # 112x112 alignCrop, which is for the model rather than for eyes.
    names = ([f"probe_face{face}_preview.jpg", f"probe_face{face}_aligned.jpg"]
             if face else
             ["probe_preview.jpg", "probe_face1_preview.jpg",
              "probe_aligned.jpg", "probe_face1_aligned.jpg"])
    for name in names:
        path = os.path.join(job["out"], name)
        if os.path.exists(path):
            return send_file(path, mimetype="image/jpeg")
    return jsonify(error="no crop yet"), 404


@app.post("/api/verify")
def api_verify():
    """Re-verify a record against the chain.

    The UI posts the record back rather than reading it from disk, which is
    what makes the tamper demonstration honest: edit a field in the browser,
    send it, and watch the digest stop matching what the chain already holds.
    """
    body = request.get_json(silent=True) or {}
    record, tx = body.get("record"), body.get("tx")
    rpc = body.get("rpc") or os.getenv("RPC_URL", "memory")
    if not record or not tx:
        return jsonify(error="need both record and tx"), 400
    try:
        ok, msg, info = chain.verify(tx, record, rpc_url=rpc)
    except Exception as exc:                      # noqa: BLE001 - surfaced to UI
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 500
    return jsonify(ok=ok, message=msg, info=info)


@app.get("/api/config")
def api_config():
    from pipeline.search import build_backends
    return jsonify(
        engines=[{"name": b.name, "available": b.available()}
                 for b in build_backends()],
        rpc=os.getenv("RPC_URL", "memory"),
        threshold=DEFAULT_THRESHOLD,
        conf=DEFAULT_CONF,
        gated=bool(ACCESS_TOKEN),
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "127.0.0.1")
    if host != "127.0.0.1" and not ACCESS_TOKEN:
        raise SystemExit(
            "refusing to listen on " + host + " without ACCESS_TOKEN set.\n"
            "This service identifies faces, spends a funded key and burns a\n"
            "metered search quota. Set ACCESS_TOKEN, or bind to 127.0.0.1.")
    if ACCESS_TOKEN:
        print(f"  access token required (?token=... or X-Access-Token header)")
    print(f"  UI on http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)
