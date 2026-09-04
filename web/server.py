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
import queue
import sys
import threading
import uuid
from contextlib import redirect_stdout

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import chain                                     # noqa: E402
from pipeline.core import PipelineError, run                   # noqa: E402
from pipeline.encode import COSINE_THRESHOLD                   # noqa: E402

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UPLOADS = os.path.join(ROOT, "out", "uploads")
SENTINEL = object()

app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
JOBS = {}
JOBS_LOCK = threading.Lock()


class LogSink(io.TextIOBase):
    """Collects emitted lines and fans them out to any listening browser.

    Doubles as a stdout replacement: the search backends print progress
    directly (upload attempts, per-engine errors), and that detail is worth
    showing rather than swallowing.
    """

    def __init__(self, job):
        self.job = job

    def write(self, text):
        if text and text.strip():
            for line in text.rstrip("\n").split("\n"):
                self.job["log"].append(line)
                self.job["queue"].put(line)
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
    finally:
        job["queue"].put(SENTINEL)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/run")
def api_run():
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return jsonify(error="no image uploaded"), 400

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
        "conf": float(form.get("conf", 0.9)),
        "threshold": float(form.get("threshold", COSINE_THRESHOLD)),
        "limit": int(form.get("limit", 25)),
        "social_only": form.get("social_only") == "true",
        "no_chain": form.get("no_chain") == "true",
        "rpc": form.get("rpc") or os.getenv("RPC_URL", "memory"),
    }

    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "status": "running", "log": [],
                        "queue": queue.Queue(), "out": out_dir,
                        "image": image_path, "result": None, "error": None}

    threading.Thread(target=_worker, args=(job_id, image_path, opts),
                     daemon=True).start()
    return jsonify(job_id=job_id)


@app.get("/api/log/<job_id>")
def api_log(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404

    def stream():
        # Replay what already happened, so a late or reconnecting browser sees
        # the whole run rather than joining midway.
        for line in list(job["log"]):
            yield f"data: {json.dumps(line)}\n\n"
        if job["status"] != "running":
            yield "event: end\ndata: {}\n\n"
            return
        while True:
            line = job["queue"].get()
            if line is SENTINEL:
                yield "event: end\ndata: {}\n\n"
                return
            yield f"data: {json.dumps(line)}\n\n"

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
def api_crop(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404
    path = os.path.join(job["out"], "probe_aligned.jpg")
    if not os.path.exists(path):
        return jsonify(error="no crop yet"), 404
    return send_file(path, mimetype="image/jpeg")


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
        threshold=COSINE_THRESHOLD,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"  UI on http://127.0.0.1:{port}   (local use only -- no auth)")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
