#!/usr/bin/env python3
"""
The pipeline itself, with output as a callback rather than prints.

run_pipeline.py and the web UI both call run() so there is exactly one
implementation of what the pipeline does. Progress arrives through `emit`, so a
terminal can print it and a browser can stream it without the two versions
drifting apart.

Failures raise PipelineError instead of calling sys.exit, because a web request
that ends the server process is not a failure mode worth having.
"""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2

from . import chain
from .encode import COSINE_THRESHOLD, FaceEncoder
from .match import verify_candidates
from .search import build_backends, merge, upload_for_public_url

ALL_ENGINES = ("serpapi", "greverse", "yandex", "bing", "vision")
NEEDS_PUBLIC_URL = {"serpapi", "greverse", "yandex", "bing"}


class PipelineError(Exception):
    """A run that stopped for a reportable reason, not a crash."""


def _candidate_dict(score, c):
    return {
        "page_url": c.page_url,
        "image_url": c.image_url,
        "domain": c.domain,
        "is_social": c.is_social,
        "url_resolved": c.resolved,
        "cosine": round(float(score), 6),
        "found_by": sorted(c.backends),
        "corroborated": c.corroborated,
    }


def run(image_path, *, backend="all", conf=0.9, threshold=COSINE_THRESHOLD,
        limit=25, social_only=False, rpc=None, no_chain=False, out="out",
        emit=print):
    """Run the full pipeline. Returns a dict describing everything it did."""
    rpc = rpc or os.getenv("RPC_URL", "memory")

    def rule(title):
        emit(f"\n{'='*66}\n  {title}\n{'='*66}")

    os.makedirs(out, exist_ok=True)
    result = {"rpc": rpc, "out": out}

    # Fail fast on an unreachable or unfunded chain, before spending a search.
    if not no_chain and rpc != "memory":
        rule("PRE-FLIGHT  chain reachable and funded")
        try:
            info = chain.preflight(rpc_url=rpc)
        except Exception as exc:
            raise PipelineError(f"chain pre-flight failed: {exc}") from exc
        for k, v in info.items():
            emit(f"  {k:13s} {v}")
        result["preflight"] = info

    # ------------------------------------------------------------------ stage 1
    rule("STAGE 1/4  face detection + encoding")
    enc = FaceEncoder(conf=conf)
    image = cv2.imread(image_path)
    if image is None:
        raise PipelineError(f"could not read image: {image_path}")
    embedding, row = enc.encode_primary(image)
    if embedding is None:
        raise PipelineError(
            f"no face detected at conf>={conf}. Try a lower confidence floor.")
    x, y, w, h = row[0:4]
    emit(f"  detector : YuNet   confidence {float(row[-1]):.3f}")
    emit(f"  face box : x={int(x)} y={int(y)} w={int(w)} h={int(h)}")
    emit(f"  encoder  : SFace   {len(embedding)}-d embedding")
    crop = enc.rec.alignCrop(image, row)
    crop_path = os.path.join(out, "probe_aligned.jpg")
    cv2.imwrite(crop_path, crop)
    emit(f"  aligned  : {crop_path}  {crop.shape[1]}x{crop.shape[0]}")
    result["crop_path"] = crop_path
    result["detection"] = {"confidence": round(float(row[-1]), 4),
                           "box": [int(x), int(y), int(w), int(h)]}

    # ------------------------------------------------------------------ stage 2
    rule("STAGE 2/4  reverse image search")
    wanted = set(ALL_ENGINES) if backend == "all" else {backend}
    backends = [b for b in build_backends() if b.name in wanted]
    active = [b for b in backends if b.available()]
    for b in backends:
        emit(f"  {b.name:9s} {'ready' if b.available() else 'SKIPPED (no api key set)'}")
    if not active:
        raise PipelineError(
            "no search backend configured. Set SERPAPI_KEY and/or "
            "GOOGLE_VISION_API_KEY in .env (see .env.example).")

    public_url = None
    if any(b.name in NEEDS_PUBLIC_URL for b in active):
        emit("\n  uploading probe for a public URL (every engine but Vision needs one)...")
        public_url = upload_for_public_url(image_path)
        emit(f"  probe url: {public_url or 'FAILED - those backends will be skipped'}")
    result["public_url"] = public_url

    runnable = [b for b in active
                if not (b.name in NEEDS_PUBLIC_URL and not public_url)]
    emit(f"\n  querying {len(runnable)} engine(s) in parallel...")

    def _run(b):
        try:
            return b.name, b.search(image_path, public_url=public_url)
        except Exception as exc:
            emit(f"    [{b.name}] {exc}")
            return b.name, []

    raw = []
    per_engine = {}
    with ThreadPoolExecutor(max_workers=max(1, len(runnable))) as pool:
        for name, hits in pool.map(_run, runnable):
            emit(f"  {name:9s} {len(hits)} raw hits")
            per_engine[name] = len(hits)
            raw.extend(hits)

    candidates = merge(raw)
    social = [c for c in candidates if c.is_social]
    corrob = [c for c in candidates if c.corroborated]
    emit(f"\n  merged   : {len(candidates)} unique  |  {len(social)} social  "
         f"|  {len(corrob)} corroborated by >1 engine")
    result["search"] = {"per_engine": per_engine, "unique": len(candidates),
                        "social": len(social), "corroborated": len(corrob)}
    if not candidates:
        raise PipelineError(
            "no search results. The probe is likely not present in any index.")

    # ------------------------------------------------------------------ stage 3
    rule("STAGE 3/4  match-back verification")
    emit("  downloading each candidate and re-encoding its face;")
    emit(f"  accepting only cosine >= {threshold:.3f}\n")
    confirmed = verify_candidates(enc, embedding, candidates, threshold=threshold,
                                  limit=limit, social_only=social_only,
                                  emit=emit)
    if not confirmed:
        raise PipelineError(
            "no candidate passed face verification -- nothing worth anchoring.")
    score, best = confirmed[0]
    emit(f"\n  {len(confirmed)} verified match(es). Best:")
    emit(f"    page   : {best.page_url}")
    emit(f"    image  : {best.image_url}")
    emit(f"    domain : {best.domain}   social={best.is_social}")
    if not best.resolved:
        emit("    NOTE   : page URL is still a Google redirector -- the "
             "destination blocked resolution.")
    emit(f"    cosine : {score:.4f}")
    emit(f"    found by: {', '.join(sorted(best.backends))}")

    record = {
        "schema": "hhgoa-task3/v1",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probe_image": os.path.basename(image_path),
        "probe_sha256": hashlib.sha256(open(image_path, "rb").read()).hexdigest(),
        "match": _candidate_dict(score, best),
        "method": {"detector": "YuNet", "encoder": "SFace",
                   "threshold": threshold, "embedding_dim": len(embedding)},
    }
    rec_path = os.path.join(out, "record.json")
    with open(rec_path, "w") as fh:
        json.dump(record, fh, indent=2)
    emit(f"\n  record written: {rec_path}")
    result["record"] = record
    result["record_path"] = rec_path
    result["matches"] = [_candidate_dict(s, c) for s, c in confirmed]

    # ------------------------------------------------------------------ stage 4
    if no_chain:
        emit("\n  no-chain requested, stopping before anchor.")
        return result

    rule("STAGE 4/4  blockchain anchor")
    emit(f"  rpc     : {rpc}")
    if rpc == "memory":
        w3 = chain.connect("memory")
        key = "0x" + "11" * 31 + "12"
        acct = w3.eth.account.from_key(key)
        w3.eth.send_transaction({"from": w3.eth.accounts[0], "to": acct.address,
                                 "value": w3.to_wei(1, "ether")})
        os.environ["PRIVATE_KEY"] = key
        emit("  NOTE: in-process chain. It dies with this process, so use a")
        emit("        persistent node or testnet for a real demonstration.")
    anchored = chain.anchor(record, rpc_url=rpc, emit=emit)
    for k, v in anchored.items():
        if k not in ("tx_hash", "explorer", "block_number"):
            emit(f"  {k:13s} {v}")
    anchored["rpc"] = rpc
    anchor_path = os.path.join(out, "anchor.json")
    with open(anchor_path, "w") as fh:
        json.dump(anchored, fh, indent=2)
    result["anchor"] = anchored
    result["anchor_path"] = anchor_path

    ok, msg, _info = chain.verify(anchored["tx_hash"], record, rpc_url=rpc)
    emit(f"\n  immediate re-verify: {'PASS' if ok else 'FAIL'} -- {msg}")
    result["verified"] = ok
    return result
