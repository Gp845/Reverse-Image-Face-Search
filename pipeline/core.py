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
from .encode import (DEFAULT_CONF, DEFAULT_THRESHOLD, GROUP_MAX_SIDE,
                     MAX_SIDE, MIN_FACE_PX, FaceEncoder, downscale,
                     preview_crop, scale_row)
from .match import verify_candidates
from .search import build_backends, merge, upload_for_public_url

ALL_ENGINES = ("serpapi", "greverse", "yandex", "bing", "vision")
NEEDS_PUBLIC_URL = {"serpapi", "greverse", "yandex", "bing"}


class PipelineError(Exception):
    """A run that stopped for a reportable reason, not a crash."""


def _candidate_dict(score, c):
    return {
        "page_url": c.page_url,
        # Kept because it is the only description of the post that survives:
        # Instagram and TikTok serve bots a page with the Open Graph tags
        # removed, so the engine's own title and source line is all there is.
        "title": (c.title or "")[:300],
        "source": (c.source or "")[:200],
        "image_url": c.image_url,
        "domain": c.domain,
        "is_social": c.is_social,
        "url_resolved": c.resolved,
        "cosine": round(float(score), 6),
        "found_by": sorted(c.backends),
        "corroborated": c.corroborated,
    }


def _search(probe_path, active, emit):
    """Upload one probe image and query every engine. Returns merged candidates."""
    public_url = None
    if any(b.name in NEEDS_PUBLIC_URL for b in active):
        emit("  uploading probe for a public URL (every engine but Vision needs one)...")
        public_url = upload_for_public_url(probe_path)
        emit(f"  probe url: {public_url or 'FAILED - those backends will be skipped'}")

    runnable = [b for b in active
                if not (b.name in NEEDS_PUBLIC_URL and not public_url)]
    emit(f"  querying {len(runnable)} engine(s) in parallel...")

    def _one(b):
        try:
            return b.name, b.search(probe_path, public_url=public_url)
        except Exception as exc:
            emit(f"    [{b.name}] {exc}")
            return b.name, []

    raw, per_engine = [], {}
    with ThreadPoolExecutor(max_workers=max(1, len(runnable))) as pool:
        for name, hits in pool.map(_one, runnable):
            emit(f"  {name:9s} {len(hits)} raw hits")
            per_engine[name] = len(hits)
            raw.extend(hits)

    candidates = merge(raw)
    social = [c for c in candidates if c.is_social]
    corrob = [c for c in candidates if c.corroborated]
    emit(f"  merged   : {len(candidates)} unique  |  {len(social)} social  "
         f"|  {len(corrob)} corroborated by >1 engine")
    return candidates, {"per_engine": per_engine, "unique": len(candidates),
                        "social": len(social), "corroborated": len(corrob),
                        "public_url": public_url}


def run(image_path, *, backend="all", conf=DEFAULT_CONF, threshold=DEFAULT_THRESHOLD,
        limit=25, social_only=False, rpc=None, no_chain=False, out="out",
        faces=1, min_face=MIN_FACE_PX, emit=print):
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
    original = image
    full_h, full_w = original.shape[:2]
    detect_cap = GROUP_MAX_SIDE if faces > 1 else MAX_SIDE
    small = downscale(original, detect_cap)
    factor = full_w / float(small.shape[1])
    if factor != 1.0:
        emit(f"  resized  : {full_w}x{full_h} -> {small.shape[1]}x{small.shape[0]} "
             "for detection only (its cost grows far faster than pixel count)")

    # Detect on the small copy to bound memory, then encode from the original.
    # Faces in a group photo are small to begin with, and downscaling before
    # encoding pushed them under the minimum size -- on one 3-face photo it
    # left a single usable face.
    detected = [scale_row(r, factor) for r in enc.detect(small)]
    image = original
    found = enc.encode_rows(image, detected, max_faces=faces, min_size=min_face)
    if not found:
        raise PipelineError(
            f"no usable face at conf>={conf} and >={min_face}px. "
            f"{len(detected)} raw detection(s); try a lower confidence floor.")

    # How many distinct faces the image actually holds, regardless of --faces.
    available = len(enc.encode_rows(image, detected, min_size=min_face))

    emit(f"  detector : YuNet   {len(detected)} detection(s), "
         f"{len(found)} distinct face(s) kept (largest first)")
    emit(f"  encoder  : SFace   {len(found[0][0])}-d embedding each")
    if len(detected) > len(found):
        emit(f"  dropped  : {len(detected) - len(found)} below {min_face}px, "
             "duplicates, or beyond the requested face count")
    if available > len(found):
        emit(f"\n  NOTE: this image has {available} distinct faces but only "
             f"{len(found)} was requested.")
        emit(f"        Use --faces {min(available, 5)} to identify more "
             "(each face costs its own set of searches).")

    face_infos = []
    for i, (emb, row) in enumerate(found):
        x, y, w, h = (int(v) for v in row[0:4])
        aligned = enc.rec.alignCrop(image, row)
        prev = preview_crop(image, row)
        suffix = "" if len(found) == 1 else f"_face{i + 1}"
        a_path = os.path.join(out, f"probe{suffix}_aligned.jpg")
        p_path = os.path.join(out, f"probe{suffix}_preview.jpg")
        cv2.imwrite(a_path, aligned)
        cv2.imwrite(p_path, prev)
        emit(f"  face {i + 1:<2}  conf {float(row[-1]):.3f}  box {w}x{h} at ({x},{y})"
             f"  -> {os.path.basename(p_path)}")
        face_infos.append({"index": i + 1, "confidence": round(float(row[-1]), 4),
                           "box": [x, y, w, h], "embedding": emb,
                           "aligned_path": a_path, "preview_path": p_path})

    # Back-compat: single-face callers still expect these keys.
    result["crop_path"] = face_infos[0]["aligned_path"]
    result["preview_path"] = face_infos[0]["preview_path"]
    result["detection"] = {"confidence": face_infos[0]["confidence"],
                           "box": face_infos[0]["box"]}
    result["faces_detected"] = len(detected)
    result["faces_processed"] = len(face_infos)

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

    # One face: search the whole frame, which is what has always been searched
    # and what finds the original post. Several faces: search each face's crop,
    # because the full frame identifies the photograph while a crop identifies
    # a person, and every face would otherwise get identical results.
    per_face_probe = len(face_infos) > 1
    if per_face_probe:
        emit(f"\n  {len(face_infos)} faces -> searching each face crop separately "
             f"({len(active)} engines x {len(face_infos)} faces)")

    # ------------------------------------------------------------------ stage 3
    for fi in face_infos:
        probe = fi["preview_path"] if per_face_probe else image_path
        if per_face_probe:
            emit(f"\n  --- face {fi['index']}/{len(face_infos)} "
                 f"({fi['box'][2]}x{fi['box'][3]}px) ---")
        cands, stats = _search(probe, active, emit)
        fi["search"] = stats
        if not cands:
            emit("  no search results for this face.")
            fi["matches"] = []
            continue
        emit(f"  match-back: downloading candidates, accepting cosine >= {threshold:.3f}")
        confirmed = verify_candidates(enc, fi["embedding"], cands, threshold=threshold,
                                      limit=limit, social_only=social_only, emit=emit)
        fi["matches"] = [_candidate_dict(sc, c) for sc, c in confirmed]
        if confirmed:
            sc, best = confirmed[0]
            emit(f"  face {fi['index']}: {len(confirmed)} match(es), best "
                 f"{best.domain} cos={sc:.4f}")
        else:
            emit(f"  face {fi['index']}: no candidate passed face verification")

    identified = [f for f in face_infos if f["matches"]]
    if not identified:
        raise PipelineError(
            "no candidate passed face verification -- nothing worth anchoring.")

    rule("RESULTS")
    for fi in face_infos:
        if fi["matches"]:
            m = fi["matches"][0]
            emit(f"  face {fi['index']}  cos {m['cosine']:.4f}  {m['domain']}")
            emit(f"           {m['page_url']}")
            if m["source"]:
                emit(f"           {m['source']}")
        else:
            emit(f"  face {fi['index']}  no match")

    # The strongest identification across all faces leads the record.
    record_matches = 10
    identified.sort(key=lambda f: f["matches"][0]["cosine"], reverse=True)
    lead = identified[0]

    record = {
        "schema": "hhgoa-task3/v2",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probe_image": os.path.basename(image_path),
        "probe_sha256": hashlib.sha256(open(image_path, "rb").read()).hexdigest(),
        "faces_detected": len(detected),
        "faces_processed": len(face_infos),
        # Every face is anchored, not just the strongest. `match` is the
        # single best identification across the photo and stays for
        # single-face callers; `faces` is the actual result for a group.
        # Each face carries its ranked matches rather than only its best --
        # the record's size is irrelevant on chain, where all that lands is a
        # 38-byte hash of it however long it grows.
        "match": lead["matches"][0],
        "faces": [{"index": f["index"], "box": f["box"],
                   "confidence": f["confidence"],
                   "match": f["matches"][0] if f["matches"] else None,
                   "match_count": len(f["matches"]),
                   "matches": f["matches"][:record_matches]}
                  for f in face_infos],
        "method": {"detector": "YuNet", "encoder": "SFace",
                   "threshold": threshold, "min_face_px": MIN_FACE_PX,
                   "embedding_dim": len(lead["embedding"])},
    }
    rec_path = os.path.join(out, "record.json")
    with open(rec_path, "w") as fh:
        json.dump(record, fh, indent=2)
    emit(f"\n  record written: {rec_path}")
    result["record"] = record
    result["record_path"] = rec_path
    result["matches"] = lead["matches"]
    result["search"] = lead["search"]
    result["public_url"] = lead["search"].get("public_url")
    result["face_results"] = [{k: v for k, v in f.items() if k != "embedding"}
                              for f in face_infos]

    # ------------------------------------------------------------------ stage 4
    if no_chain:
        emit("\n  no-chain requested, stopping before anchor.")
        return result

    rule("STAGE 4/4  blockchain anchor")
    anchored_faces = sum(1 for f in record["faces"] if f["match"])
    anchored_hits = sum(len(f["matches"]) for f in record["faces"])
    emit(f"  anchoring : {len(record['faces'])} face(s), {anchored_faces} identified, "
         f"{anchored_hits} ranked match(es) -- all of it inside one hash")
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
