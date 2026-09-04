#!/usr/bin/env python3
"""
HH Goa 2026 Task 3 -- end-to-end pipeline.

  face scan  ->  reverse image search  ->  match-back  ->  blockchain anchor

Every stage prints what it did so an unedited screen recording shows the real
work rather than a spinner.
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
from dotenv import load_dotenv

from pipeline import chain
from pipeline.encode import COSINE_THRESHOLD, FaceEncoder
from pipeline.match import verify_candidates
from pipeline.search import build_backends, merge, upload_for_public_url


def rule(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Face ID + blockchain verification pipeline")
    ap.add_argument("image", help="probe image containing the face to identify")
    ap.add_argument("--backend",
                    choices=["all", "serpapi", "greverse", "yandex", "bing", "vision"],
                    default="all")
    ap.add_argument("--conf", type=float, default=0.9, help="face detection confidence floor")
    ap.add_argument("--threshold", type=float, default=COSINE_THRESHOLD,
                    help="cosine threshold for accepting a match")
    ap.add_argument("--limit", type=int, default=25, help="max candidates to download and check")
    ap.add_argument("--social-only", action="store_true",
                    help="only check candidates on known social platforms")
    ap.add_argument("--rpc", default=os.getenv("RPC_URL", "memory"),
                    help="JSON-RPC url, or 'memory' for an in-process chain")
    ap.add_argument("--no-chain", action="store_true", help="stop before anchoring")
    ap.add_argument("--out", default="out", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Fail fast on an unreachable or unfunded chain, before spending a search.
    if not args.no_chain and args.rpc != "memory":
        rule("PRE-FLIGHT  chain reachable and funded")
        try:
            for k, v in chain.preflight(rpc_url=args.rpc).items():
                print(f"  {k:13s} {v}")
        except Exception as exc:
            sys.exit(f"  chain pre-flight failed: {exc}")

    # ---------------------------------------------------------------- stage 1
    rule("STAGE 1/4  face detection + encoding")
    enc = FaceEncoder(conf=args.conf)
    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"could not read image: {args.image}")
    embedding, row = enc.encode_primary(image)
    if embedding is None:
        sys.exit(f"no face detected at conf>={args.conf}. Try a lower --conf.")
    x, y, w, h = row[0:4]
    print(f"  detector : YuNet   confidence {float(row[-1]):.3f}")
    print(f"  face box : x={int(x)} y={int(y)} w={int(w)} h={int(h)}")
    print(f"  encoder  : SFace   {len(embedding)}-d embedding")
    crop = enc.rec.alignCrop(image, row)
    crop_path = os.path.join(args.out, "probe_aligned.jpg")
    cv2.imwrite(crop_path, crop)
    print(f"  aligned  : {crop_path}  {crop.shape[1]}x{crop.shape[0]}")

    # ---------------------------------------------------------------- stage 2
    rule("STAGE 2/4  reverse image search")
    wanted = ({"serpapi", "greverse", "yandex", "bing", "vision"}
              if args.backend == "all" else {args.backend})
    backends = [b for b in build_backends() if b.name in wanted]
    active = [b for b in backends if b.available()]
    for b in backends:
        print(f"  {b.name:9s} {'ready' if b.available() else 'SKIPPED (no api key set)'}")
    if not active:
        sys.exit("\nno search backend configured. Set SERPAPI_KEY and/or "
                 "GOOGLE_VISION_API_KEY in .env (see .env.example).")

    needs_url = {"serpapi", "greverse", "yandex", "bing"}
    public_url = None
    if any(b.name in needs_url for b in active):
        print("\n  uploading probe for a public URL (every engine but Vision needs one)...")
        public_url = upload_for_public_url(args.image)
        print(f"  probe url: {public_url or 'FAILED - those backends will be skipped'}")

    runnable = [b for b in active if not (b.name in needs_url and not public_url)]
    print(f"\n  querying {len(runnable)} engine(s) in parallel...")

    def _run(b):
        try:
            return b.name, b.search(args.image, public_url=public_url)
        except Exception as exc:
            print(f"    [{b.name}] {exc}")
            return b.name, []

    raw = []
    with ThreadPoolExecutor(max_workers=max(1, len(runnable))) as pool:
        for name, hits in pool.map(_run, runnable):
            print(f"  {name:9s} {len(hits)} raw hits")
            raw.extend(hits)

    candidates = merge(raw)
    social = [c for c in candidates if c.is_social]
    corrob = [c for c in candidates if c.corroborated]
    print(f"\n  merged   : {len(candidates)} unique  |  {len(social)} social  "
          f"|  {len(corrob)} corroborated by >1 engine")
    if not candidates:
        sys.exit("\nno search results. The probe is likely not present in either index.")

    # ---------------------------------------------------------------- stage 3
    rule("STAGE 3/4  match-back verification")
    print("  downloading each candidate and re-encoding its face;")
    print(f"  accepting only cosine >= {args.threshold:.3f}\n")
    confirmed = verify_candidates(enc, embedding, candidates,
                                  threshold=args.threshold, limit=args.limit,
                                  social_only=args.social_only)
    if not confirmed:
        sys.exit("\nno candidate passed face verification -- nothing worth anchoring.")
    score, best = confirmed[0]
    print(f"\n  {len(confirmed)} verified match(es). Best:")
    print(f"    page   : {best.page_url}")
    print(f"    image  : {best.image_url}")
    print(f"    domain : {best.domain}   social={best.is_social}")
    if not best.resolved:
        print("    NOTE   : page URL is still a Google redirector -- the "
              "destination blocked resolution.")
    print(f"    cosine : {score:.4f}")
    print(f"    found by: {', '.join(sorted(best.backends))}")

    record = {
        "schema": "hhgoa-task3/v1",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probe_image": os.path.basename(args.image),
        "probe_sha256": hashlib.sha256(open(args.image, "rb").read()).hexdigest(),
        "match": {
            "page_url": best.page_url,
            "image_url": best.image_url,
            "domain": best.domain,
            "is_social": best.is_social,
            "url_resolved": best.resolved,
            "cosine": round(float(score), 6),
            "found_by": sorted(best.backends),
            "corroborated": best.corroborated,
        },
        "method": {"detector": "YuNet", "encoder": "SFace",
                   "threshold": args.threshold, "embedding_dim": len(embedding)},
    }
    rec_path = os.path.join(args.out, "record.json")
    with open(rec_path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\n  record written: {rec_path}")

    # ---------------------------------------------------------------- stage 4
    if args.no_chain:
        print("\n  --no-chain given, stopping before anchor.")
        return
    rule("STAGE 4/4  blockchain anchor")
    print(f"  rpc     : {args.rpc}")
    if args.rpc == "memory":
        w3 = chain.connect("memory")
        key = "0x" + "11" * 31 + "12"
        acct = w3.eth.account.from_key(key)
        w3.eth.send_transaction({"from": w3.eth.accounts[0], "to": acct.address,
                                 "value": w3.to_wei(1, "ether")})
        os.environ["PRIVATE_KEY"] = key
        print("  NOTE: in-process chain. It dies with this process, so use a")
        print("        persistent node or testnet for the real recording.")
    anchored = chain.anchor(record, rpc_url=args.rpc)
    for k, v in anchored.items():
        print(f"  {k:13s} {v}")
    anchored["rpc"] = args.rpc
    with open(os.path.join(args.out, "anchor.json"), "w") as fh:
        json.dump(anchored, fh, indent=2)

    ok, msg, info = chain.verify(anchored["tx_hash"], record, rpc_url=args.rpc)
    print(f"\n  immediate re-verify: {'PASS' if ok else 'FAIL'} -- {msg}")
    print(f"\n  Re-verify later with:")
    print(f"    python verify_onchain.py --rpc {args.rpc} "
          f"--record {rec_path} --tx {anchored['tx_hash']}")


if __name__ == "__main__":
    main()
