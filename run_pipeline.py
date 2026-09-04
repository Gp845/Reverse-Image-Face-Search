#!/usr/bin/env python3
"""
HH Goa 2026 Task 3 -- end-to-end pipeline, command line entry point.

  face scan  ->  reverse image search  ->  match-back  ->  blockchain anchor

The pipeline itself lives in pipeline/core.py so that this CLI and the web UI
run the same code. Every stage prints what it did, so an unedited screen
recording shows real work rather than a spinner.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from pipeline.core import PipelineError, run
from pipeline.encode import DEFAULT_CONF, DEFAULT_THRESHOLD, MIN_FACE_PX


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Face ID + blockchain verification pipeline")
    ap.add_argument("image", help="probe image containing the face to identify")
    ap.add_argument("--backend",
                    choices=["all", "serpapi", "greverse", "yandex", "bing", "vision"],
                    default="all")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF,
                    help="face detection confidence floor")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="cosine threshold for accepting a match "
                         "(SFace's published operating point is 0.363)")
    ap.add_argument("--limit", type=int, default=25, help="max candidates to download and check")
    ap.add_argument("--faces", type=int, default=1,
                    help="how many distinct faces to identify, largest first. "
                         "Each face costs a full set of searches, so 5 faces is "
                         "5x the quota of 1")
    ap.add_argument("--min-face", type=int, default=MIN_FACE_PX,
                    help=f"smallest face to encode, in pixels (default {MIN_FACE_PX}). "
                         "Set 0 to disable; embeddings degrade badly below ~48px")
    ap.add_argument("--social-only", action="store_true",
                    help="only check candidates on known social platforms")
    ap.add_argument("--rpc", default=os.getenv("RPC_URL", "memory"),
                    help="JSON-RPC url, or 'memory' for an in-process chain")
    ap.add_argument("--no-chain", action="store_true", help="stop before anchoring")
    ap.add_argument("--out", default="out", help="output directory")
    args = ap.parse_args()

    try:
        result = run(args.image, backend=args.backend, conf=args.conf,
                     threshold=args.threshold, limit=args.limit,
                     social_only=args.social_only, rpc=args.rpc,
                     no_chain=args.no_chain, out=args.out, faces=args.faces,
                     min_face=args.min_face, emit=print)
    except PipelineError as exc:
        sys.exit(f"\n{exc}")

    if result.get("anchor"):
        print("\n  Re-verify later with:")
        print(f"    python verify_onchain.py --rpc {result['rpc']} "
              f"--record {result['record_path']} --tx {result['anchor']['tx_hash']}")


if __name__ == "__main__":
    main()
