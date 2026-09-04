#!/usr/bin/env python3
"""Re-verify a local record against its on-chain digest, as a separate run."""

import argparse
import json
import os

from dotenv import load_dotenv

from pipeline import chain


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Verify a record against the blockchain")
    ap.add_argument("--record", default="out/record.json")
    ap.add_argument("--anchor", default="out/anchor.json",
                    help="used for the tx hash / rpc when --tx is not given")
    ap.add_argument("--tx")
    ap.add_argument("--rpc")
    args = ap.parse_args()

    record = json.load(open(args.record))
    tx, rpc = args.tx, args.rpc
    if (not tx or not rpc) and os.path.exists(args.anchor):
        a = json.load(open(args.anchor))
        tx = tx or a.get("tx_hash")
        rpc = rpc or a.get("rpc")
    if not tx:
        raise SystemExit("need --tx (or an out/anchor.json)")

    print(f"record : {args.record}")
    print(f"tx     : {tx}")
    print(f"rpc    : {rpc}\n")
    ok, msg, info = chain.verify(tx, record, rpc_url=rpc)
    if info:
        print(f"  on-chain sha256 : {info['onchain_sha256']}")
        print(f"  local    sha256 : {info['local_sha256']}")
        print(f"  block           : {info['block_number']}  chain_id {info['chain_id']}")
    print(f"\n{'VERIFIED' if ok else 'FAILED'} -- {msg}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
