#!/usr/bin/env python3
"""
Blockchain anchoring + re-verification.

The record is canonicalised to deterministic JSON, hashed with SHA-256, and the
digest written into the calldata of a transaction. No Solidity compiler is
needed, which keeps the repo runnable by a judge with nothing but pip.

Tamper-evidence: the digest is fixed at the block it landed in. Re-verification
recomputes the digest from the local record and compares it against what the
chain actually holds, so any edit to the record after the fact fails the check.

Works against any JSON-RPC endpoint -- a local Anvil/Ganache node for
development, or a public testnet for a demo a judge can open in a block explorer.
"""

import hashlib
import json
import os
import time

from web3 import Web3

MAGIC = b"HHGOA\x01"

# Block explorers, so the run prints a link a judge can open independently
# instead of asking them to take the tx hash on faith.
EXPLORERS = {
    1: "https://etherscan.io/tx/",
    11155111: "https://sepolia.etherscan.io/tx/",
    17000: "https://holesky.etherscan.io/tx/",
}


def _0x(value):
    """Normalise a hash to 0x-prefixed hex.

    hexbytes 2.x dropped the 0x prefix from .hex(), so an unprefixed string
    reaches the node and a real JSON-RPC endpoint rejects it (eth-tester does
    not, which is exactly how this hides during local testing).
    """
    if isinstance(value, str):
        return value if value.startswith("0x") else "0x" + value
    return "0x" + bytes(value).hex()


def canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def digest(record):
    return hashlib.sha256(canonical(record)).digest()


_MEMORY_W3 = None


def connect(rpc_url=None, attempts=4, emit=None):
    """Connect to any JSON-RPC endpoint, or an in-process chain.

    rpc_url="memory" spins up an ephemeral EVM inside this process so the repo
    runs with no node and no faucet. It is deliberately NOT the default for a
    real submission: an in-memory chain dies with the process, so cross-process
    re-verification -- the thing the task actually asks you to demonstrate --
    needs a persistent node (Anvil) or a public testnet.

    Retries before giving up. A single failed liveness check used to abort the
    run, and stage 4 is the worst possible place for that: the search quota is
    already spent and the record is already written, so a momentary DNS or
    network blip threw away a minute of real work and, in a single-take
    recording, the take with it. Public endpoints are exactly the kind that
    blip.
    """
    global _MEMORY_W3
    rpc_url = rpc_url or os.getenv("RPC_URL", "http://127.0.0.1:8545")
    if rpc_url == "memory":
        if _MEMORY_W3 is None:
            from web3 import EthereumTesterProvider
            _MEMORY_W3 = Web3(EthereumTesterProvider())
        return _MEMORY_W3
    say = emit or (lambda _m: None)
    last = None
    for attempt in range(1, attempts + 1):
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
        try:
            if w3.is_connected():
                if attempt > 1:
                    say(f"  rpc reachable on attempt {attempt}")
                return w3
            last = "is_connected() was false"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            delay = 2 ** (attempt - 1)
            say(f"  rpc unreachable ({last}), retrying in {delay}s "
                f"[{attempt}/{attempts}]")
            time.sleep(delay)
    raise ConnectionError(f"no JSON-RPC at {rpc_url} after {attempts} attempts: {last}")


def memory_account(w3):
    """A pre-funded key for the in-process chain."""
    return w3.eth.account.from_key(
        "0x" + "11" * 31 + "12"), w3.eth.accounts[0]


def preflight(rpc_url=None, private_key=None):
    """Check the chain is reachable and the key is funded, before doing work.

    Called at the top of a run so a one-take recording fails here -- in two
    seconds, with a clear message -- rather than after the search quota has
    already been spent.
    """
    private_key = private_key or os.getenv("PRIVATE_KEY", "")
    if not private_key:
        raise ValueError("PRIVATE_KEY not set (see .env.example)")
    w3 = connect(rpc_url)
    acct = w3.eth.account.from_key(private_key)
    balance = w3.eth.get_balance(acct.address)
    if balance == 0:
        raise ValueError(
            f"{acct.address} has 0 balance on chain {w3.eth.chain_id}. "
            "Fund it from a faucet before running.")
    return {
        "chain_id": w3.eth.chain_id,
        "address": acct.address,
        "balance_eth": float(w3.from_wei(balance, "ether")),
        "block_number": w3.eth.block_number,
    }


def anchor(record, rpc_url=None, private_key=None, emit=None):
    """Write sha256(record) to chain. Returns a dict describing the anchor.

    `emit` receives progress. The transaction hash is final the moment it is
    broadcast, so it is reported immediately rather than after inclusion --
    Sepolia produces a block every ~12s and where a broadcast lands in that
    cycle swings the wait between about 1 and 13 seconds. Without this the run
    simply goes quiet for an unpredictable stretch and looks stalled.
    """
    say = emit or (lambda _m: None)
    private_key = private_key or os.getenv("PRIVATE_KEY", "")
    if not private_key:
        raise ValueError("PRIVATE_KEY not set")
    w3 = connect(rpc_url, emit=say)
    acct = w3.eth.account.from_key(private_key)
    payload = MAGIC + digest(record)

    tx = {
        "from": acct.address,
        "to": acct.address,          # anchor to self; the calldata is the point
        "value": 0,
        "data": "0x" + payload.hex(),
        "nonce": w3.eth.get_transaction_count(acct.address),
        "chainId": w3.eth.chain_id,
    }
    tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
    try:
        base = w3.eth.get_block("latest")["baseFeePerGas"]
        tip = w3.to_wei(1.5, "gwei")
        tx["maxPriorityFeePerGas"] = tip
        tx["maxFeePerGas"] = base * 2 + tip
    except (KeyError, TypeError):
        tx["gasPrice"] = w3.eth.gas_price

    signed = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    chain_id = w3.eth.chain_id
    tx_hex = _0x(tx_hash)
    say(f"  submitted     {tx_hex}")
    if chain_id in EXPLORERS:
        say(f"  explorer      {EXPLORERS[chain_id] + tx_hex}")

    # Poll rather than wait_for_transaction_receipt so the wait can be narrated.
    # Heartbeat on elapsed time since the last one, not on exact multiples: each
    # iteration costs an RPC round trip plus the sleep, so a counter stepping
    # ~2s at a time skips whichever multiple it happens to jump over.
    started = time.time()
    deadline = started + 300
    next_beat = started + 5.0
    receipt = None
    while True:
        try:
            receipt = w3.eth.get_transaction_receipt(tx_hash)
        except Exception:
            receipt = None
        if receipt is not None:
            break
        now = time.time()
        if now > deadline:
            raise TimeoutError(f"{tx_hex} not mined within 300s")
        if now >= next_beat:
            say(f"  ...still pending after {int(now - started)}s "
                f"(a Sepolia block is ~12s)")
            next_beat = now + 5.0
        time.sleep(1)
    say(f"  mined         block {receipt['blockNumber']}")
    tx_hash_hex = _0x(receipt["transactionHash"])
    anchored = {
        "tx_hash": tx_hash_hex,
        "block_number": receipt["blockNumber"],
        "chain_id": chain_id,
        "from": acct.address,
        "sha256": digest(record).hex(),
    }
    if chain_id in EXPLORERS:
        anchored["explorer"] = EXPLORERS[chain_id] + tx_hash_hex
    return anchored


def verify(tx_hash, record, rpc_url=None):
    """Re-verify a local record against what the chain actually stores."""
    w3 = connect(rpc_url)
    tx = w3.eth.get_transaction(_0x(tx_hash))
    data = bytes(tx["input"])
    if not data.startswith(MAGIC):
        return False, "not an HHGOA anchor transaction", None
    onchain = data[len(MAGIC):len(MAGIC) + 32]
    local = digest(record)
    receipt = w3.eth.get_transaction_receipt(_0x(tx_hash))
    info = {
        "onchain_sha256": onchain.hex(),
        "local_sha256": local.hex(),
        "block_number": receipt["blockNumber"],
        "chain_id": w3.eth.chain_id,
    }
    if info["chain_id"] in EXPLORERS:
        info["explorer"] = EXPLORERS[info["chain_id"]] + _0x(tx_hash)
    if onchain == local:
        return True, "record matches on-chain digest", info
    return False, "RECORD TAMPERED - digest mismatch", info
