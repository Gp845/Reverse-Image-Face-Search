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


def connect(rpc_url=None):
    """Connect to any JSON-RPC endpoint, or an in-process chain.

    rpc_url="memory" spins up an ephemeral EVM inside this process so the repo
    runs with no node and no faucet. It is deliberately NOT the default for a
    real submission: an in-memory chain dies with the process, so cross-process
    re-verification -- the thing the task actually asks you to demonstrate --
    needs a persistent node (Anvil) or a public testnet.
    """
    global _MEMORY_W3
    rpc_url = rpc_url or os.getenv("RPC_URL", "http://127.0.0.1:8545")
    if rpc_url == "memory":
        if _MEMORY_W3 is None:
            from web3 import EthereumTesterProvider
            _MEMORY_W3 = Web3(EthereumTesterProvider())
        return _MEMORY_W3
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    if not w3.is_connected():
        raise ConnectionError(f"no JSON-RPC at {rpc_url}")
    return w3


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


def anchor(record, rpc_url=None, private_key=None):
    """Write sha256(record) to chain. Returns a dict describing the anchor."""
    private_key = private_key or os.getenv("PRIVATE_KEY", "")
    if not private_key:
        raise ValueError("PRIVATE_KEY not set")
    w3 = connect(rpc_url)
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
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    chain_id = w3.eth.chain_id
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
