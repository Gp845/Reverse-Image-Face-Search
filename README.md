# HH Goa 2026 — Task 3: Face ID + Blockchain Verification

A pipeline that detects and encodes a face, finds a **real** matching post on the
web via reverse image search, proves the match is genuine by re-encoding the
retrieved image, and writes the result to a blockchain as a tamper-evident record.

```
probe image ──▶ YuNet detect ──▶ SFace encode (128-d)
                                      │
                                      ▼
                     ┌────────── reverse image search ──────────┐
                     │  SerpApi (Google Lens)   Google Vision   │
                     └──────────────────┬───────────────────────┘
                                        ▼
                            merge + dedupe candidates
                                        ▼
                    match-back: download each hit, re-encode,
                     accept only if cosine ≥ 0.363 vs probe
                                        ▼
                    sha256(record) ──▶ blockchain calldata
                                        ▼
                        verify_onchain.py (separate run)
```

## Why the match-back step exists

A search hit alone is just a URL you are *asserting* is the same person — which
is indistinguishable from hardcoding a result. So every candidate is downloaded,
run through the same detector and encoder, and accepted only when its face
actually matches the probe embedding above SFace's published operating point
(cosine ≥ 0.363). The score that lands on-chain is a measured quantity.

Two independent search engines run in parallel and the record states which
engine(s) found each hit. A URL corroborated by both is materially harder to
fake than a single opaque call.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
```

The YuNet detector is committed. Fetch the SFace encoder (38 MB):

```bash
mkdir -p models && curl -L -o models/sface.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

### Keys

| Backend | Env var | Free tier | Notes |
|---|---|---|---|
| SerpApi (Google Lens) | `SERPAPI_KEY` | 250 searches/mo | email signup, no card. Needs a public image URL, so the probe is uploaded to `0x0.st` first |
| Google Vision web detection | `GOOGLE_VISION_API_KEY` | 1,000 units/mo | needs GCP billing enabled; takes raw bytes, no upload step |

Either one alone is enough; the pipeline reports which are active and skips the rest.

## Run

```bash
.venv/bin/python run_pipeline.py path/to/probe.jpg
```

Useful flags: `--backend serpapi|vision|all`, `--conf 0.9` (detection floor),
`--threshold 0.363` (match strictness), `--social-only`, `--limit 25`, `--no-chain`.

Then re-verify as a separate command — this is the tamper-evidence demonstration:

```bash
.venv/bin/python verify_onchain.py
```

## Which blockchain

The record is canonicalised to deterministic JSON, hashed with SHA-256, and the
32-byte digest written into transaction calldata behind a 6-byte magic prefix.
No Solidity compiler is required, so the repo runs with nothing but `pip`.

**This submission anchors to Sepolia**, the Ethereum public testnet (chain id
`11155111`). That choice is deliberate: the transaction is public, so the claim
can be checked without trusting anything this repo prints. Every run emits a
`https://sepolia.etherscan.io/tx/...` link.

```
RPC_URL=https://ethereum-sepolia-rpc.publicnode.com
PRIVATE_KEY=<a throwaway testnet key, funded from any Sepolia faucet>
```

`RPC_URL` accepts any other JSON-RPC endpoint too:

- **Anvil / Ganache** (`http://127.0.0.1:8545`) — persistent local chain, no faucet.
- **`memory`** — an in-process EVM (eth-tester). Zero setup. It dies with the
  process, so it cannot demonstrate cross-process re-verification. Development only.

A run against a real chain starts with a pre-flight that confirms the node is
reachable and the key is funded, so it fails in two seconds rather than after
the search quota has already been spent.

Re-verification fetches the transaction, extracts the stored digest, recomputes
the digest from the local `record.json`, and compares. Any edit to the record
after anchoring fails the check.

## Known limitations

- **The probe must already be indexed.** If the face is not on the public web,
  every backend returns nothing and there is correctly nothing to anchor. This
  is a property of reverse image search, not a bug.
- **This is image search, not face search across the web.** The engines match
  the *image*; the face comparison is what confirms identity afterwards.
  A photo that has never been published will not be found even if the person
  has other photos online.
- **SerpApi uploads the probe to a third-party host** (`0x0.st`) to obtain the
  public URL it requires. Only run it on images you have the right to share.
  The Vision backend has no such step.
- **Face enhancement is deliberately excluded from the judged path.** GFPGAN
  reconstructs plausible detail rather than recovering true detail, so putting
  it upstream of either the search or the embedding comparison would undermine
  the "genuine match" claim. `enhance.py` is kept for inspecting low-res crops
  by eye; it is not importable on Python 3.13 because `basicsr` fails to build
  there, and it is not in `requirements.txt`.
- **Cosine 0.363** is OpenCV's published SFace threshold. It is a balanced
  operating point, not a zero-false-positive one; raise `--threshold` for a
  stricter claim.
- Only the single highest-confidence face in the probe is encoded. Crowd photos
  are not a supported input.

## Ethics

Run this on yourself, on a consenting teammate, or on a public figure. Do not
use it to identify or locate private individuals.

## Layout

```
run_pipeline.py             end-to-end CLI
verify_onchain.py           standalone re-verification
pipeline/encode.py          YuNet detect + SFace encode
pipeline/search.py          SerpApi + Vision behind one interface
pipeline/match.py           download, re-encode, compare
pipeline/chain.py           canonicalise, hash, anchor, verify
detect_and_crop_faces.py    standalone aligned-crop tool
enhance.py                  standalone GFPGAN tool (not in the judged path)
```
