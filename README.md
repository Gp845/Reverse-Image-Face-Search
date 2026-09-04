# HH Goa 2026 — Task 3: Face ID + Blockchain Verification

A pipeline that detects and encodes a face, finds a **real** matching post on the
web via reverse image search, proves the match is genuine by re-encoding the
retrieved image, and writes the result to a blockchain as a tamper-evident record.

```
probe image ──▶ YuNet detect ──▶ SFace encode (128-d)
                                      │
                                      ▼
         ┌──────────── reverse image search (parallel) ────────────┐
         │  Google Lens   Google Reverse   Yandex   Bing  (Vision) │
         └────────────────────────┬───────────────────────────────┘
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

Two search engines run in parallel and the record states which engine(s) found
each hit. What makes that worth something is that **Google and Yandex crawl
independently** — a page both return is two separate indexes agreeing, not one
index consulted twice, and it is materially harder to fake than a single opaque
call. Yandex is also distinctly better at faces, and unlike Lens it returns the
real page URL and a full-resolution image rather than a thumbnail.

Four engines run **in parallel**: Google Lens, Google's classic reverse image
search, Yandex, and Bing. They disagree constantly, which is the point — on the
authors' own test probe they returned 88 unique candidates of which only 5 were
found by more than one engine, and the winning Instagram post was one of those 5.

All four reach the web through SerpApi, so the *vendor* is shared even though
the indexes are not. Google Vision web detection is implemented as a fifth,
fully independent backend and switches itself on the moment
`GOOGLE_VISION_API_KEY` is set. Microsoft's own Bing Search APIs were retired on
2025-08-11, so SerpApi is now the practical route to that index.

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
| SerpApi — Google Lens | `SERPAPI_KEY` | 250 searches/mo | email signup, no card. Needs a public image URL, so the probe is uploaded first. Costs 2 searches (exact + visual) |
| SerpApi — Google Reverse Image | `SERPAPI_KEY` | same quota | direct page links, no interstitial. Page-oriented, so often no thumbnail |
| SerpApi — Yandex images | `SERPAPI_KEY` | same quota | direct page URLs, full-res images. Best of the four on faces |
| SerpApi — Bing reverse image | `SERPAPI_KEY` | same quota | visually-similar rather than exact, so lower precision — the face match filters it |

One full run spends **5 SerpApi searches**, so the free 250/month is about
50 runs. `--backend <name>` restricts to a single engine while iterating.
| Google Vision web detection | `GOOGLE_VISION_API_KEY` | 1,000 units/mo | **billing must be enabled on the project** even within the free tier, or every call returns `403 BILLING_DISABLED`; takes raw bytes, no upload step |

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

## Web UI

```bash
.venv/bin/python web/server.py     # http://127.0.0.1:5000
```

Upload an image, watch the run stream line by line, and get the match plus its
Etherscan link. The record is shown in an editable box: change any character and
hit re-verify to watch the on-chain digest stop matching. That is the whole
tamper-evidence argument in one click.

The UI calls `pipeline.core.run` — the same function the CLI calls — so the two
cannot drift apart. Progress reaches the browser over Server-Sent Events, since
a run takes around 75 seconds and the log is most of what makes the result
credible.

**It is for local use.** There is no authentication, jobs are held in memory,
and the process has access to a funded private key. Do not expose it to a
network you do not control.

### Deploying it

A run exceeds the 60s function limit on Vercel's Hobby tier, and
`opencv-python` plus `sface.onnx` comes to roughly 130 MB against a 250 MB
bundle, so the pipeline does not belong in a serverless function. The shape
that works is a static or Next.js front end on Vercel talking to this server
running somewhere without a request timeout — a small VM, Fly, or Railway — with
the private key held only by that worker.

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
- **SerpApi uploads the probe to a third-party host** to obtain the public URL
  it requires. Only run it on images you have the right to share. The Vision
  backend has no such step. The uploader falls through `catbox.moe` →
  `uguu.se` → `x0.at` and checks that the returned URL actually serves image
  bytes, because a host that answers with an HTML viewer page looks like a
  success and then silently yields zero Lens results. The original single host
  (`0x0.st`) has since disabled uploads outright, which is why this is a chain
  rather than a constant.
- **Google Lens gives no direct page URLs.** Every result links through a
  `google.com/goto` interstitial whose payload is opaque (decoding it yields no
  printable text). `resolve_page_url` recovers the real address from the
  destination's `og:url`/`canonical`. In practice **~55% resolve**; the rest are
  sites that answer the proxied fetch with 403/406 or omit both tags. An
  unresolved candidate can still match on the face, so the record carries
  `url_resolved: false` rather than presenting a redirector as the post.
- **Google Vision web detection requires billing enabled** on the GCP project,
  even inside the 1,000 free units/month. Without it the API returns
  `403 PERMISSION_DENIED / BILLING_DISABLED` and the backend self-disables.
- **Face enhancement is deliberately excluded from the judged path.** GFPGAN
  reconstructs plausible detail rather than recovering true detail, so putting
  it upstream of either the search or the embedding comparison would undermine
  the "genuine match" claim. `enhance.py` is kept for inspecting low-res crops
  by eye; it is not importable on Python 3.13 because `basicsr` fails to build
  there, and it is not in `requirements.txt`.
- **Cross-engine corroboration is a bonus, not something to count on.** Lens
  and Yandex overlapped on zero pages in testing: 3 Lens hits and 40 Yandex
  hits merged to 43 unique. The reason is structural — the probe is uploaded to
  a throwaway host, so Lens `exact_matches` usually finds nothing for that URL
  and falls back to a short `visual_matches` list, while Yandex returns a deep
  one. `corroborated: true` therefore means something when it appears, but the
  face match, not agreement between engines, is what actually carries the claim.
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
web/server.py               local web UI (Flask + SSE)
web/static/index.html       the page
pipeline/core.py            the pipeline, shared by the CLI and the UI
pipeline/encode.py          YuNet detect + SFace encode
pipeline/search.py          SerpApi + Vision behind one interface
pipeline/match.py           download, re-encode, compare
pipeline/chain.py           canonicalise, hash, anchor, verify
detect_and_crop_faces.py    standalone aligned-crop tool
enhance.py                  standalone GFPGAN tool (not in the judged path)
```
