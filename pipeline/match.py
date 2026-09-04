#!/usr/bin/env python3
"""
Match-back verification.

A search hit on its own is just a URL we are asserting is the same person.
This step downloads each candidate image, runs the same detector+encoder over
it, and only accepts the hit when the face inside it actually matches the probe
embedding above SFace's published threshold. That is what makes the on-chain
record mean something.
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

from .encode import COSINE_THRESHOLD, cosine

MAX_BYTES = 12 * 1024 * 1024
FETCH_WORKERS = 8


def fetch_image(url, timeout=30):
    try:
        r = requests.get(url, timeout=timeout, stream=True,
                         headers={"User-Agent": "Mozilla/5.0 (hhgoa-task3)"})
        r.raise_for_status()
        buf = b""
        for chunk in r.iter_content(65536):
            buf += chunk
            if len(buf) > MAX_BYTES:
                return None
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def verify_candidates(encoder, probe_embedding, candidates, threshold=COSINE_THRESHOLD,
                      limit=25, social_only=False, verbose=True,
                      workers=FETCH_WORKERS, emit=print):
    """Return candidates whose face genuinely matches the probe, best score first.

    Downloads run concurrently because they are pure I/O and dominate the wall
    clock, but the encoding stays on this thread: the OpenCV detector and
    recognizer carry per-instance state (input size, last frame), so sharing one
    across threads races. Scoring in candidate order also keeps the printed log
    deterministic, which matters when the run is being screen-recorded.
    """
    queued = []
    for c in candidates:
        if len(queued) >= limit:
            break
        if social_only and not c.is_social:
            continue
        url = c.image_url or c.page_url
        if url and url.lower().startswith("http"):
            queued.append((c, url))

    if not queued:
        return []

    with ThreadPoolExecutor(max_workers=min(workers, len(queued))) as pool:
        images = list(pool.map(lambda q: fetch_image(q[1]), queued))

    confirmed = []
    for n, ((c, _url), img) in enumerate(zip(queued, images), start=1):
        if img is None:
            if verbose:
                emit(f"    [{n:2d}] unreachable       {c.domain}")
            continue
        emb, _ = encoder.encode_primary(img)
        if emb is None:
            if verbose:
                emit(f"    [{n:2d}] no face in image  {c.domain}")
            continue
        score = cosine(probe_embedding, emb)
        ok = score >= threshold
        if verbose:
            emit(f"    [{n:2d}] cos={score:+.4f} {'MATCH  ' if ok else 'no match'} "
                 f"{c.domain} {'(corroborated)' if c.corroborated else ''}")
        if ok:
            confirmed.append((score, c))
    # Ties are real: the same image reached us as two candidates (say an
    # Instagram post and an unresolved redirector pointing at it), so they score
    # identically. Break toward the one that makes a better record rather than
    # leaving it to upstream ordering and sort stability.
    confirmed.sort(key=lambda t: (-t[0], not t[1].is_social, not t[1].resolved,
                                  not t[1].corroborated))
    return confirmed
