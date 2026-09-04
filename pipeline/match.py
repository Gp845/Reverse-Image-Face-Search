#!/usr/bin/env python3
"""
Match-back verification.

A search hit on its own is just a URL we are asserting is the same person.
This step downloads each candidate image, runs the same detector+encoder over
it, and only accepts the hit when the face inside it actually matches the probe
embedding above SFace's published threshold. That is what makes the on-chain
record mean something.
"""

import cv2
import numpy as np
import requests

from .encode import COSINE_THRESHOLD, cosine

MAX_BYTES = 12 * 1024 * 1024


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
                      limit=25, social_only=False, verbose=True):
    """Return candidates whose face genuinely matches the probe, best score first."""
    confirmed = []
    checked = 0
    for c in candidates:
        if checked >= limit:
            break
        if social_only and not c.is_social:
            continue
        url = c.image_url or c.page_url
        if not url or not url.lower().startswith("http"):
            continue
        checked += 1
        img = fetch_image(url)
        if img is None:
            if verbose:
                print(f"    [{checked:2d}] unreachable       {c.domain}")
            continue
        emb, _ = encoder.encode_primary(img)
        if emb is None:
            if verbose:
                print(f"    [{checked:2d}] no face in image  {c.domain}")
            continue
        score = cosine(probe_embedding, emb)
        ok = score >= threshold
        if verbose:
            print(f"    [{checked:2d}] cos={score:+.4f} {'MATCH  ' if ok else 'no match'} "
                  f"{c.domain} {'(corroborated)' if c.corroborated else ''}")
        if ok:
            c_score = score
            confirmed.append((c_score, c))
    confirmed.sort(key=lambda t: t[0], reverse=True)
    return confirmed
