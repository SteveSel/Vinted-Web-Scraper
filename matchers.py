import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

import torch
from google import genai
from google.genai import types
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from models import Listing, MatchResult
from utils import (
    detect_mime_from_bytes,
    download_image_bytes,
    get_gemini_api_key,
    get_groq_api_key,
    guess_image_mime_type,
    image_bytes_to_data_url,
)


# -- Matcher protocol: any matcher implements score(candidate_bytes) -> (score, best_ref) --

class Matcher(Protocol):
    def prepare_references(self, reference_paths: list[Path]) -> None: ...
    def score(self, image_bytes: bytes) -> tuple[float, str]: ...


# -- Groq vision LLM matcher --

class GroqMatcher:
    def __init__(self, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> None:
        self.model_name = model_name
        self.client = None
        try:
            from groq import Groq
            self.Groq = Groq
            self.available = True
        except ImportError:
            print("Warning: groq package not installed. Install with: pip install groq")
            self.available = False

        self._ref_bytes: dict[str, tuple[bytes, str]] = {}

    def prepare_references(self, reference_paths: list[Path]) -> None:
        self._ref_bytes = {
            ref.name: (ref.read_bytes(), guess_image_mime_type(ref)) for ref in reference_paths
        }

    def _ensure_client(self) -> None:
        if self.client is not None:
            return
        api_key = get_groq_api_key()
        if not api_key:
            raise RuntimeError("Groq API key is missing. Set GROQ_API_KEY in .env or environment.")
        self.client = self.Groq(api_key=api_key)

    def _compare_pair(self, ref_bytes: bytes, ref_mime: str, cand_bytes: bytes, cand_mime: str) -> float:
        self._ensure_client()
        ref_url = image_bytes_to_data_url(ref_bytes, ref_mime)
        cand_url = image_bytes_to_data_url(cand_bytes, cand_mime)
        prompt = (
            "Image 1 is the REFERENCE cassette tape. Image 2 is a CANDIDATE listing.\n"
            "Rate visual similarity from 0.0 to 1.0:\n"
            "  1.0 = identical product (same brand, model, colorway)\n"
            "  0.7 = same brand, different model or edition\n"
            "  0.5 = same type of product, different brand\n"
            "  0.2 = vaguely similar category\n"
            "  0.0 = completely different product\n"
            "Respond with ONLY the number, nothing else."
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": ref_url}},
                    {"type": "image_url", "image_url": {"url": cand_url}},
                ],
            }],
            temperature=0,
            max_completion_tokens=16,
        )
        raw_text = response.choices[0].message.content or ""

        raw_text = raw_text.strip()
        match = re.search(r"([01](?:\.\d+)?)", raw_text)
        if match:
            score = float(match.group(1))
            if score == 0.0:
                print(f"  [Groq] Score=0 — raw response: {raw_text[:200]}")
            return score
        raise RuntimeError(f"Could not parse numeric score from Groq response: {raw_text}")

    def score(self, image_bytes: bytes) -> tuple[float, str]:
        if not self.available:
            raise RuntimeError("Groq matcher unavailable (groq package not installed).")
        if not self._ref_bytes:
            raise RuntimeError("GroqMatcher references not prepared.")

        cand_mime = detect_mime_from_bytes(image_bytes)
        best_score, best_ref = 0.0, ""
        first = True
        for ref_name, (ref_bytes, ref_mime) in self._ref_bytes.items():
            if not first:
                time.sleep(1.5)  # ponytail: avoid 429 storms, increase if still rate-limited
            first = False
            s = self._compare_pair(ref_bytes, ref_mime, image_bytes, cand_mime)
            if s > best_score:
                best_score, best_ref = s, ref_name
        return best_score, best_ref


# -- CLIP embedding matcher (simplified: no rotation variants, no hash blending) --

class ClipMatcher:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.reference_embeddings: torch.Tensor | None = None
        self.reference_names: list[str] = []

    def _embed(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            raw = self.model.get_image_features(**inputs)
            if isinstance(raw, torch.Tensor):
                feats = raw
            elif hasattr(raw, "image_embeds") and isinstance(raw.image_embeds, torch.Tensor):
                feats = raw.image_embeds
            elif hasattr(raw, "pooler_output") and isinstance(raw.pooler_output, torch.Tensor):
                feats = raw.pooler_output
            elif hasattr(raw, "last_hidden_state") and isinstance(raw.last_hidden_state, torch.Tensor):
                feats = raw.last_hidden_state[:, 0, :]
            else:
                raise RuntimeError(f"Unsupported CLIP output type: {type(raw)}")
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats

    def prepare_references(self, reference_paths: list[Path]) -> None:
        images = [Image.open(p).convert("RGB") for p in reference_paths]
        self.reference_embeddings = self._embed(images)
        self.reference_names = [p.name for p in reference_paths]

    def score(self, image_bytes: bytes) -> tuple[float, str]:
        if self.reference_embeddings is None:
            raise RuntimeError("ClipMatcher references not prepared.")
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cand = self._embed([img])
        sims = (cand @ self.reference_embeddings.T).squeeze(0)
        best_idx = int(sims.argmax().item())
        return float(sims[best_idx].item()), self.reference_names[best_idx]


# -- Gemini batch matcher --

def _parse_gemini_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass
    return []


class GeminiMatcher:
    """Batch-scores candidates against references via Gemini vision API."""

    def __init__(self, model: str = "gemini-2.5-flash", batch_size: int = 5) -> None:
        self.model = model
        self.batch_size = max(1, batch_size)
        self._ref_paths: list[Path] = []
        api_key = get_gemini_api_key()
        if not api_key:
            raise RuntimeError(
                "Gemini API key is missing. Add GEMINI_API_KEY (or GOOGLE_API_KEY) to .env."
            )
        self.client = genai.Client(api_key=api_key)

    def prepare_references(self, reference_paths: list[Path]) -> None:
        self._ref_paths = list(reference_paths)

    def score(self, image_bytes: bytes) -> tuple[float, str]:
        # ponytail: single-image path delegates to batch of 1
        results = self.score_batch([(b"", image_bytes)])
        if results:
            return results[0]
        return 0.0, ""

    def score_batch(self, candidates: list[tuple[str, bytes]]) -> list[tuple[float, str]]:
        """Score a batch of (title, image_bytes) candidates. Returns [(score, reason), ...]."""
        instruction = (
            "You are matching marketplace listing images.\n"
            "Task: compare each candidate listing image against all provided reference images.\n"
            "Return ONLY valid JSON array. One object per candidate, in the same order.\n"
            "Each object: {\"is_match\": bool, \"score\": float 0..1, \"reason\": string}.\n"
            "Consider visual product identity, logo, typography, colorway, shape. "
            "Ignore minor lighting/background differences."
        )
        parts: list[types.Part] = [types.Part.from_text(text=instruction)]
        for ref in self._ref_paths:
            img_bytes = ref.read_bytes()
            ext = ref.suffix.lower().replace(".", "")
            mime = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

        for idx, (title, img_bytes) in enumerate(candidates, 1):
            parts.append(types.Part.from_text(text=f"Candidate #{idx} title: {title}"))
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        response = self.client.models.generate_content(model=self.model, contents=parts)
        raw = response.text or ""
        payloads = _parse_gemini_json_array(raw)

        results: list[tuple[float, str]] = []
        for i in range(len(candidates)):
            p = payloads[i] if i < len(payloads) else {}
            results.append((float(p.get("score", 0.0)), str(p.get("reason", ""))))
        return results


# -- Unified evaluate function --

def evaluate_listings(
    listings: list[Listing],
    reference_images: list[Path],
    matcher_mode: str,
    min_score: float,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    gemini_model: str = "gemini-2.5-flash",
    gemini_batch_size: int = 5,
) -> list[MatchResult]:
    """Single entry point for all matcher modes."""

    # -- Gemini uses batch API, handle separately --
    if matcher_mode == "gemini":
        return _evaluate_gemini_batched(
            listings, reference_images, gemini_model, min_score, gemini_batch_size
        )

    # -- CLIP / Groq use the Matcher protocol --
    if matcher_mode == "groq":
        m = GroqMatcher(model_name=groq_model)
        if not m.available:
            raise RuntimeError("Groq unavailable (groq package not installed).")
    else:
        m = ClipMatcher(model_name=clip_model_name)

    ref_map = {ref.name: ref for ref in reference_images}

    # Group by source reference to avoid re-preparing for every listing
    groups: dict[str, list[Listing]] = {}
    for listing in listings:
        key = listing.source_reference_image if listing.source_reference_image in ref_map else ""
        groups.setdefault(key, []).append(listing)

    results: list[MatchResult] = []
    idx = 0
    total = len(listings)

    for source_key, group_listings in groups.items():
        refs = [ref_map[source_key]] if source_key else reference_images
        m.prepare_references(refs)

        for listing in group_listings:
            idx += 1
            if matcher_mode == "groq" and idx > 1:
                time.sleep(1.5)  # ponytail: rate-limit courtesy, increase if still 429
            try:
                img_bytes = download_image_bytes(listing.image_url)
                score, best_ref = m.score(img_bytes)
                is_match = score >= min_score
                reason = f"Best {matcher_mode.upper()} match: '{best_ref}' = {score:.3f}"
                if not is_match:
                    reason = f"Score {score:.2f} < threshold {min_score:.2f}. {reason}"
                print(f"[{idx}/{total}] {listing.title[:70]} -> match={is_match} score={score:.2f} ref={best_ref}")
                results.append(MatchResult(
                    listing=listing, is_match=is_match, score=score, reason=reason,
                    raw_model_response=json.dumps({
                        "matcher": matcher_mode, "best_reference": best_ref, "score": round(score, 6),
                    }),
                ))
            except Exception as exc:
                print(f"[{idx}/{total}] Failed: '{listing.title}': {exc}")
                results.append(MatchResult(
                    listing=listing, is_match=False, score=0.0,
                    reason=f"Evaluation failed: {exc}", raw_model_response="",
                ))

    return results


def _evaluate_gemini_batched(
    listings: list[Listing],
    reference_images: list[Path],
    gemini_model: str,
    min_score: float,
    batch_size: int,
) -> list[MatchResult]:
    gm = GeminiMatcher(model=gemini_model, batch_size=batch_size)
    gm.prepare_references(reference_images)
    results: list[MatchResult] = []
    safe_bs = max(1, batch_size)

    for batch_start in range(0, len(listings), safe_bs):
        batch = listings[batch_start : batch_start + safe_bs]
        batch_candidates: list[tuple[str, bytes]] = []

        for listing in batch:
            try:
                img_bytes = download_image_bytes(listing.image_url)
                batch_candidates.append((listing.title, img_bytes))
            except Exception as exc:
                print(f"Download failed for '{listing.title}': {exc}")

        scores: list[tuple[float, str]] = []
        if batch_candidates:
            try:
                scores = gm.score_batch(batch_candidates)
            except Exception as exc:
                print(f"Gemini batch call failed: {exc}")

        cand_idx = 0
        for offset, listing in enumerate(batch):
            global_idx = batch_start + offset + 1
            if offset >= len(batch_candidates):
                results.append(MatchResult(
                    listing=listing, is_match=False, score=0.0,
                    reason="Could not download listing image", raw_model_response="",
                ))
                continue

            score_val, reason = scores[cand_idx] if cand_idx < len(scores) else (0.0, "")
            cand_idx += 1
            is_match = score_val >= min_score
            if not is_match:
                reason = f"Score {score_val:.2f} < threshold {min_score:.2f}. {reason}"
            print(f"[{global_idx}/{len(listings)}] {listing.title[:70]} -> match={is_match} score={score_val:.2f}")
            results.append(MatchResult(
                listing=listing, is_match=is_match, score=score_val,
                reason=reason, raw_model_response="",
            ))

    return results
