import base64
import json
import os
import re
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

from models import MatchResult

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_TITLE_BLACKLIST = [
    "video",
    "videocassette",
    "movie",
    "pelicula",
    "película",
    "film",
    "organizer",
    "sacchi",
    "cassettiera",
    "porta",
    "telecamera",
    "recorder",
    "scarpiera",
    "runner",
    "tovagliette",
    "pochette",
    "pantalone",
    "smalti",
    "chiusura",
    "canotta",
    "bush",
    "chicco",
    "marella",
    "vera pelle",
    "brocante",
    "jvs",
    "digital",
    "camera",
    "overdrive",
    "classic",
    "stereo",
    "cassette recorder",
    "cassette player",
    "cassette deck",
    "cassette box",
    "cassette storage",
    "cassette case",
    "cassette organizer",
    "wm",
    "walkman",
    "micro",
]


def load_environment() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    load_dotenv(override=False)


def get_gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def get_groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def guess_image_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def normalize_text_for_match(value: str) -> str:
    lowered = value.lower().strip()
    no_accents = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )
    return no_accents


def parse_blacklist_words(raw_words: str) -> list[str]:
    if not raw_words.strip():
        return []
    words = []
    for chunk in raw_words.split(","):
        w = normalize_text_for_match(chunk)
        if w:
            words.append(w)
    return list(dict.fromkeys(words))


def filter_listings_by_title_blacklist(
    listings: list["Listing"], blacklist_words: list[str]
) -> tuple[list["Listing"], list["Listing"]]:
    if not blacklist_words:
        return listings, []
    kept: list["Listing"] = []
    removed: list["Listing"] = []
    for listing in listings:
        title_norm = normalize_text_for_match(listing.title)
        if any(word in title_norm for word in blacklist_words):
            removed.append(listing)
        else:
            kept.append(listing)
    return kept, removed


def build_vinted_search_url(keyword: str) -> str:
    return f"https://www.vinted.com/catalog?search_text={quote(keyword)}"


def download_image_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.content


def load_reference_images(reference_dir: Path) -> list[Path]:
    if not reference_dir.exists():
        raise FileNotFoundError(f"References folder not found: {reference_dir}")

    refs = [p for p in reference_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    if not refs:
        raise ValueError(f"No reference images found in {reference_dir}")
    return sorted(refs)


def persist_results(results: list[MatchResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"matches_{ts}.json"

    output_payload = []
    for r in results:
        try:
            result_dict = asdict(r)
            result_dict["is_match"] = bool(result_dict["is_match"])
            result_dict["score"] = float(result_dict["score"])
            result_dict["reason"] = str(result_dict["reason"])
            result_dict["raw_model_response"] = str(result_dict["raw_model_response"])
            output_payload.append(result_dict)
        except Exception as exc:
            print(f"Warning: Failed to serialize result for '{r.listing.title}': {exc}")
            continue

    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return output_path


def print_top_matches(results: list[MatchResult]) -> None:
    matched = sorted((r for r in results if r.is_match), key=lambda x: x.score, reverse=True)
    if not matched:
        print("\nNo matches found on this page.")
        return

    print("\nTop matches:")
    for i, result in enumerate(matched, start=1):
        print(
            f"{i}. score={result.score:.2f} | price={result.listing.price} | title={result.listing.title}\n"
            f"   {result.listing.url}\n"
            f"   reason={result.reason}"
        )
