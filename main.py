import argparse
import io
import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import Page
from playwright.sync_api import Locator
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
import torch
from transformers import CLIPModel, CLIPProcessor


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


@dataclass
class Listing:
    title: str
    url: str
    image_url: str
    price: str
    source_reference_image: str = ""


@dataclass
class MatchResult:
    listing: Listing
    is_match: bool
    score: float
    reason: str
    raw_model_response: str


def load_environment() -> None:
    # Always load .env from the project root (same dir as this script),
    # even when command is executed from another working directory.
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    # Also allow inherited shell environment vars.
    load_dotenv(override=False)


def get_gemini_api_key() -> str:
    # Support both explicit GEMINI key and common Google key variable.
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


class OrbMatcher:
    def __init__(self) -> None:
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.reference_features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
            dict(checks=50)
        )

    def _extract_features(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Extract ORB keypoints and descriptors from PIL image."""
        # Convert PIL to OpenCV format
        img_array = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detect and compute features
        keypoints, descriptors = self.orb.detectAndCompute(img_cv, None)

        if descriptors is None:
            # Return empty arrays if no features found
            return np.array([]), np.array([]).reshape(0, 32)

        return keypoints, descriptors

    def prepare_references(self, reference_paths: list[Path]) -> None:
        """Prepare reference images by extracting their features."""
        self.reference_features = {}
        for ref_path in reference_paths:
            try:
                img = Image.open(ref_path).convert("RGB")
                keypoints, descriptors = self._extract_features(img)
                if len(descriptors) > 0:
                    self.reference_features[ref_path.name] = (keypoints, descriptors)
                    print(f"Extracted {len(descriptors)} features from {ref_path.name}")
                else:
                    print(f"Warning: No features found in {ref_path.name}")
            except Exception as exc:
                print(f"Failed to process {ref_path.name}: {exc}")

    def match(self, image_bytes: bytes) -> tuple[float, str]:
        """Match candidate image against all references using ORB features."""
        if not self.reference_features:
            raise RuntimeError("OrbMatcher references are not prepared.")

        # Extract features from candidate image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cand_keypoints, cand_descriptors = self._extract_features(img)

        if len(cand_descriptors) == 0:
            return 0.0, ""

        best_score = 0.0
        best_ref = ""

        # Compare against each reference
        for ref_name, (ref_keypoints, ref_descriptors) in self.reference_features.items():
            try:
                # Find matches using FLANN
                matches = self.flann.knnMatch(cand_descriptors, ref_descriptors, k=2)

                # Apply Lowe's ratio test
                good_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                    # If only 1 match, still consider it if distance is reasonable
                    elif len(match) == 1:
                        m = match[0]
                        if m.distance < 50:  # Reasonable distance threshold
                            good_matches.append(m)

                if len(good_matches) < 4:
                    continue

                # Calculate score based on number of good matches and quality
                match_ratio = len(good_matches) / min(len(cand_keypoints), len(ref_keypoints))
                avg_distance = np.mean([m.distance for m in good_matches])

                # Normalize distance (lower is better, so invert)
                distance_score = max(0, 1.0 - avg_distance / 100.0)

                # Combine ratio and distance scores
                score = 0.7 * match_ratio + 0.3 * distance_score

                if score > best_score:
                    best_score = score
                    best_ref = ref_name

            except Exception as exc:
                print(f"Matching failed for {ref_name}: {exc}")
                continue

        return best_score, best_ref


class ClipMatcher:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.reference_embeddings: torch.Tensor | None = None
        self.reference_names: list[str] = []
        self.reference_hashes: dict[str, int] = {}

    @staticmethod
    def _make_variants(image: Image.Image) -> list[Image.Image]:
        base = image.convert("RGB")
        return [
            base,
            base.rotate(-12, resample=Image.BICUBIC, expand=False),
            base.rotate(12, resample=Image.BICUBIC, expand=False),
        ]

    @staticmethod
    def _average_hash(image: Image.Image, hash_size: int = 8) -> int:
        grayscale = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        if hasattr(grayscale, "get_flattened_data"):
            pixels = list(grayscale.get_flattened_data())
        else:
            pixels = list(grayscale.getdata())
        avg = sum(pixels) / len(pixels)
        result = 0
        for idx, pixel in enumerate(pixels):
            if pixel > avg:
                result |= 1 << idx
        return result

    @staticmethod
    def _hamming_distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def _embed_images(self, images: list[Image.Image]) -> torch.Tensor:
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
                raise RuntimeError(
                    f"Unsupported CLIP image feature output type: {type(raw)}"
                )
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats

    def prepare_references(self, reference_paths: list[Path]) -> None:
        ref_variants: list[Image.Image] = []
        ref_names: list[str] = []
        self.reference_hashes = {}
        for ref_path in reference_paths:
            img = Image.open(ref_path).convert("RGB")
            variants = self._make_variants(img)
            ref_variants.extend(variants)
            ref_names.extend([ref_path.name] * len(variants))
            self.reference_hashes[ref_path.name] = self._average_hash(img)

        embeddings = self._embed_images(ref_variants)
        self.reference_embeddings = embeddings
        self.reference_names = ref_names

    def match(self, image_bytes: bytes) -> tuple[float, str]:
        if self.reference_embeddings is None or not self.reference_names:
            raise RuntimeError("ClipMatcher references are not prepared.")
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        candidate_variants = self._make_variants(img)
        cand_embeddings = self._embed_images(candidate_variants)
        sim = cand_embeddings @ self.reference_embeddings.T

        unique_refs: list[str] = []
        ref_to_cols: dict[str, list[int]] = {}
        for idx, ref_name in enumerate(self.reference_names):
            if ref_name not in ref_to_cols:
                unique_refs.append(ref_name)
                ref_to_cols[ref_name] = []
            ref_to_cols[ref_name].append(idx)

        best_ref = ""
        best_score = -1.0
        best_consistency = 0.0
        for ref_name in unique_refs:
            cols = ref_to_cols[ref_name]
            ref_sim = sim[:, cols]
            current_best = float(ref_sim.max().item())
            if current_best <= best_score:
                continue
            best_ref = ref_name
            best_score = current_best
            best_consistency = float(ref_sim.max(dim=1).values.mean().item())

        if best_ref:
            hash_similarity = 1.0
            if best_ref in self.reference_hashes:
                candidate_hash = self._average_hash(img)
                ref_hash = self.reference_hashes[best_ref]
                hash_distance = self._hamming_distance(candidate_hash, ref_hash)
                hash_similarity = max(0.0, 1.0 - hash_distance / 64.0)
            score = float(0.65 * best_score + 0.25 * best_consistency + 0.10 * hash_similarity)
        else:
            score = 0.0

        return score, best_ref


def load_reference_images(reference_dir: Path) -> list[Path]:
    if not reference_dir.exists():
        raise FileNotFoundError(f"References folder not found: {reference_dir}")

    refs = [p for p in reference_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    if not refs:
        raise ValueError(f"No reference images found in {reference_dir}")
    return sorted(refs)


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
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(words))


def filter_listings_by_title_blacklist(
    listings: list[Listing], blacklist_words: list[str]
) -> tuple[list[Listing], list[Listing]]:
    if not blacklist_words:
        return listings, []
    kept: list[Listing] = []
    removed: list[Listing] = []
    for listing in listings:
        title_norm = normalize_text_for_match(listing.title)
        if any(word in title_norm for word in blacklist_words):
            removed.append(listing)
        else:
            kept.append(listing)
    return kept, removed


def build_vinted_search_url(keyword: str) -> str:
    return f"https://www.vinted.com/catalog?search_text={quote(keyword)}"


def extract_listings_from_page(
    page: Page, max_listings: int, seen_urls: set[str], source_reference_image: str = ""
) -> list[Listing]:
    listings: list[Listing] = []
    def add_listing(url: str, image_url: str, title: str, price: str) -> None:
        if not url or url in seen_urls or not image_url:
            return
        listings.append(
            Listing(
                title=title or "Untitled listing",
                url=url,
                image_url=image_url,
                price=price or "Unknown price",
                source_reference_image=source_reference_image,
            )
        )
        seen_urls.add(url)

    # Pass 1: traditional anchor-based cards.
    card_links = page.query_selector_all("a[data-testid='product-item-anchor'], a[href*='/items/']")
    for link in card_links:
        if len(listings) >= max_listings:
            break
        href = link.get_attribute("href") or ""
        if not href:
            continue
        full_url = href if href.startswith("http") else f"https://www.vinted.com{href}"
        if "/items/" not in full_url:
            continue

        img = link.query_selector("img")
        image_url = (img.get_attribute("src") or img.get_attribute("data-src") or "") if img else ""
        title = (img.get_attribute("alt") or "").strip() if img else ""
        if not title:
            title_node = link.query_selector("[data-testid='product-item-title'], p")
            title = title_node.inner_text().strip() if title_node else "Untitled listing"
        price_node = link.query_selector("[data-testid='product-item-price'], span")
        price = price_node.inner_text().strip() if price_node else "Unknown price"
        add_listing(full_url, image_url, title, price)

    # Pass 2: explicit product-item image blocks (works for visual-search results layout).
    image_cards = page.query_selector_all("[data-testid*='product-item-id-'][data-testid$='--image']")
    for card in image_cards:
        if len(listings) >= max_listings:
            break
        testid = card.get_attribute("data-testid") or ""
        img = card.query_selector("img")
        if not img:
            continue

        image_url = img.get_attribute("src") or img.get_attribute("data-src") or ""
        title = (img.get_attribute("alt") or "").strip()
        price = "Unknown price"
        if title:
            price_match = re.search(r"(\d+[.,]\d+\s*€)", title)
            if price_match:
                price = price_match.group(1)

        url = ""
        # Try to recover real listing URL from nearest anchor in ancestors/siblings.
        try:
            href = card.evaluate(
                """
                (el) => {
                  const nearAnchor = el.closest('a[href*="/items/"]')
                    || el.parentElement?.querySelector('a[href*="/items/"]')
                    || el.parentElement?.parentElement?.querySelector('a[href*="/items/"]');
                  return nearAnchor ? nearAnchor.getAttribute('href') : '';
                }
                """
            )
            if isinstance(href, str) and href:
                url = href if href.startswith("http") else f"https://www.vinted.com{href}"
        except Exception:
            pass

        # Last fallback: build canonical URL from data-testid item id.
        if not url:
            id_match = re.search(r"product-item-id-(\d+)", testid)
            if id_match:
                url = f"https://www.vinted.com/items/{id_match.group(1)}"

        add_listing(url, image_url, title, price)

    return listings


def scrape_first_page_listings(
    search_url: str, max_listings: int = 24, headed: bool = False
) -> list[Listing]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed, slow_mo=250 if headed else 0)
        page = browser.new_page()
        try:
            page.goto(search_url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(3500)
        except PlaywrightTimeoutError:
            browser.close()
            raise RuntimeError("Vinted page timeout. Try again or adjust the search query.")

        listings = extract_listings_from_page(page=page, max_listings=max_listings, seen_urls=set())
        browser.close()

    return listings


def click_first_visible(page: Page, selectors: list[str], timeout_ms: int = 1500) -> bool:
    for selector in selectors:
        try:
            locator: Locator = page.locator(selector).first
            if locator.count() > 0 and locator.is_visible(timeout=timeout_ms):
                locator.click(timeout=timeout_ms)
                page.wait_for_timeout(300)
                return True
        except Exception:
            continue
    return False


def dismiss_blocking_modals(page: Page) -> None:
    # Vinted sometimes opens a domain/country modal overlay that blocks interactions.
    close_selectors = [
        "button[aria-label='Close']",
        "button[aria-label*='close' i]",
        "[data-testid='domain-select-modal'] button",
        "[data-testid='domain-select-modal--overlay'] button",
        "button:has-text('Continue')",
        "button:has-text('Got it')",
        "button:has-text('Accept all')",
        "button:has-text('Accept')",
    ]

    # Try a few passes because one modal can reveal another (e.g. cookies after domain select).
    for _ in range(4):
        clicked = click_first_visible(page, close_selectors, timeout_ms=1200)
        if not clicked:
            break

    # Last-resort cleanup for lingering overlays that intercept pointer events.
    page.evaluate(
        """
        () => {
          const overlays = document.querySelectorAll(
            "[data-testid='domain-select-modal--overlay'], .ReactModal__Overlay, .web_ui__Dialog__overlay"
          );
          overlays.forEach((el) => {
            if (el instanceof HTMLElement) {
              el.style.pointerEvents = "none";
            }
          });
        }
        """
    )
    page.wait_for_timeout(200)


def run_vinted_image_search(page: Page, image_path: Path) -> None:
    dismiss_blocking_modals(page)

    def expand_react_crop_to_full_image() -> bool:
        try:
            crop_root = page.locator(".ReactCrop").first
            if crop_root.count() == 0 or not crop_root.is_visible(timeout=1200):
                return False

            image = page.locator(".ReactCrop__image, .ReactCrop img").first
            nw_handle = page.locator(".ReactCrop__drag-handle.ord-nw").first
            se_handle = page.locator(".ReactCrop__drag-handle.ord-se").first
            if image.count() == 0 or nw_handle.count() == 0 or se_handle.count() == 0:
                return False

            image_box = image.bounding_box()
            nw_box = nw_handle.bounding_box()
            se_box = se_handle.bounding_box()
            if not image_box or not nw_box or not se_box:
                return False

            # Drag NW handle to image top-left.
            page.mouse.move(nw_box["x"] + nw_box["width"] / 2, nw_box["y"] + nw_box["height"] / 2)
            page.mouse.down()
            page.mouse.move(image_box["x"] + 2, image_box["y"] + 2, steps=20)
            page.mouse.up()
            page.wait_for_timeout(120)

            # Re-read SE handle because coordinates change after first drag.
            se_box_after = se_handle.bounding_box()
            if not se_box_after:
                return False

            # Drag SE handle to image bottom-right.
            page.mouse.move(
                se_box_after["x"] + se_box_after["width"] / 2,
                se_box_after["y"] + se_box_after["height"] / 2,
            )
            page.mouse.down()
            page.mouse.move(
                image_box["x"] + image_box["width"] - 2,
                image_box["y"] + image_box["height"] - 2,
                steps=25,
            )
            page.mouse.up()
            page.wait_for_timeout(180)
            return True
        except Exception:
            return False

    def finalize_uploaded_image_search() -> None:
        # Give crop/search modal time to appear after upload.
        page.wait_for_timeout(1200)

        # First try direct ReactCrop handle expansion to full image.
        expanded_crop = expand_react_crop_to_full_image()

        # Then try UI controls for uncrop/original where available.
        uncrop_selectors = [
            "button:has-text('No crop')",
            "button:has-text('Uncrop')",
            "button:has-text('Original')",
            "button:has-text('Full image')",
            "button:has-text('Sin recorte')",
            "button:has-text('Sin recortar')",
            "button:has-text('Sin cortar')",
            "button:has-text('Recorte')",
            "button:has-text('Recortar')",
            "[data-testid*='crop-toggle']",
            "[data-testid*='uncrop']",
            "[data-testid*='original']",
        ]
        clicked_uncrop = click_first_visible(page, uncrop_selectors, timeout_ms=900)
        page.wait_for_timeout(250)
        if expanded_crop or clicked_uncrop:
            print("Crop adjusted to full image before search.")

        # Confirm and run visual search.
        search_after_upload = [
            "button:has-text('Buscar')",
            "button:has-text('Search')",
            "[role='button']:has-text('Buscar')",
            "[role='button']:has-text('Search')",
            ".web_ui__Button__content:has-text('Buscar')",
            ".web_ui__Button__content:has-text('Search')",
            "button:has-text('Apply')",
            "button:has-text('Done')",
            "[data-testid*='search']",
            "[data-testid*='apply']",
        ]
        clicked_search = click_first_visible(page, search_after_upload, timeout_ms=1800)
        if not clicked_search:
            # Role/name based fallback for buttons with nested spans/icons.
            try:
                role_btn = page.get_by_role("button", name=re.compile(r"^(Buscar|Search)$", re.I)).first
                if role_btn.count() > 0 and role_btn.is_visible(timeout=1200):
                    role_btn.click(timeout=1800)
                    clicked_search = True
            except Exception:
                pass

        if not clicked_search:
            # Some variants use submit buttons without visible text.
            try:
                submit_btn = page.locator("button[type='submit']").first
                if submit_btn.count() > 0 and submit_btn.is_visible(timeout=1200):
                    submit_btn.click(timeout=1500)
                    clicked_search = True
            except Exception:
                pass

        if not clicked_search:
            # Last resort: click nearest clickable ancestor of visible "Search/Buscar" text.
            try:
                page.evaluate(
                    """
                    () => {
                      const labels = Array.from(document.querySelectorAll('*'))
                        .filter(el => {
                          const t = (el.textContent || '').trim().toLowerCase();
                          return t === 'search' || t === 'buscar';
                        });
                      for (const el of labels) {
                        const clickable = el.closest('button,[role="button"],a,[data-testid],[class*="Button"]');
                        if (clickable instanceof HTMLElement) {
                          clickable.click();
                          return;
                        }
                      }
                    }
                    """
                )
            except Exception:
                pass

        page.wait_for_timeout(2500)

    file_input_locator = page.locator("input[type='file']")
    if file_input_locator.count() > 0:
        file_input_locator.first.set_input_files(str(image_path), timeout=4000)
        finalize_uploaded_image_search()
        return

    def try_add_image_button() -> bool:
        add_image_selectors = [
            "button:has-text('Upload an image')",
            "button:has-text('Add an image')",
            "button:has-text('Add image')",
            "button:has-text('Upload image')",
            "button:has-text('Upload photo')",
            "button:has-text('Choose image')",
            "button:has-text('Dodaj zdjęcie')",
            "button:has-text('Dodaj obraz')",
            "[data-testid*='add-image']",
            "[data-testid*='upload-image']",
            "[data-testid*='add-photo']",
        ]
        for sel in add_image_selectors:
            try:
                btn = page.locator(sel).first
                if btn.count() == 0 or not btn.is_visible(timeout=900):
                    continue
                try:
                    with page.expect_file_chooser(timeout=3000) as fc_info:
                        btn.click(timeout=1500)
                    fc_info.value.set_files(str(image_path))
                    finalize_uploaded_image_search()
                    return True
                except PlaywrightTimeoutError:
                    # Sometimes click opens/updates hidden input instead of native chooser event.
                    dynamic_input = page.locator("input[type='file']")
                    if dynamic_input.count() > 0:
                        dynamic_input.first.set_input_files(str(image_path), timeout=4000)
                        finalize_uploaded_image_search()
                        return True
            except Exception:
                continue
        return False

    trigger_selectors = [
        # Specific/common candidates first.
        "button[aria-label*='image' i]",
        "button[aria-label*='photo' i]",
        "button[aria-label*='camera' i]",
        "button[aria-label*='upload' i]",
        "button[title*='image' i]",
        "button[title*='photo' i]",
        "button[title*='camera' i]",
        "button[title*='upload' i]",
        # data-testid variants.
        "[data-testid*='image-search']",
        "[data-testid*='search-by-image']",
        "[data-testid*='photo']",
        "[data-testid*='camera']",
        "[data-testid*='upload']",
        # Generic locale-aware text fallbacks.
        "button:has-text('Search by image')",
        "button:has-text('Image search')",
        "button:has-text('Photo search')",
        "button:has-text('Zdjęciem')",
        "button:has-text('zdjęciem')",
        "button:has-text('Foto')",
        "button:has-text('Bild')",
        # Last-resort generic clickable controls.
        "button[class*='search']",
        "a[class*='search']",
    ]

    for selector in trigger_selectors:
        try:
            trigger = page.locator(selector).first
            if trigger.count() == 0:
                continue
            if not trigger.is_visible(timeout=1200):
                continue
            with page.expect_file_chooser(timeout=4000) as fc_info:
                trigger.click()
            file_chooser = fc_info.value
            file_chooser.set_files(str(image_path))
            finalize_uploaded_image_search()
            return
        except PlaywrightTimeoutError:
            # In some UIs this first click opens image-search panel, then "Add image" triggers chooser.
            if try_add_image_button():
                return
            continue
        except Exception:
            if try_add_image_button():
                return
            continue

    # Targeted fallback for Vinted's camera-icon search button (no text/testid in some locales).
    camera_buttons = page.locator("button:has(svg), [role='button']:has(svg)")
    camera_count = min(camera_buttons.count(), 120)
    for idx in range(camera_count):
        try:
            btn = camera_buttons.nth(idx)
            if not btn.is_visible(timeout=500):
                continue
            html = (btn.inner_html(timeout=1000) or "").lower()
            if "ai-camera" not in html and "camera-16" not in html and "aicamera16" not in html:
                continue
            try:
                with page.expect_file_chooser(timeout=3000) as fc_info:
                    btn.click(timeout=1500)
                fc_info.value.set_files(str(image_path))
                finalize_uploaded_image_search()
                return
            except PlaywrightTimeoutError:
                if try_add_image_button():
                    return
                # Some variants click open a hidden file input directly.
                dynamic_input = page.locator("input[type='file']")
                if dynamic_input.count() > 0:
                    dynamic_input.first.set_input_files(str(image_path), timeout=4000)
                    finalize_uploaded_image_search()
                    return
        except Exception:
            continue

    # Final fallback: try clicking small icon buttons and check if hidden file input appears.
    icon_like = page.locator(
        "button[aria-label], button[title], [data-testid], button svg, [role='button'] svg"
    )
    scan_limit = min(icon_like.count(), 80)
    for idx in range(scan_limit):
        try:
            target = icon_like.nth(idx)
            if not target.is_visible(timeout=500):
                continue
            target.click(timeout=1200)
            page.wait_for_timeout(250)
            dismiss_blocking_modals(page)
            if try_add_image_button():
                return
            dynamic_input = page.locator("input[type='file']")
            if dynamic_input.count() > 0:
                dynamic_input.first.set_input_files(str(image_path), timeout=4000)
                finalize_uploaded_image_search()
                return
        except Exception:
            continue

    raise RuntimeError(
        "Could not find Vinted image-search upload control. "
        "The page UI may have changed and selectors need an update."
    )


def scrape_by_reference_images(
    reference_images: list[Path], per_reference_max: int = 24, headed: bool = False
) -> list[Listing]:
    listings: list[Listing] = []
    seen_urls: set[str] = set()
    per_ref_cap = max(1, per_reference_max)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed, slow_mo=250 if headed else 0)
        page = browser.new_page()

        for idx, ref in enumerate(reference_images, start=1):
            try:
                print(f"Running Vinted image search [{idx}/{len(reference_images)}]: {ref.name}")
                page.goto("https://www.vinted.com/catalog", wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(2500)
                run_vinted_image_search(page, ref)
                page.wait_for_timeout(3000)
                from_this_ref = extract_listings_from_page(
                    page=page,
                    max_listings=per_ref_cap,
                    seen_urls=seen_urls,
                    source_reference_image=ref.name,
                )
                listings.extend(from_this_ref)
                print(f"Collected {len(from_this_ref)} listings from reference image {ref.name}")
            except Exception as exc:
                print(f"Image-search failed for {ref.name}: {exc}")

        browser.close()

    return listings


def download_image_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.content


def parse_gemini_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    return {"is_match": False, "score": 0.0, "reason": f"Could not parse model output: {text}"}


def parse_gemini_json_array_response(text: str) -> list[dict[str, Any]]:
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


def compare_candidates_batch_to_references(
    client: genai.Client,
    model: str,
    reference_images: list[Path],
    candidates: list[tuple[str, bytes]],
) -> tuple[list[dict[str, Any]], str]:
    instruction = (
        "You are matching marketplace listing images.\n"
        "Task: compare each candidate listing image against all provided reference images.\n"
        "Return ONLY valid JSON array. One object per candidate, in the same order as candidates.\n"
        "Each object must contain keys: is_match (bool), score (float 0..1), reason (short string).\n"
        "Consider visual product identity, logo, typography, colorway, and shape. "
        "Ignore minor lighting/background differences."
    )

    parts: list[types.Part] = [types.Part.from_text(text=instruction)]
    for ref in reference_images:
        img_bytes = ref.read_bytes()
        ext = ref.suffix.lower().replace(".", "")
        mime_type = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))

    for idx, (candidate_title, candidate_image_bytes) in enumerate(candidates, start=1):
        parts.append(types.Part.from_text(text=f"Candidate #{idx} title: {candidate_title}"))
        parts.append(types.Part.from_bytes(data=candidate_image_bytes, mime_type="image/jpeg"))

    response = client.models.generate_content(
        model=model,
        contents=parts,
    )
    raw_text = response.text or ""
    payloads = parse_gemini_json_array_response(raw_text)
    return payloads, raw_text


def evaluate_listings_gemini(
    listings: list[Listing],
    reference_images: list[Path],
    gemini_model: str,
    min_score: float,
    batch_size: int,
) -> list[MatchResult]:
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY (or GOOGLE_API_KEY) "
            "to your .env file in the project root."
        )

    client = genai.Client(api_key=api_key)
    results: list[MatchResult] = []

    safe_batch_size = max(1, batch_size)
    for batch_start in range(0, len(listings), safe_batch_size):
        batch = listings[batch_start : batch_start + safe_batch_size]
        batch_images: list[bytes | None] = []
        batch_candidates: list[tuple[str, bytes]] = []

        for listing in batch:
            try:
                img_bytes = download_image_bytes(listing.image_url)
                batch_images.append(img_bytes)
                batch_candidates.append((listing.title, img_bytes))
            except Exception as exc:
                print(f"[{batch_start + 1}-{batch_start + len(batch)}/{len(listings)}] Download failed for '{listing.title}': {exc}")
                batch_images.append(None)

        payloads: list[dict[str, Any]] = []
        raw = ""
        try:
            if batch_candidates:
                payloads, raw = compare_candidates_batch_to_references(
                    client=client,
                    model=gemini_model,
                    reference_images=reference_images,
                    candidates=batch_candidates,
                )
        except Exception as exc:
            print(f"[{batch_start + 1}-{batch_start + len(batch)}/{len(listings)}] Batch Gemini call failed: {exc}")

        payload_idx = 0
        for offset, listing in enumerate(batch, start=1):
            global_idx = batch_start + offset
            if batch_images[offset - 1] is None:
                results.append(
                    MatchResult(
                        listing=listing,
                        is_match=False,
                        score=0.0,
                        reason="Evaluation failed: could not download listing image",
                        raw_model_response="",
                    )
                )
                continue

            payload = payloads[payload_idx] if payload_idx < len(payloads) else {}
            payload_idx += 1
            is_match = bool(payload.get("is_match", False))
            score = float(payload.get("score", 0.0))
            reason = str(payload.get("reason", "No reason provided"))
            if score < min_score:
                is_match = False
                reason = f"Score {score:.2f} is below threshold {min_score:.2f}. {reason}"
            print(
                f"[{global_idx}/{len(listings)}] {listing.title[:70]} -> "
                f"match={is_match} score={score:.2f}"
            )
            results.append(
                MatchResult(
                    listing=listing,
                    is_match=is_match,
                    score=score,
                    reason=reason,
                    raw_model_response=raw,
                )
            )

    return results


def evaluate_listings_orb(
    listings: list[Listing],
    reference_images: list[Path],
    min_score: float,
) -> list[MatchResult]:
    matcher = OrbMatcher()
    matcher.prepare_references(reference_images)
    results: list[MatchResult] = []

    for idx, listing in enumerate(listings, start=1):
        try:
            image_bytes = download_image_bytes(listing.image_url)
            score, best_ref = matcher.match(image_bytes)
            is_match = score >= min_score
            reason = f"Best ORB feature match with '{best_ref}' is {score:.3f}"
            if not is_match:
                reason = f"Score {score:.2f} is below threshold {min_score:.2f}. {reason}"
            print(
                f"[{idx}/{len(listings)}] {listing.title[:70]} -> "
                f"match={is_match} score={score:.2f} ref={best_ref}"
            )
            results.append(
                MatchResult(
                    listing=listing,
                    is_match=is_match,
                    score=score,
                    reason=reason,
                    raw_model_response=json.dumps({
                        "matcher": "orb",
                        "best_reference": best_ref,
                        "score": round(score, 6)
                    }),
                )
            )
        except Exception as exc:
            print(f"[{idx}/{len(listings)}] Failed on listing '{listing.title}': {exc}")
            results.append(
                MatchResult(
                    listing=listing,
                    is_match=False,
                    score=0.0,
                    reason=f"Evaluation failed: {exc}",
                    raw_model_response="",
                )
            )
    return results


def evaluate_listings_clip(
    listings: list[Listing],
    reference_images: list[Path],
    min_score: float,
    clip_model_name: str,
) -> list[MatchResult]:
    matcher = ClipMatcher(clip_model_name)
    matcher.prepare_references(reference_images)
    results: list[MatchResult] = []

    for idx, listing in enumerate(listings, start=1):
        try:
            image_bytes = download_image_bytes(listing.image_url)
            score, best_ref = matcher.match(image_bytes)
            is_match = score >= min_score
            reason = f"Best CLIP similarity with '{best_ref}' is {score:.3f}"
            if not is_match:
                reason = f"Score {score:.2f} is below threshold {min_score:.2f}. {reason}"
            print(
                f"[{idx}/{len(listings)}] {listing.title[:70]} -> "
                f"match={is_match} score={score:.2f} ref={best_ref}"
            )
            results.append(
                MatchResult(
                    listing=listing,
                    is_match=is_match,
                    score=score,
                    reason=reason,
                    raw_model_response=json.dumps({
                        "matcher": "clip",
                        "best_reference": best_ref,
                        "score": round(score, 6)
                    }),
                )
            )
        except Exception as exc:
            print(f"[{idx}/{len(listings)}] Failed on listing '{listing.title}': {exc}")
            results.append(
                MatchResult(
                    listing=listing,
                    is_match=False,
                    score=0.0,
                    reason=f"Evaluation failed: {exc}",
                    raw_model_response="",
                )
            )
    return results


def test_match(reference_path: Path, candidate_path: Path, matcher_mode: str, clip_model_name: str) -> None:
    if matcher_mode == "orb":
        matcher = OrbMatcher()
        matcher.prepare_references([reference_path])
        candidate_bytes = candidate_path.read_bytes()
        score, best_ref = matcher.match(candidate_bytes)
        matcher_name = "ORB"
    else:  # clip
        matcher = ClipMatcher(clip_model_name)
        matcher.prepare_references([reference_path])
        candidate_bytes = candidate_path.read_bytes()
        score, best_ref = matcher.match(candidate_bytes)
        matcher_name = "CLIP"

    print(f"\n=== {matcher_name} comparison test ===")
    print(f"reference: {reference_path}")
    print(f"candidate: {candidate_path}")
    print(f"best reference match: {best_ref}")
    print(f"similarity score: {score:.6f}")
    print(f"=== End of {matcher_name} test ===\n")


def persist_results(results: list[MatchResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"matches_{ts}.json"

    # Convert to dict and ensure all values are JSON serializable
    output_payload = []
    for r in results:
        try:
            result_dict = asdict(r)
            # Ensure all fields are properly serializable
            result_dict['is_match'] = bool(result_dict['is_match'])
            result_dict['score'] = float(result_dict['score'])
            result_dict['reason'] = str(result_dict['reason'])
            result_dict['raw_model_response'] = str(result_dict['raw_model_response'])
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


def run_scan(
    search_mode: str,
    matcher_mode: str,
    search_url: str,
    keyword: str,
    references_dir: str,
    max_listings: int,
    per_reference_max: int,
    max_reference_images: int,
    gemini_batch_size: int,
    clip_model_name: str,
    title_blacklist_words: list[str],
    min_score: float,
    output_dir: str,
    seen_match_urls: set[str],
    headed: bool,
) -> None:
    refs = load_reference_images(Path(references_dir))
    if max_reference_images > 0:
        refs = refs[:max_reference_images]
    print(f"Loaded {len(refs)} reference images from '{references_dir}'")
    if search_mode == "image":
        print("Running Vinted image-search mode using reference images...")
        listings = scrape_by_reference_images(
            reference_images=refs, per_reference_max=per_reference_max, headed=headed
        )
    else:
        print(f"Search URL: {search_url}")
        print(f"Keyword: {keyword}")
        print("Scraping first page listings...")
        listings = scrape_first_page_listings(search_url, max_listings=max_listings, headed=headed)

    print(f"Collected {len(listings)} listing candidates.")
    listings, removed = filter_listings_by_title_blacklist(listings, title_blacklist_words)
    if title_blacklist_words:
        print(
            f"Title blacklist filtered {len(removed)} listing(s). "
            f"{len(listings)} listing(s) remain for visual matching."
        )
    if not listings:
        print("No listings found. Try another keyword or URL.")
        return

    if matcher_mode == "clip":
        print(f"Evaluating with CLIP model: {clip_model_name}")
        results = evaluate_listings_clip(
            listings=listings,
            reference_images=refs,
            min_score=min_score,
            clip_model_name=clip_model_name,
        )
    elif matcher_mode == "orb":
        print("Evaluating with ORB feature matching")
        results = evaluate_listings_orb(
            listings=listings,
            reference_images=refs,
            min_score=min_score,
        )
    else:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        print(f"Evaluating with Gemini model: {model}")
        print(f"Gemini batch size: {max(1, gemini_batch_size)} listing(s) per request")
        results = evaluate_listings_gemini(
            listings=listings,
            reference_images=refs,
            gemini_model=model,
            min_score=min_score,
            batch_size=gemini_batch_size,
        )
    output_path = persist_results(results, Path(output_dir))
    print_top_matches(results)

    new_matches = [r for r in results if r.is_match and r.listing.url not in seen_match_urls]
    if new_matches:
        print("\nNew matches since last cycle:")
        for result in sorted(new_matches, key=lambda x: x.score, reverse=True):
            print(
                f"- score={result.score:.2f} | price={result.listing.price} | title={result.listing.title}\n"
                f"  {result.listing.url}"
            )
            seen_match_urls.add(result.listing.url)
    else:
        print("\nNo new matches since last cycle.")

    print(f"\nSaved detailed results to: {output_path}")


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(
        description="Vinted visual match assistant (first-page scrape + Gemini image comparison)"
    )
    parser.add_argument("--keyword", default="", help="Vinted search keyword (required for keyword mode)")
    parser.add_argument(
        "--search-url",
        default="",
        help="Optional custom Vinted catalog URL. If set, --keyword is still used only for logs.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["keyword", "image"],
        default="keyword",
        help="Use keyword URL search or Vinted image-search upload mode.",
    )
    parser.add_argument(
        "--matcher-mode",
        choices=["clip", "orb", "gemini"],
        default="clip",
        help="Image matching engine. clip=CLIP semantic, orb=ORB features, gemini=API.",
    )
    parser.add_argument("--references-dir", default="references", help="Folder with reference images")
    parser.add_argument("--max-listings", type=int, default=24, help="Max listings from first page")
    parser.add_argument(
        "--per-reference-max",
        type=int,
        default=12,
        help="Max listings to collect per reference image in image-search mode.",
    )
    parser.add_argument(
        "--max-reference-images",
        type=int,
        default=0,
        help="Limit how many reference images are used (0 = all).",
    )
    parser.add_argument(
        "--gemini-batch-size",
        type=int,
        default=5,
        help="How many listing images to evaluate per Gemini request.",
    )
    parser.add_argument(
        "--clip-model-name",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model name for local matching.",
    )
    parser.add_argument(
        "--blacklist-words",
        default=",".join(DEFAULT_TITLE_BLACKLIST),
        help="Comma-separated words to exclude by listing title before image matching.",
    )
    parser.add_argument("--min-score", type=float, default=0.70, help="Minimum Gemini score to count as match")
    parser.add_argument("--output-dir", default="output", help="JSON output folder")
    parser.add_argument(
        "--test-match",
        action="store_true",
        help="Run a local image comparison test between a reference image and a candidate image.",
    )
    parser.add_argument(
        "--test-matcher",
        choices=["clip", "orb"],
        default="clip",
        help="Matcher to use for --test-match (default: clip).",
    )
    parser.add_argument(
        "--test-reference-path",
        default="",
        help="Path to the reference image for --test-clip.",
    )
    parser.add_argument(
        "--test-candidate-path",
        default="",
        help="Path to the candidate image for --test-clip.",
    )
    parser.add_argument(
        "--loop-minutes",
        type=int,
        default=5,
        help="Polling interval in minutes (default: 5). Set 0 or negative to run once.",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Open Chromium in visible (headed) mode so you can watch actions live.",
    )
    args = parser.parse_args()
    title_blacklist_words = parse_blacklist_words(args.blacklist_words)

    if args.test_match:
        if not args.test_reference_path or not args.test_candidate_path:
            raise RuntimeError("--test-match requires --test-reference-path and --test-candidate-path.")
        test_match(
            Path(args.test_reference_path),
            Path(args.test_candidate_path),
            args.test_matcher,
            args.clip_model_name,
        )
        return

    search_url = args.search_url or build_vinted_search_url(args.keyword)
    if args.search_mode == "keyword" and not args.keyword and not args.search_url:
        raise RuntimeError("In keyword mode, provide --keyword or --search-url.")
    seen_match_urls: set[str] = set()

    if args.loop_minutes <= 0:
        run_scan(
            search_mode=args.search_mode,
            matcher_mode=args.matcher_mode,
            search_url=search_url,
            keyword=args.keyword,
            references_dir=args.references_dir,
            max_listings=args.max_listings,
            per_reference_max=args.per_reference_max,
            max_reference_images=args.max_reference_images,
            gemini_batch_size=args.gemini_batch_size,
            clip_model_name=args.clip_model_name,
            title_blacklist_words=title_blacklist_words,
            min_score=args.min_score,
            output_dir=args.output_dir,
            seen_match_urls=seen_match_urls,
            headed=args.show_browser,
        )
        return

    print(
        f"Starting continuous monitoring every {args.loop_minutes} minute(s). "
        "Press Ctrl+C to stop."
    )
    cycle = 1
    while True:
        cycle_start = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n=== Scan cycle #{cycle} at {cycle_start} ===")
        try:
            run_scan(
                search_mode=args.search_mode,
                matcher_mode=args.matcher_mode,
                search_url=search_url,
                keyword=args.keyword,
                references_dir=args.references_dir,
                max_listings=args.max_listings,
                per_reference_max=args.per_reference_max,
                max_reference_images=args.max_reference_images,
                gemini_batch_size=args.gemini_batch_size,
                clip_model_name=args.clip_model_name,
                title_blacklist_words=title_blacklist_words,
                min_score=args.min_score,
                output_dir=args.output_dir,
                seen_match_urls=seen_match_urls,
                headed=args.show_browser,
            )
        except Exception as exc:
            print(f"Cycle #{cycle} failed: {exc}")

        cycle += 1
        print(f"\nWaiting {args.loop_minutes} minute(s) before next scan...")
        time.sleep(args.loop_minutes * 60)


if __name__ == "__main__":
    main()
