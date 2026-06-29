import re
from pathlib import Path

from playwright.sync_api import Locator, Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

from models import Listing


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
    close_selectors = [
        "button[aria-label='Close']",
        "button[aria-label*='close' i]",
        "[data-testid='domain-select-modal'] button",
        "button:has-text('Continue')",
        "button:has-text('Got it')",
        "button:has-text('Accept all')",
        "button:has-text('Accept')",
    ]
    for _ in range(4):
        if not click_first_visible(page, close_selectors, timeout_ms=1200):
            break

    # ponytail: nuke overlay pointer-events so clicks go through
    page.evaluate("""
        () => {
          document.querySelectorAll(
            "[data-testid='domain-select-modal--overlay'], .ReactModal__Overlay, .web_ui__Dialog__overlay"
          ).forEach(el => { if (el instanceof HTMLElement) el.style.pointerEvents = "none"; });
        }
    """)
    page.wait_for_timeout(200)


def _try_upload_in_panel(page: Page, image_path: Path) -> bool:
    """After a trigger opens a panel, try to find the actual upload button or file input inside."""
    # ponytail: one layer of "find the upload button inside the panel"
    upload_selectors = [
        "button:has-text('Upload an image')",
        "button:has-text('Add an image')",
        "button:has-text('Add image')",
        "button:has-text('Upload image')",
        "button:has-text('Upload photo')",
        "button:has-text('Choose image')",
        "button:has-text('Search by photo')",
        "button:has-text('Search with image')",
        "button:has-text('Buscar con imagen')",
        "button:has-text('Dodaj zdjęcie')",
        "[data-testid*='add-image']",
        "[data-testid*='upload-image']",
        "[data-testid*='add-photo']",
    ]
    for sel in upload_selectors:
        try:
            btn = page.locator(sel).first
            if btn.count() == 0 or not btn.is_visible(timeout=900):
                continue
            try:
                with page.expect_file_chooser(timeout=3000) as fc:
                    btn.click(timeout=1500)
                fc.value.set_files(str(image_path))
                _finalize_image_search(page)
                return True
            except PlaywrightTimeoutError:
                # Button didn't open file chooser — check for dynamic file input
                pass
        except Exception:
            continue

    # Check if a file input appeared dynamically after clicking
    fi = page.locator("input[type='file']")
    if fi.count() > 0:
        fi.first.set_input_files(str(image_path), timeout=4000)
        _finalize_image_search(page)
        return True

    return False


def run_vinted_image_search(page: Page, image_path: Path) -> None:
    """Upload a reference image to Vinted's image search. Tries the primary paths only."""
    dismiss_blocking_modals(page)

    # 1. Direct file input (simplest case)
    file_input = page.locator("input[type='file']")
    if file_input.count() > 0:
        file_input.first.set_input_files(str(image_path), timeout=4000)
        _finalize_image_search(page)
        return

    # 2. Known image-search button — may open file chooser directly or open a panel first
    image_btn = page.locator("button[data-testid='search-by-image-button']")
    if image_btn.count() > 0 and image_btn.first.is_visible(timeout=1500):
        try:
            with page.expect_file_chooser(timeout=4000) as fc:
                image_btn.first.click(timeout=1500)
            fc.value.set_files(str(image_path))
            _finalize_image_search(page)
            return
        except PlaywrightTimeoutError:
            if "/login" in page.url or "/signin" in page.url:
                raise RuntimeError(
                    "Vinted redirected to login page for image search. Login required."
                )
            # Button opened a panel but no file chooser yet — try the upload button inside
            if _try_upload_in_panel(page, image_path):
                return
        except Exception:
            pass

    # 3. Broad selector fallback — buttons with image/photo/camera in aria-label or title
    trigger_selectors = [
        "button[aria-label*='image' i]",
        "button[aria-label*='photo' i]",
        "button[aria-label*='camera' i]",
        "button[title*='image' i]",
        "button[title*='Image' i]",
        "[data-testid*='image-search']",
        "[data-testid*='search-by-image']",
        "button:has-text('Search by image')",
        "button:has-text('Image search')",
        "button:has-text('Buscar con imagen')",
    ]
    for selector in trigger_selectors:
        try:
            trigger = page.locator(selector).first
            if trigger.count() == 0 or not trigger.is_visible(timeout=1000):
                continue
            with page.expect_file_chooser(timeout=4000) as fc:
                trigger.click()
            fc.value.set_files(str(image_path))
            _finalize_image_search(page)
            return
        except PlaywrightTimeoutError:
            # Trigger opened a panel — try upload button or dynamic file input inside
            if _try_upload_in_panel(page, image_path):
                return
        except Exception:
            continue

    # 4. Last resort: any file input that appeared after all the clicking
    fallback_input = page.locator("input[type='file']")
    if fallback_input.count() > 0:
        fallback_input.first.set_input_files(str(image_path), timeout=4000)
        _finalize_image_search(page)
        return

    raise RuntimeError(
        "Could not find Vinted image-search upload control. "
        "The page UI may have changed, or login may be required."
    )


def _expand_crop_to_full_image(page: Page) -> bool:
    """Drag ReactCrop NW/SE handles to cover the entire uploaded image."""
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

        # Drag NW handle to top-left corner of the image
        page.mouse.move(nw_box["x"] + nw_box["width"] / 2, nw_box["y"] + nw_box["height"] / 2)
        page.mouse.down()
        page.mouse.move(image_box["x"] + 2, image_box["y"] + 2, steps=20)
        page.mouse.up()
        page.wait_for_timeout(120)

        # Drag SE handle to bottom-right corner of the image
        se_box = se_handle.bounding_box()
        if not se_box:
            return False
        page.mouse.move(se_box["x"] + se_box["width"] / 2, se_box["y"] + se_box["height"] / 2)
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


def _finalize_image_search(page: Page) -> None:
    """After image upload: expand crop to full image, then click Search."""
    page.wait_for_timeout(1200)

    # 1. Expand crop area to cover the full uploaded image
    expanded = _expand_crop_to_full_image(page)

    # 2. Try uncrop/no-crop buttons as alternative
    clicked_uncrop = click_first_visible(page, [
        "button:has-text('No crop')",
        "button:has-text('Uncrop')",
        "button:has-text('Original')",
        "button:has-text('Full image')",
        "button:has-text('Sin recorte')",
        "button:has-text('Sin recortar')",
        "[data-testid*='crop-toggle']",
        "[data-testid*='uncrop']",
    ], timeout_ms=900)
    page.wait_for_timeout(250)

    if expanded or clicked_uncrop:
        print("Crop adjusted to full image before search.")

    # 3. Click search button — multiple strategies
    search_selectors = [
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
    clicked_search = click_first_visible(page, search_selectors, timeout_ms=1800)

    # Fallback: try get_by_role
    if not clicked_search:
        try:
            role_btn = page.get_by_role("button", name=re.compile(r"^(Buscar|Search)$", re.I)).first
            if role_btn.count() > 0 and role_btn.is_visible(timeout=1200):
                role_btn.click(timeout=1800)
                clicked_search = True
        except Exception:
            pass

    # Fallback: submit button
    if not clicked_search:
        try:
            submit_btn = page.locator("button[type='submit']").first
            if submit_btn.count() > 0 and submit_btn.is_visible(timeout=1200):
                submit_btn.click(timeout=1500)
                clicked_search = True
        except Exception:
            pass

    if clicked_search:
        print("Search button clicked, waiting for results...")
    else:
        print("Warning: could not find search button after image upload.")

    page.wait_for_timeout(3000)


def extract_listings_from_page(
    page: Page, max_listings: int, seen_urls: set[str], source_reference_image: str = ""
) -> list[Listing]:
    listings: list[Listing] = []

    def add(url: str, image_url: str, title: str, price: str) -> None:
        if not url or url in seen_urls or not image_url:
            return
        listings.append(Listing(
            title=title or "Untitled listing",
            url=url,
            image_url=image_url,
            price=price or "Unknown price",
            source_reference_image=source_reference_image,
        ))
        seen_urls.add(url)

    # Primary: product item anchors
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
        add(full_url, image_url, title, price)

    # Fallback: image-card testids with embedded item IDs
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
        try:
            href = card.evaluate("""
                (el) => {
                  const a = el.closest('a[href*="/items/"]')
                    || el.parentElement?.querySelector('a[href*="/items/"]')
                    || el.parentElement?.parentElement?.querySelector('a[href*="/items/"]');
                  return a ? a.getAttribute('href') : '';
                }
            """)
            if isinstance(href, str) and href:
                url = href if href.startswith("http") else f"https://www.vinted.com{href}"
        except Exception:
            pass

        if not url:
            id_match = re.search(r"product-item-id-(\d+)", testid)
            if id_match:
                url = f"https://www.vinted.com/items/{id_match.group(1)}"

        add(url, image_url, title, price)

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
