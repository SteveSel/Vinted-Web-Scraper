import argparse
import os
import time
from pathlib import Path

from scraper import scrape_by_reference_images, scrape_first_page_listings
from matchers import evaluate_listings
from utils import (
    DEFAULT_TITLE_BLACKLIST,
    build_vinted_search_url,
    filter_listings_by_title_blacklist,
    load_environment,
    load_reference_images,
    parse_blacklist_words,
    persist_results,
    print_top_matches,
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

    groq_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    results = evaluate_listings(
        listings=listings,
        reference_images=refs,
        matcher_mode=matcher_mode,
        min_score=min_score,
        clip_model_name=clip_model_name,
        groq_model=groq_model,
        gemini_model=gemini_model,
        gemini_batch_size=gemini_batch_size,
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
        description="Vinted visual match assistant (first-page scrape + cloud or local image comparison)"
    )
    parser.add_argument("--keyword", default="", help="Vinted search keyword (required for keyword mode)")
    parser.add_argument(
        "--search-url", default="",
        help="Optional custom Vinted catalog URL. If set, --keyword is still used only for logs.",
    )
    parser.add_argument(
        "--search-mode", choices=["keyword", "image"], default="keyword",
        help="Use keyword URL search or Vinted image-search upload mode.",
    )
    parser.add_argument(
        "--matcher-mode", choices=["clip", "gemini", "groq"], default="clip",
        help="Image matching engine. clip=CLIP semantic, gemini=Gemini API, groq=Groq cloud.",
    )
    parser.add_argument("--references-dir", default="references", help="Folder with reference images")
    parser.add_argument("--max-listings", type=int, default=24, help="Max listings from first page")
    parser.add_argument(
        "--per-reference-max", type=int, default=12,
        help="Max listings per reference image in image-search mode.",
    )
    parser.add_argument(
        "--max-reference-images", type=int, default=0,
        help="Limit how many reference images are used (0 = all).",
    )
    parser.add_argument(
        "--gemini-batch-size", type=int, default=5,
        help="How many listing images to evaluate per Gemini request.",
    )
    parser.add_argument(
        "--clip-model-name", default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model name for local matching.",
    )
    parser.add_argument(
        "--blacklist-words", default=",".join(DEFAULT_TITLE_BLACKLIST),
        help="Comma-separated words to exclude by listing title before image matching.",
    )
    parser.add_argument("--min-score", type=float, default=0.70, help="Minimum score to count as match")
    parser.add_argument("--output-dir", default="output", help="JSON output folder")
    parser.add_argument(
        "--loop-minutes", type=int, default=5,
        help="Polling interval in minutes (default: 5). Set 0 or negative to run once.",
    )
    parser.add_argument(
        "--show-browser", action="store_true",
        help="Open Chromium in visible (headed) mode so you can watch actions live.",
    )
    args = parser.parse_args()
    title_blacklist_words = parse_blacklist_words(args.blacklist_words)

    search_url = args.search_url or build_vinted_search_url(args.keyword)
    if args.search_mode == "keyword" and not args.keyword and not args.search_url:
        raise RuntimeError("In keyword mode, provide --keyword or --search-url.")
    seen_match_urls: set[str] = set()

    scan_kwargs = dict(
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

    if args.loop_minutes <= 0:
        run_scan(**scan_kwargs)
        return

    print(
        f"Starting continuous monitoring every {args.loop_minutes} minute(s). "
        "Press Ctrl+C to stop."
    )
    cycle = 1
    while True:
        cycle_start = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        print(f"\n=== Scan cycle #{cycle} at {cycle_start} ===")
        try:
            run_scan(**scan_kwargs)
        except Exception as exc:
            print(f"Cycle #{cycle} failed: {exc}")

        cycle += 1
        print(f"\nWaiting {args.loop_minutes} minute(s) before next scan...")
        time.sleep(args.loop_minutes * 60)


if __name__ == "__main__":
    main()
