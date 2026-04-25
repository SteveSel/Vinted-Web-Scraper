# Vinted Visual Search Assistant

This project scrapes the **first page of Vinted listings** and compares each listing image to local reference images in the `references` folder using either local CLIP embeddings (default) or Gemini API.

## What it does

1. Loads all reference images from `references/`.
2. Opens a Vinted search page (`--keyword` or custom `--search-url`).
3. Scrapes listing cards from the first results page.
4. Downloads each listing image.
5. Scores visual similarity with local CLIP (default) or Gemini API.
6. Prints matches and saves full output in `output/matches_*.json`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Create env file:

```bash
cp .env.example .env
```

Then set your Gemini key:

```bash
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
```

If you use default CLIP matcher, Gemini key is not required.

## Usage

Keyword search (runs continuously every 5 minutes by default):

```bash
python main.py --keyword "teac cassette"
```

Run once only:

```bash
python main.py --keyword "teac cassette" --loop-minutes 0
```

Image-search mode (uses Vinted "search by image" with all files in `references/`):

```bash
python main.py --search-mode image --loop-minutes 5
```

Watch Chromium live while it runs:

```bash
python main.py --search-mode image --loop-minutes 5 --show-browser
```

Use local CLIP matcher explicitly (no API):

```bash
python main.py --search-mode image --matcher-mode clip --loop-minutes 0 --max-reference-images 5
```

Title blacklist before visual matching (defaults include `video,videocassette,movie,pelicula,film`):

```bash
python main.py --search-mode image --matcher-mode clip --blacklist-words "video,videocassette,movie,pelicula,film"
```

Custom Vinted URL (if you want to control filters manually):

```bash
python main.py \
  --search-mode keyword \
  --keyword "ignored-for-url" \
  --search-url "https://www.vinted.com/catalog?search_text=teac&order=newest_first" \
  --max-listings 24 \
  --min-score 0.75 \
  --loop-minutes 5
```

## Notes

- Vinted markup may change; selectors in `main.py` may need occasional tweaks.
- This assistant only evaluates listings present on the **first loaded page**.
- The Gemini model is prompted to return strict JSON (`is_match`, `score`, `reason`).
- On each loop, results are still saved to JSON, and console output highlights only new matched listings not seen in previous cycles.
- In `image` mode, the script uploads each reference image to Vinted image search, scrapes first-page results for each upload, and deduplicates listing URLs before Gemini scoring.
