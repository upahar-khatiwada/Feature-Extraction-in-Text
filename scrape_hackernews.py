"""Scrape story text from Hacker News's public API.

Uses only the standard library and Hacker News's public Firebase-backed API:
https://github.com/HackerNews/API. Output is a CSV with a single `text`
column, matching demo.csv, so it can be passed straight to main.py.
"""

import argparse
import csv
import html
import re
import time
import urllib.error
import urllib.request

BASE_URL = "https://hacker-news.firebaseio.com/v0"
USER_AGENT = "feature-extraction-demo/1.0 (dataset scraper for a school project)"


def fetch_json(path):
    request = urllib.request.Request(f"{BASE_URL}/{path}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=10) as response:
        import json
        return json.load(response)


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags (e.g. <p>, <a href=...>)
    return re.sub(r"\s+", " ", text).strip()


def scrape_stories(target_count, story_type="new", delay=0.2):
    """Collect up to target_count story texts (title + self-text) from Hacker News."""
    story_ids = fetch_json(f"{story_type}stories.json")

    posts = []
    for story_id in story_ids:
        if len(posts) >= target_count:
            break

        item = fetch_json(f"item/{story_id}.json")
        if not item or item.get("type") != "story" or item.get("dead") or item.get("deleted"):
            continue

        title = item.get("title", "")
        body = item.get("text", "")
        text = clean_text(f"{title}. {body}" if body else title)
        if text:
            posts.append(text)

        time.sleep(delay)  # be polite to the shared public API

    return posts[:target_count]


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Hacker News story text into a text-column CSV."
    )
    parser.add_argument(
        "--type", dest="story_type", default="new", choices=["new", "top", "best", "ask", "show"]
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--out", default="hackernews_dataset.csv")
    args = parser.parse_args()

    try:
        posts = scrape_stories(args.count, story_type=args.story_type)
    except urllib.error.HTTPError as error:
        print(f"Hacker News request failed: {error}")
        return
    except urllib.error.URLError as error:
        print(f"Could not reach Hacker News: {error}")
        return

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for text in posts:
            writer.writerow([text])

    print(f"Saved {len(posts)} stories ({args.story_type}) to {args.out}")


if __name__ == "__main__":
    main()
