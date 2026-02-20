"""
Download YouTube video transcripts using proxies.
Uses youtube-transcript-api with GenericProxyConfig for proxy rotation.
"""

import os
import time
import sys

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig

OUTPUT_DIR = r"D:\Claude Code Projects\zimmwriter-project-new\docs\subs"

PROXIES_RAW = [
    ("89.185.0.44", 61234, "98350_TTckX", "9Pvy5B9A7i"),
    ("206.206.112.42", 61234, "98350_TTckX", "9Pvy5B9A7i"),
    ("81.22.138.44", 61234, "98350_TTckX", "9Pvy5B9A7i"),
    ("91.124.16.4", 61234, "98350_TTckX", "9Pvy5B9A7i"),
    ("91.124.90.85", 61234, "98350_TTckX", "9Pvy5B9A7i"),
    ("144.168.10.229", 61234, "98350_TTckX", "9Pvy5B9A7i"),
    ("149.126.91.115", 61234, "98350_TTckX", "9Pvy5B9A7i"),
    ("155.117.195.17", 61232, "98350_TTckX", "9Pvy5B9A7i"),
    ("155.117.195.147", 61232, "98350_TTckX", "9Pvy5B9A7i"),
    ("108.171.61.140", 61232, "98350_TTckX", "9Pvy5B9A7i"),
]

PROXY_URLS = [
    f"http://{user}:{pwd}@{host}:{port}"
    for host, port, user, pwd in PROXIES_RAW
]

VIDEOS = [
    (11, "ewYaFzWjVn8", "How to Connect ZimmWriter to WordPress"),
    (12, "xcCKd5RWOv0", "How to Create Bulk Product Roundups"),
    (13, "Q0KPL712uVs", "How to Use SERP Scraping"),
    (14, "omiQYLPujQQ", "How to Use Custom Outlines"),
    (15, "KkDls0WQRY8", "How to Use Custom Prompts"),
    (16, "WiTto3iTN-0", "Find a Bazillion Niches Using ZimmWriter"),
    (17, "d0dzrQGmVNg", "Bulk Convert 1,000 SEO Keywords to 5,000 Blog Post Titles"),
    (18, "3I-2A5-ZgJQ", "How to Use the SERP Discombobulator"),
    (19, "YkB5jL8fl_w", "How to Use the Text Discombobulator"),
    (20, "TiBx7Z9DbCY", "How to 1-Click Nuke AI Words"),
    (21, "hkL7S9cQLQM", "How to Mimic Any Writing Style"),
    (22, "eh1hMMwZ0EI", "How to Run ZimmWriter on Mac (M-Series + macOS 14+)"),
    (23, "Jvircu1yhwI", "Bulk Generate 10,000 AI Images"),
    (24, "3Mpq6g9n8Jg", "How to Run ZimmWriter on Mac (Using UTM)"),
]

DELAY_BETWEEN_VIDEOS = 3


def fetch_transcript_with_proxy(video_id, proxy_url):
    proxy_config = GenericProxyConfig(https_url=proxy_url, http_url=proxy_url)
    ytt = YouTubeTranscriptApi(proxy_config=proxy_config)
    transcript = ytt.fetch(video_id, languages=["en"])
    return transcript


def format_transcript(transcript):
    result_lines = []
    for entry in transcript:
        text = entry.text.replace(chr(10), " ").strip()
        if text:
            result_lines.append(text)
    return chr(10).join(result_lines)


def download_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success_count = 0
    fail_count = 0
    proxy_index = 0

    for lesson_num, video_id, title in VIDEOS:
        output_path = os.path.join(OUTPUT_DIR, f"{video_id}.txt")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            print(f"[SKIP] Lesson {lesson_num} - {title} (already exists)")
            success_count += 1
            continue

        print()
        print(f"[{lesson_num}/24] Downloading: {title} ({video_id})")

        transcript = None
        attempts = 0
        max_attempts = len(PROXY_URLS)

        while attempts < max_attempts:
            idx = proxy_index % len(PROXY_URLS)
            current_proxy_url = PROXY_URLS[idx]
            proxy_host = PROXIES_RAW[idx][0]
            print(f"  Trying proxy {idx + 1}/{len(PROXY_URLS)}: {proxy_host}...", end=" ", flush=True)

            try:
                transcript = fetch_transcript_with_proxy(video_id, current_proxy_url)
                print("SUCCESS")
                break
            except Exception as e:
                error_msg = str(e)[:120]
                print(f"FAILED ({error_msg})")
                proxy_index += 1
                attempts += 1
                time.sleep(1)

        if transcript is not None:
            text = format_transcript(transcript)
            header = f"# Lesson {lesson_num} - {title}" + chr(10) + chr(10)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(header + text)
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  Saved: {output_path} ({size_kb:.1f} KB)")
            success_count += 1
        else:
            print(f"  FAILED: All {max_attempts} proxies failed for {video_id}")
            fail_count += 1

        if lesson_num != VIDEOS[-1][0]:
            print(f"  Waiting {DELAY_BETWEEN_VIDEOS}s...")
            time.sleep(DELAY_BETWEEN_VIDEOS)

    print()
    print("=" * 60)
    print(f"DONE: {success_count} succeeded, {fail_count} failed out of {len(VIDEOS)} videos")
    print(f"Output directory: {OUTPUT_DIR}")
    return fail_count == 0


if __name__ == "__main__":
    ok = download_all()
    sys.exit(0 if ok else 1)
