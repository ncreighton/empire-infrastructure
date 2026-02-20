#!/usr/bin/env python3
"""
Download missing ZimmWriter lesson transcripts (11-24) from YouTube.

IMPORTANT: YouTube blocks transcript API requests from cloud/datacenter IPs.
This Azure VM (20.153.180.11) is blocked. To download these transcripts:

Option 1 (Recommended): Run this script from a residential network
  - Copy this file to your home PC/laptop
  - Run: python download_transcripts.py

Option 2: Use a VPN on this VM
  - Connect a VPN with residential IP
  - Then run this script

Option 3: Use a proxy
  - python download_transcripts.py --proxy socks5://host:port

Usage:
  python download_transcripts.py [--proxy URL] [--delay SECONDS]
"""
import os
import sys
import time
import argparse

SUBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'subs')

MISSING_VIDEOS = [
    ('ewYaFzWjVn8', 'ZimmWriter Lesson 11 - How to Connect ZimmWriter to Wordpress'),
    ('xcCKd5RWOv0', 'ZimmWriter Lesson 12 - How to Create Bulk Product Roundups'),
    ('Q0KPL712uVs', 'ZimmWriter Lesson 13 - How to Use SERP Scraping'),
    ('omiQYLPujQQ', 'ZimmWriter Lesson 14 - How to Use Custom Outlines'),
    ('KkDls0WQRY8', 'ZimmWriter Lesson 15 - How to Use Custom Prompts'),
    ('WiTto3iTN-0', 'ZimmWriter Lesson 16 - Find a Bazillion Niches Using ZimmWriter'),
    ('d0dzrQGmVNg', 'ZimmWriter Lesson 17 - Bulk Convert 1,000 SEO Keywords to 5,000 Blog Post Titles'),
    ('3I-2A5-ZgJQ', 'ZimmWriter Lesson 18 - How to Use the SERP Discombobulator'),
    ('YkB5jL8fl_w', 'ZimmWriter Lesson 19 - How to Use the Text Discombobulator'),
    ('TiBx7Z9DbCY', 'ZimmWriter Lesson 20 - How to 1-Click Nuke AI Words'),
    ('hkL7S9cQLQM', 'ZimmWriter Lesson 21 - How to Mimic Any Writing Style'),
    ('eh1hMMwZ0EI', 'ZimmWriter Lesson 22 - How to Run ZimmWriter on Mac (M-Series and macOS 14+)'),
    ('Jvircu1yhwI', 'ZimmWriter Lesson 23 - Bulk Generate 10,000 AI Images'),
    ('3Mpq6g9n8Jg', 'ZimmWriter Lesson 24 - How to Run ZimmWriter on Mac (Using UTM)'),
]


def download_with_api(vid_id, title, proxy_url=None):
    """Download transcript using youtube-transcript-api."""
    from youtube_transcript_api import YouTubeTranscriptApi

    kwargs = {}
    if proxy_url:
        from youtube_transcript_api.proxies import GenericProxyConfig
        kwargs['proxy_config'] = GenericProxyConfig(
            http_url=proxy_url, https_url=proxy_url
        )

    ytt = YouTubeTranscriptApi(**kwargs)
    transcript = ytt.fetch(vid_id)
    return '\n'.join(entry.text for entry in transcript)


def main():
    parser = argparse.ArgumentParser(description='Download ZimmWriter lesson transcripts')
    parser.add_argument('--proxy', help='Proxy URL (e.g., socks5://host:port)')
    parser.add_argument('--delay', type=int, default=5, help='Delay between requests in seconds (default: 5)')
    args = parser.parse_args()

    proxy_url = args.proxy or os.environ.get('PROXY_URL')

    os.makedirs(SUBS_DIR, exist_ok=True)
    existing = set(f.replace('.txt', '') for f in os.listdir(SUBS_DIR) if f.endswith('.txt'))
    to_download = [(v, t) for v, t in MISSING_VIDEOS if v not in existing]

    if not to_download:
        print('All transcripts already downloaded!')
        return

    print(f'Need to download {len(to_download)} transcripts')
    print(f'Already have {len(existing)} transcripts in {SUBS_DIR}')
    if proxy_url:
        print(f'Using proxy: {proxy_url}')
    print()

    success = 0
    failed = []

    for i, (vid_id, title) in enumerate(to_download):
        if i > 0:
            print(f'  Waiting {args.delay}s...', flush=True)
            time.sleep(args.delay)

        print(f'[{i+1}/{len(to_download)}] {vid_id} - {title}', flush=True)
        try:
            text = download_with_api(vid_id, title, proxy_url)
            filepath = os.path.join(SUBS_DIR, f'{vid_id}.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f'# {title}\n\n{text}')
            size = os.path.getsize(filepath)
            print(f'  OK: {size:,} bytes', flush=True)
            success += 1
        except Exception as e:
            err = str(e)[:150]
            print(f'  FAIL: {err}', flush=True)
            failed.append((vid_id, title))

    print(f'\nResults: {success}/{len(to_download)} succeeded, {len(failed)} failed')
    if failed:
        print('\nFailed videos:')
        for vid_id, title in failed:
            print(f'  {vid_id} - {title}')
        if 'block' in str(failed).lower() or not success:
            print('\n--- YouTube is blocking this IP ---')
            print('This is common for cloud/datacenter IPs (Azure, AWS, GCP).')
            print('Solutions:')
            print('  1. Run this script from a home PC/laptop (residential IP)')
            print('  2. Connect a VPN with a residential IP, then re-run')
            print('  3. Use a residential proxy: --proxy socks5://host:port')


if __name__ == '__main__':
    main()
