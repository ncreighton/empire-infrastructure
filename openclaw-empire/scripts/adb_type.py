"""ADB reliable text typer - sends text via ADB avoiding autocorrect issues.
Uses input text for individual words and keyevent for spaces/special chars."""

import subprocess
import time
import sys

from adb_config import ADB, DEVICE as DEV

def adb_cmd(cmd):
    """Run an ADB shell command."""
    full = [ADB, "-s", DEV, "shell"] + cmd.split()
    result = subprocess.run(full, capture_output=True, text=True, timeout=10)
    if result.returncode != 0 and result.stderr:
        print(f"  ERR: {result.stderr.strip()}", file=sys.stderr)
    return result

def type_char(ch):
    """Type a single character via ADB."""
    if ch == ' ':
        adb_cmd("input keyevent 62")  # SPACE
    elif ch == '\n':
        adb_cmd("input keyevent 66")  # ENTER
    elif ch == '.':
        adb_cmd("input keyevent 56")  # PERIOD
    elif ch == ',':
        adb_cmd("input keyevent 55")  # COMMA
    elif ch == '-':
        adb_cmd("input keyevent 69")  # MINUS
    elif ch == '+':
        adb_cmd("input keyevent 81")  # PLUS
    elif ch == '(':
        adb_cmd("input text '('")
    elif ch == ')':
        adb_cmd("input text ')'")
    elif ch == '/':
        adb_cmd("input keyevent 76")  # SLASH
    elif ch == ':':
        adb_cmd("input text ':'")
    elif ch == ';':
        adb_cmd("input keyevent 74")  # SEMICOLON
    elif ch == '|':
        adb_cmd("input text '|'")
    elif ch == '&':
        adb_cmd("input text '\\&'")
    elif ch == '@':
        adb_cmd("input keyevent 77")  # AT
    elif ch == "'":
        adb_cmd("input keyevent 75")  # APOSTROPHE
    else:
        adb_cmd(f"input text '{ch}'")

def type_word(word):
    """Type a word by sending each character individually."""
    for ch in word:
        type_char(ch)
        time.sleep(0.05)

def type_text(text):
    """Type full text, word by word with proper delays."""
    words = text.split(' ')
    total = len(words)
    for i, word in enumerate(words):
        if word == '':
            # Multiple spaces or leading space
            adb_cmd("input keyevent 62")
            continue

        # Handle embedded newlines within a word
        parts = word.split('\n')
        for j, part in enumerate(parts):
            if part:
                type_word(part)
            if j < len(parts) - 1:
                adb_cmd("input keyevent 66")  # ENTER
                time.sleep(0.1)

        # Add space after word (except last)
        if i < total - 1:
            adb_cmd("input keyevent 62")  # SPACE
            time.sleep(0.08)

        # Progress indicator
        if (i + 1) % 10 == 0:
            pct = int((i + 1) / total * 100)
            print(f"  [{pct}%] {i+1}/{total} words typed...")

def select_all_and_delete():
    """Select all text in current field and delete it."""
    # Long press to trigger selection
    adb_cmd("input keyevent --longpress 29")  # Long press 'a' key
    time.sleep(0.5)
    # Try Ctrl+A via keyevent
    # On Android, we can try sending key combination
    # KEYCODE_A = 29, with META_CTRL_ON
    subprocess.run([ADB, "-s", DEV, "shell", "input", "keyevent", "--meta", "4096", "29"],
                   capture_output=True, timeout=5)
    time.sleep(0.3)
    adb_cmd("input keyevent 67")  # DEL
    time.sleep(0.3)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python adb_type.py <text_file>")
        print("  Reads text from file and types it via ADB")
        sys.exit(1)

    filepath = sys.argv[1]
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Typing {len(text)} characters ({len(text.split())} words)...")
    print(f"Estimated time: ~{len(text.split()) * 0.15:.0f} seconds")

    type_text(text)
    print("\nDone!")
