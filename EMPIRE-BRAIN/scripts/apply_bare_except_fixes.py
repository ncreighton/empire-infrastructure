"""
Apply bare_except anti-pattern fixes across the empire codebase.

Reads approved enhancement records from brain.db where enhancement_type='refactor'
and title matches 'bare_except', then replaces all bare `except:` with
`except Exception:` in each file. Marks enhancements as 'applied' when done.

Usage:
    python scripts/apply_bare_except_fixes.py
"""

import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'knowledge', 'brain.db')
DB_PATH = os.path.normpath(DB_PATH)

# Regex to match bare except on a line, including inline code after the colon.
# Matches: `except:`, `except :`, `except: pass`, `except: return {}`, `except:  # comment`
# Does NOT match: `except ValueError:`, `except (A, B):`, `except Exception:`
# Strategy: use re.sub on each line to replace `except` followed by optional
# whitespace then `:` — but only when there's no exception class between them.
BARE_EXCEPT_RE = re.compile(r'(\bexcept)\s*:')

# A line-level check to confirm this is a real bare except (not inside a string
# or a typed except). We check that the `except` keyword is followed only by
# optional whitespace before the colon.
BARE_EXCEPT_LINE = re.compile(r'^(\s*)except\s*:(.*)$')

# Typed except patterns we must NOT touch
TYPED_EXCEPT = re.compile(r'^\s*except\s+\S')


def fix_bare_excepts_in_file(file_path: str) -> dict:
    """Replace bare `except:` with `except Exception:` in a single file.

    Handles both standalone `except:` and inline forms like `except: pass`.

    Returns a dict with:
        - replacements: number of lines changed
        - error: error message if any, else None
    """
    result = {'replacements': 0, 'error': None}

    if not os.path.exists(file_path):
        result['error'] = 'file not found'
        return result

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            original_lines = f.readlines()
    except Exception as e:
        result['error'] = f'read error: {e}'
        return result

    new_lines = []
    for line in original_lines:
        match = BARE_EXCEPT_LINE.match(line)
        if match and not TYPED_EXCEPT.match(line):
            indent = match.group(1)
            rest = match.group(2)  # everything after the colon
            new_line = f'{indent}except Exception:{rest}\n'
            new_lines.append(new_line)
            result['replacements'] += 1
        else:
            new_lines.append(line)

    if result['replacements'] > 0:
        try:
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.writelines(new_lines)
        except Exception as e:
            result['error'] = f'write error: {e}'
            return result

    return result


def main():
    print(f'EMPIRE-BRAIN: Apply bare_except fixes')
    print(f'=' * 60)
    print(f'Database: {DB_PATH}')
    print()

    if not os.path.exists(DB_PATH):
        print(f'ERROR: Database not found at {DB_PATH}')
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Fetch all approved bare_except enhancements
    cur.execute("""
        SELECT id, file_path, project_slug
        FROM enhancements
        WHERE enhancement_type = 'refactor'
          AND title LIKE '%bare_except%'
          AND status = 'approved'
        ORDER BY file_path
    """)
    records = cur.fetchall()
    total = len(records)

    if total == 0:
        print('No approved bare_except enhancements found.')
        conn.close()
        return

    print(f'Found {total} approved bare_except enhancements to apply.')
    print()

    # Track statistics
    stats = {
        'total': total,
        'files_fixed': 0,
        'files_skipped': 0,
        'files_not_found': 0,
        'files_error': 0,
        'total_replacements': 0,
        'already_clean': 0,
    }

    applied_ids = []

    for i, record in enumerate(records, 1):
        rec_id = record['id']
        file_path = record['file_path']
        project = record['project_slug'] or 'unknown'
        short_path = file_path.replace('D:\\Claude Code Projects\\', '')

        result = fix_bare_excepts_in_file(file_path)

        if result['error']:
            if result['error'] == 'file not found':
                stats['files_not_found'] += 1
                print(f'  [{i:3d}/{total}] SKIP (missing): {short_path}')
            else:
                stats['files_error'] += 1
                print(f'  [{i:3d}/{total}] ERROR: {short_path} - {result["error"]}')
        elif result['replacements'] == 0:
            stats['already_clean'] += 1
            stats['files_skipped'] += 1
            applied_ids.append(rec_id)
            print(f'  [{i:3d}/{total}] CLEAN (0 bare excepts): {short_path}')
        else:
            stats['files_fixed'] += 1
            stats['total_replacements'] += result['replacements']
            applied_ids.append(rec_id)
            print(f'  [{i:3d}/{total}] FIXED ({result["replacements"]} replacements): {short_path}')

    # Mark applied enhancements in the database
    now = datetime.now(timezone.utc).isoformat()
    if applied_ids:
        placeholders = ','.join('?' for _ in applied_ids)
        cur.execute(f"""
            UPDATE enhancements
            SET status = 'applied', applied_at = ?
            WHERE id IN ({placeholders})
        """, [now] + applied_ids)
        conn.commit()
        print(f'\nMarked {len(applied_ids)} enhancements as applied in brain.db.')

    conn.close()

    # Print summary
    print()
    print(f'=' * 60)
    print(f'SUMMARY')
    print(f'=' * 60)
    print(f'  Total enhancements:     {stats["total"]}')
    print(f'  Files fixed:            {stats["files_fixed"]}')
    print(f'  Total replacements:     {stats["total_replacements"]}')
    print(f'  Already clean:          {stats["already_clean"]}')
    print(f'  Files not found:        {stats["files_not_found"]}')
    print(f'  Files with errors:      {stats["files_error"]}')
    print(f'  DB records applied:     {len(applied_ids)}')
    print()


if __name__ == '__main__':
    main()
