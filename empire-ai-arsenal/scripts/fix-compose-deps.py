#!/usr/bin/env python3
"""Remove cross-file depends_on from compose files."""
import re
import os

os.chdir("/opt/arsenal")

files = [
    "compose/tier2-intelligence.yml",
    "compose/tier3-crawling.yml",
]

for f in files:
    with open(f) as fh:
        content = fh.read()

    # Remove depends_on blocks (they reference cross-file services)
    content = re.sub(
        r'\n    depends_on:\n(      [^\n]+\n)+',
        '\n',
        content
    )

    with open(f, "w") as fh:
        fh.write(content)

    print(f"Fixed {f}")

print("Done - depends_on blocks removed from tier2 and tier3")
