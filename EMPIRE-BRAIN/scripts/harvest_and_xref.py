"""Auto-harvest learnings + generate cross-references for EMPIRE-BRAIN."""
import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "knowledge" / "brain.db"


def content_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def harvest_learnings(conn):
    """Harvest learnings from patterns, events, and project gaps."""
    added = 0

    # 1. High-confidence patterns
    patterns = conn.execute(
        "SELECT * FROM patterns WHERE confidence >= 0.7 ORDER BY confidence DESC"
    ).fetchall()
    for p in patterns:
        name = p["name"]
        ptype = p["pattern_type"] or "general"
        conf = p["confidence"] if p["confidence"] is not None else 0.5
        used = p["used_by_projects"] or ""

        existing = conn.execute(
            "SELECT id FROM learnings WHERE content LIKE ?", (f"%{name}%",)
        ).fetchone()
        if existing:
            continue

        content = (
            f'Pattern "{name}" (type: {ptype}, confidence: {conf:.0%}) '
            f"detected across projects: {used}. Consider standardizing as a shared system."
        )
        h = content_hash(content)
        try:
            conn.execute(
                "INSERT INTO learnings (content, category, source, content_hash) VALUES (?, ?, ?, ?)",
                (content, "pattern-insight", "auto-harvest", h),
            )
            added += 1
        except sqlite3.IntegrityError:
            pass

    # 2. Recurring event types
    events = conn.execute(
        "SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type HAVING cnt >= 3 ORDER BY cnt DESC"
    ).fetchall()
    for evt in events:
        etype = evt["event_type"]
        count = evt["cnt"]
        existing = conn.execute(
            "SELECT id FROM learnings WHERE content LIKE ? AND source = ?",
            (f"%{etype}%", "auto-harvest"),
        ).fetchone()
        if existing:
            continue
        content = f'Event "{etype}" occurred {count} times. Recurring workflow pattern worth monitoring.'
        h = content_hash(content)
        try:
            conn.execute(
                "INSERT INTO learnings (content, category, source, content_hash) VALUES (?, ?, ?, ?)",
                (content, "workflow-insight", "auto-harvest", h),
            )
            added += 1
        except sqlite3.IntegrityError:
            pass

    # 3. High-function projects with zero skills
    projects = conn.execute(
        "SELECT slug, function_count FROM projects WHERE skill_count = 0 AND function_count > 50 ORDER BY function_count DESC"
    ).fetchall()
    for proj in projects:
        slug = proj["slug"]
        fn = proj["function_count"] or 0
        existing = conn.execute(
            "SELECT id FROM learnings WHERE content LIKE ? AND source = ?",
            (f"%{slug}%skill%", "auto-harvest"),
        ).fetchone()
        if existing:
            continue
        content = f'Project "{slug}" has {fn} functions but 0 skills. A SKILL.md would improve Brain discoverability.'
        h = content_hash(content)
        try:
            conn.execute(
                "INSERT INTO learnings (content, category, source, content_hash) VALUES (?, ?, ?, ?)",
                (content, "gap-insight", "auto-harvest", h),
            )
            added += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    return added


def generate_cross_references(conn):
    """Generate cross-references between patterns, learnings, skills, and projects."""
    added = 0

    def insert_xref(stype, sid, ttype, tid, rel, strength):
        nonlocal added
        existing = conn.execute(
            "SELECT id FROM cross_references WHERE source_type=? AND source_id=? AND target_type=? AND target_id=?",
            (stype, sid, ttype, tid),
        ).fetchone()
        if existing:
            return
        conn.execute(
            "INSERT INTO cross_references (source_type, source_id, target_type, target_id, relationship, strength) VALUES (?,?,?,?,?,?)",
            (stype, sid, ttype, tid, rel, strength),
        )
        added += 1

    # 1. Pattern -> Project
    patterns = conn.execute(
        "SELECT id, used_by_projects FROM patterns WHERE used_by_projects IS NOT NULL"
    ).fetchall()
    for p in patterns:
        used = p["used_by_projects"] or ""
        for proj_name in [s.strip() for s in used.split(",") if s.strip()]:
            proj = conn.execute(
                "SELECT id FROM projects WHERE slug = ? OR name LIKE ?",
                (proj_name, f"%{proj_name}%"),
            ).fetchone()
            if proj:
                insert_xref("pattern", p["id"], "project", proj["id"], "used_by", 0.8)

    # 2. Learning -> Project (match source to project slug)
    learnings = conn.execute(
        "SELECT id, source FROM learnings WHERE source IS NOT NULL"
    ).fetchall()
    for l in learnings:
        source = l["source"] or ""
        proj = conn.execute(
            "SELECT id FROM projects WHERE slug = ? OR name LIKE ?",
            (source, f"%{source}%"),
        ).fetchone()
        if proj:
            insert_xref("learning", l["id"], "project", proj["id"], "derived_from", 0.6)

    # 3. Project <-> Project (same category)
    categories = conn.execute(
        "SELECT DISTINCT category FROM projects WHERE category IS NOT NULL AND category != 'uncategorized'"
    ).fetchall()
    for cat_row in categories:
        cat = cat_row["category"]
        cat_projs = conn.execute(
            "SELECT id FROM projects WHERE category = ?", (cat,)
        ).fetchall()
        for i in range(len(cat_projs)):
            for j in range(i + 1, len(cat_projs)):
                insert_xref(
                    "project", cat_projs[i]["id"],
                    "project", cat_projs[j]["id"],
                    "same_category", 0.5,
                )

    # 4. Skill -> Project
    skills = conn.execute(
        "SELECT id, project_slug FROM skills WHERE project_slug IS NOT NULL"
    ).fetchall()
    for s in skills:
        proj = conn.execute(
            "SELECT id FROM projects WHERE slug = ?", (s["project_slug"],)
        ).fetchone()
        if proj:
            insert_xref("skill", s["id"], "project", proj["id"], "belongs_to", 1.0)

    conn.commit()
    return added


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    learnings = harvest_learnings(conn)
    print(f"Learnings harvested: {learnings}")

    xrefs = generate_cross_references(conn)
    print(f"Cross-references added: {xrefs}")

    # Totals
    total_l = conn.execute("SELECT COUNT(*) FROM learnings").fetchone()[0]
    total_x = conn.execute("SELECT COUNT(*) FROM cross_references").fetchone()[0]
    print(f"--- TOTALS ---")
    print(f"Total learnings: {total_l}")
    print(f"Total cross-references: {total_x}")

    conn.close()
