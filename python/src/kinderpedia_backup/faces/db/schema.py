"""SQLite DDL and ordered, idempotent schema migrations for `Metadata/faces_index.sqlite`.

Every database (fresh or legacy) is brought to `SCHEMA_VERSION` by `apply_migrations`, which runs each
pending migration exactly once inside its own `BEGIN IMMEDIATE` transaction and records progress in
`meta.schema_version`. Each migration is also idempotent, so a legacy DB written before versioning
existed (no `schema_version` row) is migrated from version 0 safely.

Adding a migration: append `(next_version, "name", fn)` to `MIGRATIONS`. Never edit or reorder a
migration that has shipped; write a new one instead (e.g. to re-create the validation triggers after
adding a new `AssignmentSource` literal — `tests/test_faces_data.py` guards that).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import get_args

from ..models import AssignmentSource, AssignmentStatus, PersonRole

# ----------------------------------------------------------------------------------------------
# Base DDL (v1). Kept as a list of single statements so it can run inside one transaction.
# ----------------------------------------------------------------------------------------------

BASE_TABLES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS photos (
        photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
        rel_path TEXT UNIQUE NOT NULL,
        archive_child TEXT NOT NULL,
        year TEXT NOT NULL,
        month TEXT NOT NULL,
        album_leaf TEXT NOT NULL,
        filename TEXT NOT NULL,
        kinderpedia_album_id INTEGER,
        album_title TEXT NOT NULL DEFAULT '',
        album_date TEXT NOT NULL DEFAULT '',
        classroom_groups TEXT NOT NULL DEFAULT '',
        kinderpedia_image_id INTEGER,
        seq_in_album INTEGER,
        image_dateadd TEXT NOT NULL DEFAULT '',
        best_live_url TEXT NOT NULL DEFAULT '',
        download_url TEXT NOT NULL DEFAULT '',
        fullsize_url TEXT NOT NULL DEFAULT '',
        thumb_url TEXT NOT NULL DEFAULT '',
        width INTEGER NOT NULL DEFAULT 0,
        height INTEGER NOT NULL DEFAULT 0,
        file_size INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL,
        detector_id TEXT NOT NULL,
        face_count INTEGER NOT NULL DEFAULT 0,
        scanned_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_photos_child_year ON photos(archive_child, year)",
    "CREATE INDEX IF NOT EXISTS idx_photos_kp_img_id ON photos(kinderpedia_image_id)",
    """
    CREATE TABLE IF NOT EXISTS people (
        person_id INTEGER PRIMARY KEY AUTOINCREMENT,
        display_name TEXT UNIQUE NOT NULL COLLATE NOCASE,
        role TEXT NOT NULL DEFAULT 'classmate',
        archive_hint TEXT NOT NULL DEFAULT '',
        cover_face_id INTEGER,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clusters (
        cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
        archive_child TEXT NOT NULL,
        year TEXT NOT NULL,
        person_id INTEGER REFERENCES people(person_id) ON DELETE SET NULL,
        medoid_face_id INTEGER,
        sample_face_ids_json TEXT NOT NULL DEFAULT '[]',
        face_count INTEGER NOT NULL DEFAULT 0,
        photo_count INTEGER NOT NULL DEFAULT 0,
        suggested_person_id INTEGER REFERENCES people(person_id) ON DELETE SET NULL,
        suggested_similarity REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clusters_year ON clusters(year, face_count DESC)",
    """
    CREATE TABLE IF NOT EXISTS faces (
        face_id INTEGER PRIMARY KEY AUTOINCREMENT,
        photo_id INTEGER NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
        face_idx INTEGER NOT NULL,
        det_score REAL NOT NULL,
        bbox_top INTEGER NOT NULL,
        bbox_right INTEGER NOT NULL,
        bbox_bottom INTEGER NOT NULL,
        bbox_left INTEGER NOT NULL,
        embedding_f16 BLOB NOT NULL,
        cluster_id INTEGER REFERENCES clusters(cluster_id) ON DELETE SET NULL,
        suppress_auto INTEGER NOT NULL DEFAULT 0,
        is_manual INTEGER NOT NULL DEFAULT 0,
        UNIQUE(photo_id, face_idx)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id)",
    "CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id)",
    """
    CREATE TABLE IF NOT EXISTS reference_seeds (
        seed_id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
        source_key TEXT UNIQUE NOT NULL,
        anchor_year INTEGER,
        embedding_f16 BLOB NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS assignments (
        photo_id INTEGER NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
        person_id INTEGER NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
        face_id INTEGER REFERENCES faces(face_id) ON DELETE SET NULL,
        status TEXT NOT NULL,
        source TEXT NOT NULL,
        similarity REAL NOT NULL DEFAULT 0.0,
        is_anchor INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (photo_id, person_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_assignments_person_status ON assignments(person_id, status, similarity DESC)",
    """
    CREATE INDEX IF NOT EXISTS idx_assignments_person_cov
        ON assignments(person_id, status, similarity DESC, source, face_id, photo_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_assignments_cat_faces
        ON assignments(face_id, person_id)
        WHERE face_id IS NOT NULL AND status IN ('confirmed', 'auto_high')
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_assignments_confirmed_photo
        ON assignments(photo_id, person_id)
        WHERE status = 'confirmed'
    """,
    "CREATE INDEX IF NOT EXISTS idx_assignments_face ON assignments(face_id)",
    """
    CREATE TABLE IF NOT EXISTS school_classes (
        class_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL COLLATE NOCASE,
        archive_child TEXT NOT NULL DEFAULT '',
        start_date TEXT NOT NULL DEFAULT '',
        end_date TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS class_members (
        class_id INTEGER NOT NULL REFERENCES school_classes(class_id) ON DELETE CASCADE,
        person_id INTEGER NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
        PRIMARY KEY (class_id, person_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_class_members_person ON class_members(person_id)",
    """
    CREATE TABLE IF NOT EXISTS person_face_pressure (
        person_id INTEGER NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
        face_id INTEGER NOT NULL REFERENCES faces(face_id) ON DELETE CASCADE,
        weight REAL NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (person_id, face_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS person_assignment_counts (
        person_id INTEGER PRIMARY KEY REFERENCES people(person_id) ON DELETE CASCADE,
        confirmed_count INTEGER NOT NULL DEFAULT 0,
        auto_high_count INTEGER NOT NULL DEFAULT 0,
        album_cohesion_count INTEGER NOT NULL DEFAULT 0,
        candidate_count INTEGER NOT NULL DEFAULT 0,
        unsure_count INTEGER NOT NULL DEFAULT 0,
        rejected_count INTEGER NOT NULL DEFAULT 0,
        categorized_face_count INTEGER NOT NULL DEFAULT 0
    )
    """,
)

_COUNT_COLS = """
        person_id,
        confirmed_count,
        auto_high_count,
        album_cohesion_count,
        candidate_count,
        unsure_count,
        rejected_count,
        categorized_face_count
"""


def _count_values(alias: str) -> str:
    return f"""
        {alias}.person_id,
        CASE WHEN {alias}.status = 'confirmed' THEN 1 ELSE 0 END,
        CASE WHEN {alias}.status = 'auto_high' THEN 1 ELSE 0 END,
        CASE WHEN {alias}.status = 'auto_high' AND {alias}.source IN ('album_cohesion', 'model_trusted') THEN 1 ELSE 0 END,
        CASE WHEN {alias}.status = 'candidate' THEN 1 ELSE 0 END,
        CASE WHEN {alias}.status = 'unsure' THEN 1 ELSE 0 END,
        CASE WHEN {alias}.status = 'rejected' THEN 1 ELSE 0 END,
        CASE WHEN {alias}.face_id IS NOT NULL AND {alias}.status IN ('confirmed', 'auto_high') THEN 1 ELSE 0 END
"""


_COUNT_UPSERT_TAIL = """
    ON CONFLICT(person_id) DO UPDATE SET
        confirmed_count = confirmed_count + excluded.confirmed_count,
        auto_high_count = auto_high_count + excluded.auto_high_count,
        album_cohesion_count = album_cohesion_count + excluded.album_cohesion_count,
        candidate_count = candidate_count + excluded.candidate_count,
        unsure_count = unsure_count + excluded.unsure_count,
        rejected_count = rejected_count + excluded.rejected_count,
        categorized_face_count = categorized_face_count + excluded.categorized_face_count;
"""

_COUNT_DECREMENT = """
    UPDATE person_assignment_counts
    SET confirmed_count = confirmed_count - (CASE WHEN OLD.status = 'confirmed' THEN 1 ELSE 0 END),
        auto_high_count = auto_high_count - (CASE WHEN OLD.status = 'auto_high' THEN 1 ELSE 0 END),
        album_cohesion_count = album_cohesion_count - (CASE WHEN OLD.status = 'auto_high' AND OLD.source IN ('album_cohesion', 'model_trusted') THEN 1 ELSE 0 END),
        candidate_count = candidate_count - (CASE WHEN OLD.status = 'candidate' THEN 1 ELSE 0 END),
        unsure_count = unsure_count - (CASE WHEN OLD.status = 'unsure' THEN 1 ELSE 0 END),
        rejected_count = rejected_count - (CASE WHEN OLD.status = 'rejected' THEN 1 ELSE 0 END),
        categorized_face_count = categorized_face_count - (CASE WHEN OLD.face_id IS NOT NULL AND OLD.status IN ('confirmed', 'auto_high') THEN 1 ELSE 0 END)
    WHERE person_id = OLD.person_id;
"""

COUNTS_TRIGGERS: tuple[str, ...] = (
    f"""
    CREATE TRIGGER IF NOT EXISTS trg_assignments_ai_counts
    AFTER INSERT ON assignments
    BEGIN
        INSERT INTO person_assignment_counts ({_COUNT_COLS})
        VALUES ({_count_values("NEW")})
        {_COUNT_UPSERT_TAIL}
    END
    """,  # noqa: S608 - constant SQL fragments only
    f"""
    CREATE TRIGGER IF NOT EXISTS trg_assignments_ad_counts
    AFTER DELETE ON assignments
    BEGIN
        {_COUNT_DECREMENT}
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS trg_assignments_au_counts
    AFTER UPDATE OF person_id, face_id, status, source ON assignments
    BEGIN
        {_COUNT_DECREMENT}
        INSERT INTO person_assignment_counts ({_COUNT_COLS})
        VALUES ({_count_values("NEW")})
        {_COUNT_UPSERT_TAIL}
    END
    """,  # noqa: S608 - constant SQL fragments only
)

REBUILD_COUNTS_SQL = f"""
    INSERT INTO person_assignment_counts ({_COUNT_COLS})
    SELECT
        person_id,
        SUM(CASE WHEN status = 'confirmed' THEN 1 ELSE 0 END),
        SUM(CASE WHEN status = 'auto_high' THEN 1 ELSE 0 END),
        SUM(CASE WHEN status = 'auto_high' AND source IN ('album_cohesion', 'model_trusted') THEN 1 ELSE 0 END),
        SUM(CASE WHEN status = 'candidate' THEN 1 ELSE 0 END),
        SUM(CASE WHEN status = 'unsure' THEN 1 ELSE 0 END),
        SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END),
        SUM(CASE WHEN face_id IS NOT NULL AND status IN ('confirmed', 'auto_high') THEN 1 ELSE 0 END)
    FROM assignments
    GROUP BY person_id
"""  # noqa: S608 - constant SQL fragments only

ASSIGNMENT_STATUSES: tuple[str, ...] = tuple(get_args(AssignmentStatus))
ASSIGNMENT_SOURCES: tuple[str, ...] = tuple(get_args(AssignmentSource))
PERSON_ROLES: tuple[str, ...] = tuple(get_args(PersonRole))


def _sql_list(values: tuple[str, ...]) -> str:
    for v in values:
        if not v.replace("_", "").isalnum():  # literal enum values only; never user input
            raise ValueError(f"Unsafe enum literal {v!r}")
    return ", ".join(f"'{v}'" for v in values)


def validation_trigger_statements() -> tuple[str, ...]:
    """`BEFORE INSERT/UPDATE` guards enforcing the `status` / `source` / `role` enums.

    SQLite cannot add `CHECK` constraints without a full table rebuild (which would also have to
    re-create every FK, index and counts trigger on the 120k-row live tables); validation triggers
    give the same fail-fast guarantee for new writes without touching existing rows.
    """
    st = _sql_list(ASSIGNMENT_STATUSES)
    src = _sql_list(ASSIGNMENT_SOURCES)
    roles = _sql_list(PERSON_ROLES)
    a_when = f"NEW.status NOT IN ({st}) OR NEW.source NOT IN ({src})"
    p_when = f"NEW.role NOT IN ({roles})"
    return (
        "DROP TRIGGER IF EXISTS trg_assignments_validate_ins",
        "DROP TRIGGER IF EXISTS trg_assignments_validate_upd",
        "DROP TRIGGER IF EXISTS trg_people_validate_ins",
        "DROP TRIGGER IF EXISTS trg_people_validate_upd",
        f"""
        CREATE TRIGGER trg_assignments_validate_ins BEFORE INSERT ON assignments
        WHEN {a_when}
        BEGIN SELECT RAISE(ABORT, 'assignments: invalid status or source'); END
        """,
        f"""
        CREATE TRIGGER trg_assignments_validate_upd BEFORE UPDATE OF status, source ON assignments
        WHEN {a_when}
        BEGIN SELECT RAISE(ABORT, 'assignments: invalid status or source'); END
        """,
        f"""
        CREATE TRIGGER trg_people_validate_ins BEFORE INSERT ON people
        WHEN {p_when}
        BEGIN SELECT RAISE(ABORT, 'people: invalid role'); END
        """,
        f"""
        CREATE TRIGGER trg_people_validate_upd BEFORE UPDATE OF role ON people
        WHEN {p_when}
        BEGIN SELECT RAISE(ABORT, 'people: invalid role'); END
        """,
    )


# ----------------------------------------------------------------------------------------------
# Migrations
# ----------------------------------------------------------------------------------------------


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _m1_baseline(conn: sqlite3.Connection) -> None:
    """Base tables/indexes + the columns legacy (pre-versioning) databases may lack."""
    for stmt in BASE_TABLES:
        conn.execute(stmt)
    face_cols = _columns(conn, "faces")
    if "suppress_auto" not in face_cols:
        conn.execute("ALTER TABLE faces ADD COLUMN suppress_auto INTEGER NOT NULL DEFAULT 0")
    if "is_manual" not in face_cols:
        conn.execute("ALTER TABLE faces ADD COLUMN is_manual INTEGER NOT NULL DEFAULT 0")
    if "birth_date" not in _columns(conn, "people"):
        conn.execute("ALTER TABLE people ADD COLUMN birth_date TEXT NOT NULL DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_suppressed ON faces(face_id) WHERE suppress_auto != 0")


def _m2_counts_triggers(conn: sqlite3.Connection) -> None:
    """(Re)create the trigger-maintained `person_assignment_counts` (v2 semantics) and rebuild it.

    v2: `album_cohesion_count` also counts `model_trusted` auto matches (no review needed).
    """
    for trg in ("trg_assignments_ai_counts", "trg_assignments_ad_counts", "trg_assignments_au_counts"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trg}")
    for stmt in COUNTS_TRIGGERS:
        conn.execute(stmt)
    conn.execute("DELETE FROM person_assignment_counts")
    conn.execute(REBUILD_COUNTS_SQL)
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('person_assignment_counts_v1', '2') "
        "ON CONFLICT(key) DO UPDATE SET value = '2'"
    )


def _m3_ignore_role_repair(conn: sqlite3.Connection) -> None:
    """One-time repair (previously executed on every open): an `ignore`-role person must not hold a
    face that a real person also holds."""
    conn.execute(
        """
        DELETE FROM assignments
        WHERE face_id IS NOT NULL
          AND status != 'rejected'
          AND person_id IN (SELECT person_id FROM people WHERE role = 'ignore')
          AND EXISTS (
              SELECT 1
              FROM assignments a2
              JOIN people pe2 ON pe2.person_id = a2.person_id
              WHERE a2.face_id = assignments.face_id
                AND a2.status != 'rejected'
                AND pe2.role != 'ignore'
          )
        """
    )


def _m4_content_hash_and_face_refs(conn: sqlite3.Connection) -> None:
    """`photos.content_hash` (cheap head/tail fingerprint) + `ON DELETE SET NULL` semantics for the
    plain-INTEGER face references `people.cover_face_id` / `clusters.medoid_face_id`."""
    if "content_hash" not in _columns(conn, "photos"):
        conn.execute("ALTER TABLE photos ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clusters_medoid ON clusters(medoid_face_id)")
    conn.execute("DROP TRIGGER IF EXISTS trg_faces_ad_refs")
    conn.execute(
        """
        CREATE TRIGGER trg_faces_ad_refs AFTER DELETE ON faces
        BEGIN
            UPDATE people SET cover_face_id = NULL WHERE cover_face_id = OLD.face_id;
            UPDATE clusters SET medoid_face_id = NULL WHERE medoid_face_id = OLD.face_id;
        END
        """
    )
    conn.execute(
        "UPDATE people SET cover_face_id = NULL WHERE cover_face_id IS NOT NULL "
        "AND cover_face_id NOT IN (SELECT face_id FROM faces)"
    )
    conn.execute(
        "UPDATE clusters SET medoid_face_id = NULL WHERE medoid_face_id IS NOT NULL "
        "AND medoid_face_id NOT IN (SELECT face_id FROM faces)"
    )


def _m5_enum_validation(conn: sqlite3.Connection) -> None:
    for stmt in validation_trigger_statements():
        conn.execute(stmt)


def _m6_label_keeps(conn: sqlite3.Connection) -> None:
    """`label_keeps`: confirmed labels the user reviewed and kept although the label-consistency
    guard flagged them as suspicious (see `faces/label_guard.py`); they count as normal anchors /
    kNN voters again."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS label_keeps (
            face_id INTEGER NOT NULL REFERENCES faces(face_id) ON DELETE CASCADE,
            person_id INTEGER NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
            kept_at TEXT NOT NULL,
            PRIMARY KEY (face_id, person_id)
        )
        """
    )


def _m7_album_index(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_photos_child_album ON photos(archive_child, album_leaf)"
    )


Migration = tuple[int, str, Callable[[sqlite3.Connection], None]]

MIGRATIONS: tuple[Migration, ...] = (
    (1, "baseline", _m1_baseline),
    (2, "counts_triggers_v2", _m2_counts_triggers),
    (3, "ignore_role_repair", _m3_ignore_role_repair),
    (4, "content_hash_and_face_refs", _m4_content_hash_and_face_refs),
    (5, "enum_validation_triggers", _m5_enum_validation),
    (6, "label_keeps", _m6_label_keeps),
    (7, "album_index", _m7_album_index),
)

SCHEMA_VERSION = MIGRATIONS[-1][0]


def read_schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.OperationalError:
        return 0
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    """Run every pending migration (each in its own `BEGIN IMMEDIATE` transaction); return versions applied.

    Safe against a concurrent process (e.g. `faces scan` + `faces ui`) migrating the same file: the
    version is re-read after the write lock is taken.
    """
    if conn.in_transaction:
        conn.commit()
    applied: list[int] = []
    if read_schema_version(conn) >= SCHEMA_VERSION:
        return applied
    for version, _name, fn in MIGRATIONS:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            if read_schema_version(conn) >= version:
                conn.rollback()
                continue
            fn(conn)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(version),),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        applied.append(version)
    return applied
