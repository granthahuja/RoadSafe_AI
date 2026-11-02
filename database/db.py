import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "roadsafe.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            media_type TEXT,
            accident_probability REAL,
            normal_probability REAL,
            accident_detected INTEGER,
            location TEXT,
            vehicle_count INTEGER DEFAULT 0,
            person_count INTEGER DEFAULT 0,
            vehicle_types TEXT DEFAULT '',
            visual_activity TEXT DEFAULT 'Not available',
            incident_priority TEXT DEFAULT 'Not assessed',
            review_flag INTEGER DEFAULT 0,
            review_reason TEXT DEFAULT ''
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS rescue_cases (
            case_id TEXT PRIMARY KEY,
            incident_id INTEGER,
            hospital_name TEXT DEFAULT 'Not assigned',
            status TEXT DEFAULT 'Awaiting operator review',
            operator_verified INTEGER DEFAULT 0,
            response_status TEXT DEFAULT 'Not initiated',
            created_at TEXT,
            acknowledgement_timestamp TEXT,
            assistance_timestamp TEXT,
            FOREIGN KEY (incident_id) REFERENCES incidents(id)
        )
    """)

    existing_rescue_columns = {
        row[1] for row in conn.execute(
            "PRAGMA table_info(rescue_cases)"
        ).fetchall()
    }

    rescue_new_columns = {
        "filename": "TEXT DEFAULT ''",
        "accident_probability": "REAL DEFAULT 0",
        "incident_priority": "TEXT DEFAULT 'Not assessed'",
        "review_flag": "INTEGER DEFAULT 0",
        "response_timestamp": "TEXT",
        "acknowledgement_pending": "INTEGER DEFAULT 0"
    }

    for column, data_type in rescue_new_columns.items():
        if column not in existing_rescue_columns:
            conn.execute(
                f"ALTER TABLE rescue_cases ADD COLUMN {column} {data_type}"
            )

    # Safely add new columns to existing databases.
    existing_columns = {
        row[1] for row in conn.execute(
            "PRAGMA table_info(incidents)"
        ).fetchall()
    }

    new_columns = {
        "vehicle_count": "INTEGER DEFAULT 0",
        "person_count": "INTEGER DEFAULT 0",
        "vehicle_types": "TEXT DEFAULT ''",
        "visual_activity": "TEXT DEFAULT 'Not available'",
        "incident_priority": "TEXT DEFAULT 'Not assessed'",
        "review_flag": "INTEGER DEFAULT 0",
        "review_reason": "TEXT DEFAULT ''"
    }

    for column, data_type in new_columns.items():
        if column not in existing_columns:
            conn.execute(
                f"ALTER TABLE incidents ADD COLUMN {column} {data_type}"
            )

    conn.commit()
    conn.close()


def save_incident(data):
    conn = get_connection()

    conn.execute("""
        INSERT INTO incidents (
            timestamp,
            filename,
            media_type,
            accident_probability,
            normal_probability,
            accident_detected,
            location,
            vehicle_count,
            person_count,
            vehicle_types,
            visual_activity,
            incident_priority,
            review_flag,
            review_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("timestamp"),
        data.get("filename"),
        data.get("media_type"),
        data.get("accident_probability"),
        data.get("normal_probability"),
        int(data.get("accident", False)),
        data.get("location", "Not available"),
        data.get("vehicle_count", 0),
        data.get("person_count", 0),
        ", ".join(data.get("vehicle_types", [])),
        data.get("visual_activity", "Not available"),
        data.get("incident_priority", "Not assessed"),
        int(data.get("review_flag", False)),
        data.get("review_reason", "")
    ))

    conn.commit()
    conn.close()


def get_all_incidents():
    conn = get_connection()

    rows = conn.execute("""
        SELECT * FROM incidents
        ORDER BY id DESC
    """).fetchall()

    conn.close()
    return rows

def save_rescue_case(case):
    conn = get_connection()

    conn.execute("""
        INSERT OR REPLACE INTO rescue_cases (
            case_id,
            hospital_name,
            status,
            operator_verified,
            response_status,
            created_at,
            acknowledgement_timestamp,
            assistance_timestamp,
            filename,
            accident_probability,
            incident_priority,
            review_flag,
            response_timestamp,
            acknowledgement_pending
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        case.get("case_id"),
        case.get("hospital_name", "Not assigned"),
        case.get("status", "Awaiting operator review"),
        int(case.get("operator_verified", False)),
        case.get("response_status", "Not initiated"),
        case.get("timestamp"),
        case.get("acknowledgement_timestamp"),
        case.get("assistance_timestamp"),
        case.get("filename", ""),
        case.get("accident_probability", 0.0),
        case.get("incident_priority", "Not assessed"),
        int(case.get("review_flag", False)),
        case.get("response_timestamp"),
        int(case.get("acknowledgement_pending", False))
    ))

    conn.commit()
    conn.close()


def update_rescue_case(case):
    conn = get_connection()

    conn.execute("""
        UPDATE rescue_cases
        SET hospital_name = ?,
            status = ?,
            operator_verified = ?,
            response_status = ?,
            acknowledgement_timestamp = ?,
            assistance_timestamp = ?,
            response_timestamp = ?,
            acknowledgement_pending = ?
        WHERE case_id = ?
    """, (
        case.get("hospital_name", "Not assigned"),
        case.get("status"),
        int(case.get("operator_verified", False)),
        case.get("response_status"),
        case.get("acknowledgement_timestamp"),
        case.get("assistance_timestamp"),
        case.get("response_timestamp"),
        int(case.get("acknowledgement_pending", False)),
        case.get("case_id")
    ))

    conn.commit()
    conn.close()


def get_all_rescue_cases():
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT * FROM rescue_cases
        ORDER BY created_at DESC
    """).fetchall()

    cases = [dict(row) for row in rows]

    conn.close()
    return cases