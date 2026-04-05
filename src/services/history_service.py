import json
from typing import Dict, List, Optional

from src.core.database import get_connection


def severity_number_to_label(severity: int) -> str:
    try:
        severity = int(severity)
    except (TypeError, ValueError):
        return "Moderate"

    if severity <= 1:
        return "Mild"
    if severity == 2:
        return "Moderate"
    return "Severe"


def create_prediction_session(user_id: int, raw_input: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO sessions (user_id, raw_input)
        VALUES (?, ?)
        """,
        (user_id, raw_input)
    )

    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id


def save_session_symptoms(session_id: int, symptoms: List[Dict]):
    conn = get_connection()
    cursor = conn.cursor()

    for item in symptoms:
        cursor.execute(
            """
            INSERT INTO session_symptoms (session_id, symptom, severity_input, final_severity)
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                item["symptom"],
                int(item["severity"]),
                int(item["severity"]),
            )
        )

    conn.commit()
    conn.close()


def save_session_prediction(
    session_id: int,
    mode: str,
    predicted_disease: str,
    confidence: float,
    ai_response: str,
    description: str,
    precautions: List[str],
    top_k_predictions: List[Dict],
    severity_summary: Dict,
    history_context: Optional[Dict] = None,
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            session_id,
            mode,
            predicted_disease,
            confidence,
            ai_response,
            description,
            precautions_json,
            top_k_predictions_json,
            severity_total,
            severity_avg,
            severity_condition,
            history_context_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            mode,
            predicted_disease,
            float(confidence),
            ai_response,
            description,
            json.dumps(precautions or [], ensure_ascii=False),
            json.dumps(top_k_predictions or [], ensure_ascii=False),
            int(severity_summary.get("total", 0)),
            float(severity_summary.get("avg", 0)),
            severity_summary.get("condition", "Mild"),
            json.dumps(history_context or {}, ensure_ascii=False),
        )
    )

    conn.commit()
    conn.close()


def has_previous_history(user_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM sessions
        WHERE user_id = ?
        """,
        (user_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return False

    return int(row["total"]) > 0


def get_previous_sessions_for_context(user_id: int, limit: int = 5):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT s.id, s.created_at, s.raw_input
        FROM sessions s
        WHERE s.user_id = ?
        ORDER BY s.id DESC
        LIMIT ?
        """,
        (user_id, limit)
    )

    session_rows = cursor.fetchall()
    result = []

    for session_row in session_rows:
        session_id = session_row["id"]

        cursor.execute(
            """
            SELECT symptom, severity_input, final_severity
            FROM session_symptoms
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        )
        symptoms_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT mode, predicted_disease, confidence, ai_response,
                   description, precautions_json, top_k_predictions_json,
                   severity_total, severity_avg, severity_condition,
                   history_context_json
            FROM predictions
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        )
        prediction_rows = cursor.fetchall()

        current_prediction = None
        history_prediction = None

        for pred in prediction_rows:
            item = dict(pred)
            if item["mode"] == "current_session":
                current_prediction = item
            elif item["mode"] == "history_aware":
                history_prediction = item

        result.append({
            "session_id": session_id,
            "created_at": session_row["created_at"],
            "raw_input": session_row["raw_input"],
            "symptoms": [
                {
                    "name": row["symptom"],
                    "severity": severity_number_to_label(row["final_severity"]),
                    "severityValue": int(row["final_severity"]),
                }
                for row in symptoms_rows
            ],
            "prediction": current_prediction["predicted_disease"] if current_prediction else None,
            "current_prediction": current_prediction,
            "history_prediction": history_prediction,
        })

    conn.close()
    return result


def get_recent_history_symptoms(
    user_id: int,
    limit_sessions: int = 3,
    max_items: int = 17,
    exclude_session_id: Optional[int] = None,
):
    """
    ترجع أعراض الجلسات السابقة بترتيب زمني:
    الأقدم -> الأحدث
    ثم نقصها إلى آخر max_items فقط.
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT s.id
        FROM sessions s
        WHERE s.user_id = ?
    """
    params = [user_id]

    if exclude_session_id is not None:
        query += " AND s.id != ?"
        params.append(exclude_session_id)

    query += """
        ORDER BY s.id DESC
        LIMIT ?
    """
    params.append(limit_sessions)

    cursor.execute(query, tuple(params))
    session_rows = cursor.fetchall()

    # رجعهم للأقدم -> الأحدث
    session_ids = [row["id"] for row in reversed(session_rows)]

    merged_history = []

    for session_id in session_ids:
        cursor.execute(
            """
            SELECT symptom, final_severity
            FROM session_symptoms
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        )
        symptom_rows = cursor.fetchall()

        for row in symptom_rows:
            merged_history.append({
                "symptom": row["symptom"],
                "severity": int(row["final_severity"]),
            })

    conn.close()

    return merged_history[-max_items:]


def merge_history_with_current(
    history_symptoms: List[Dict],
    current_symptoms: List[Dict],
    max_len: int = 17
):
    """
    history-aware sequence:
    previous history first -> current symptoms last
    """
    merged = []

    for item in history_symptoms:
        merged.append({
            "symptom": item["symptom"],
            "severity": int(item["severity"]),
        })

    for item in current_symptoms:
        merged.append({
            "symptom": item["symptom"],
            "severity": int(item["severity"]),
        })

    return merged[-max_len:]


def get_user_history(user_id: int):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT s.id AS session_id, s.created_at
        FROM sessions s
        WHERE s.user_id = ?
        ORDER BY s.id DESC
        """,
        (user_id,)
    )

    sessions = cursor.fetchall()
    result = []

    for session_row in sessions:
        session_id = session_row["session_id"]

        cursor.execute(
            """
            SELECT symptom, final_severity
            FROM session_symptoms
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        )
        symptom_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT mode, predicted_disease, confidence, ai_response,
                   description, precautions_json, top_k_predictions_json,
                   severity_total, severity_avg, severity_condition,
                   history_context_json
            FROM predictions
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        )
        prediction_rows = cursor.fetchall()

        current_prediction = None
        history_prediction = None

        for pred in prediction_rows:
            item = dict(pred)
            if item["mode"] == "current_session":
                current_prediction = item
            elif item["mode"] == "history_aware":
                history_prediction = item

        parsed_symptoms = [
            {
                "name": row["symptom"],
                "severity": severity_number_to_label(row["final_severity"]),
                "severityValue": int(row["final_severity"]),
            }
            for row in symptom_rows
        ]

        severity_score = sum(int(item["severityValue"]) for item in parsed_symptoms)

        main_prediction = history_prediction or current_prediction

        result.append({
            "id": session_id,
            "timestamp": session_row["created_at"],
            "date": session_row["created_at"],
            "symptoms": parsed_symptoms,
            "prediction": main_prediction["predicted_disease"] if main_prediction else None,
            "confidence": round(float(main_prediction["confidence"]) * 100, 2) if main_prediction else 0,
            "aiResponse": main_prediction["ai_response"] if main_prediction else "",
            "severityScore": severity_score,
            "currentPrediction": current_prediction,
            "historyAwarePrediction": history_prediction,
        })

    conn.close()
    return result