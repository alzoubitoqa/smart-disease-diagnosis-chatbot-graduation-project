import json
from typing import Dict, List, Optional
from collections import Counter

from src.core.database import get_connection


def severity_number_to_label(severity: int) -> str:
    try:
        severity = int(severity)
    except (TypeError, ValueError):
        return "Moderate"

    if severity <= 2:
        return "Mild"
    if severity == 5:
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


def _deduplicate_keep_max_severity(symptoms: List[Dict]) -> List[Dict]:
    """
    Merge duplicate symptoms and keep the maximum severity.
    Preserve first appearance order as much as possible.
    """
    ordered = []
    seen = {}

    for item in symptoms:
        symptom = str(item["symptom"]).strip().lower()
        severity = int(item["severity"])

        if symptom not in seen:
            seen[symptom] = {
                "symptom": symptom,
                "severity": severity
            }
            ordered.append(symptom)
        else:
            seen[symptom]["severity"] = max(seen[symptom]["severity"], severity)

    return [seen[symptom] for symptom in ordered]


def _rank_history_symptoms(
    history_symptoms: List[Dict],
    current_symptoms: List[Dict]
) -> List[Dict]:
    """
    Score history symptoms so we only keep the most useful ones.
    Priority:
    1) symptoms also present in current session
    2) repeated symptoms in history
    3) more recent appearance (last occurrences naturally come later in input)
    4) higher severity
    """
    current_names = {
        str(item["symptom"]).strip().lower()
        for item in current_symptoms
    }

    symptom_counter = Counter(
        str(item["symptom"]).strip().lower()
        for item in history_symptoms
    )

    indexed_items = []
    for idx, item in enumerate(history_symptoms):
        symptom = str(item["symptom"]).strip().lower()
        severity = int(item["severity"])

        overlap_bonus = 100 if symptom in current_names else 0
        frequency_bonus = symptom_counter[symptom] * 10
        recency_bonus = idx
        severity_bonus = severity

        score = overlap_bonus + frequency_bonus + recency_bonus + severity_bonus

        indexed_items.append({
            "symptom": symptom,
            "severity": severity,
            "score": score
        })

    best_per_symptom = {}
    for item in indexed_items:
        symptom = item["symptom"]
        if symptom not in best_per_symptom:
            best_per_symptom[symptom] = item
        else:
            current_best = best_per_symptom[symptom]
            if item["score"] > current_best["score"]:
                best_per_symptom[symptom] = item
            elif item["score"] == current_best["score"]:
                current_best["severity"] = max(current_best["severity"], item["severity"])
                best_per_symptom[symptom] = current_best

    ranked = list(best_per_symptom.values())
    ranked.sort(key=lambda x: x["score"], reverse=True)

    return [
        {
            "symptom": item["symptom"],
            "severity": item["severity"]
        }
        for item in ranked
    ]


def get_recent_history_symptoms(
    user_id: int,
    limit_sessions: int = 3,
    max_items: int = 17,
    exclude_session_id: Optional[int] = None,
):
    """
    Return a cleaned history symptom list from recent sessions.

    Improvements over old version:
    - keep order from older -> newer sessions
    - deduplicate repeated symptoms
    - keep maximum severity per repeated symptom
    - avoid returning an oversized noisy history block
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
                "symptom": str(row["symptom"]).strip().lower(),
                "severity": int(row["final_severity"]),
            })

    conn.close()

    if not merged_history:
        return []

    # Clean duplicates and keep strongest version
    cleaned_history = _deduplicate_keep_max_severity(merged_history)

    # Keep only the most recent max_items after cleaning
    return cleaned_history[-max_items:]


def merge_history_with_current(
    history_symptoms: List[Dict],
    current_symptoms: List[Dict],
    max_len: int = 17
):
    """
    Smart history-aware merge.

    Logic:
    - Current symptoms are always highest priority
    - History is filtered and ranked before merging
    - Duplicate symptoms are merged with max severity
    - Only a limited number of history symptoms are added
    - Final sequence prefers relevance over raw concatenation
    """
    if not current_symptoms:
        return []

    # Normalize current first
    current_clean = [
        {
            "symptom": str(item["symptom"]).strip().lower(),
            "severity": int(item["severity"]),
        }
        for item in current_symptoms
    ]
    current_clean = _deduplicate_keep_max_severity(current_clean)

    if not history_symptoms:
        return current_clean[-max_len:]

    history_clean = [
        {
            "symptom": str(item["symptom"]).strip().lower(),
            "severity": int(item["severity"]),
        }
        for item in history_symptoms
    ]

    # Rank history symptoms by usefulness relative to current session
    ranked_history = _rank_history_symptoms(history_clean, current_clean)

    current_names = {item["symptom"] for item in current_clean}

    overlapping_history = [item for item in ranked_history if item["symptom"] in current_names]
    non_overlapping_history = [item for item in ranked_history if item["symptom"] not in current_names]

    # Keep history compact so it helps instead of dominating
    max_history_items = max(2, min(6, max_len - len(current_clean)))
    selected_history = []

    for item in overlapping_history:
        if len(selected_history) >= max_history_items:
            break
        selected_history.append(item)

    for item in non_overlapping_history:
        if len(selected_history) >= max_history_items:
            break
        selected_history.append(item)

    # Merge with current placed last so model sees the latest state clearly
    merged = selected_history + current_clean

    # Deduplicate again after merge and keep strongest severity
    merged = _deduplicate_keep_max_severity(merged)

    # Ensure current symptoms remain at the end / highest importance
    final_history_part = [item for item in merged if item["symptom"] not in current_names]
    final_current_part = []

    current_map = {item["symptom"]: item for item in current_clean}
    merged_map = {item["symptom"]: item for item in merged}

    for item in current_clean:
        symptom = item["symptom"]
        if symptom in merged_map:
            final_current_part.append({
                "symptom": symptom,
                "severity": max(
                    int(current_map[symptom]["severity"]),
                    int(merged_map[symptom]["severity"])
                )
            })

    final_sequence = final_history_part + final_current_part

    return final_sequence[-max_len:]


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