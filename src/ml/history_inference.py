from copy import deepcopy


LOW_CONFIDENCE_THRESHOLD = 0.30
MIN_REPEATED_SYMPTOMS = 2
MIN_SIMILAR_SESSIONS = 1
HISTORY_BOOST = 0.08


def _safe_list(value):
    return value if isinstance(value, list) else []


def _safe_dict(value):
    return value if isinstance(value, dict) else {}


def should_apply_history(current_result: dict, history_context: dict) -> dict:
    confidence = float(current_result.get("confidence", 0.0))
    repeated_symptoms = _safe_list(history_context.get("repeated_symptoms"))
    similar_sessions_count = int(history_context.get("similar_sessions_count", 0))
    last_predicted_diseases = _safe_list(history_context.get("last_predicted_diseases"))

    reasons = []

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("current_confidence_is_low")

    if len(repeated_symptoms) >= MIN_REPEATED_SYMPTOMS:
        reasons.append("repeated_symptoms_detected")

    if similar_sessions_count >= MIN_SIMILAR_SESSIONS:
        reasons.append("similar_previous_sessions_detected")

    if last_predicted_diseases:
        reasons.append("recent_related_predictions_exist")

    apply_history = (
        confidence < LOW_CONFIDENCE_THRESHOLD
        and len(repeated_symptoms) >= MIN_REPEATED_SYMPTOMS
        and similar_sessions_count >= MIN_SIMILAR_SESSIONS
    )

    return {
        "apply_history": apply_history,
        "reasons": reasons,
        "confidence": confidence,
        "repeated_symptoms_count": len(repeated_symptoms),
        "similar_sessions_count": similar_sessions_count,
    }


def rerank_predictions_with_history(prediction_result: dict, history_context: dict):
    disease_frequency = _safe_dict(history_context.get("disease_frequency"))
    original_predictions = deepcopy(_safe_list(prediction_result.get("top_k_predictions", [])))

    reranked = []
    for item in original_predictions:
        disease = item.get("disease")
        base_confidence = float(item.get("confidence", 0.0))
        freq = int(disease_frequency.get(disease, 0))

        adjusted_confidence = base_confidence + (freq * HISTORY_BOOST)

        reranked.append({
            **item,
            "base_confidence": base_confidence,
            "history_boost": round(freq * HISTORY_BOOST, 4),
            "confidence": adjusted_confidence,
        })

    reranked.sort(key=lambda x: x["confidence"], reverse=True)

    total = sum(item["confidence"] for item in reranked)
    if total > 0:
        for item in reranked:
            item["confidence"] = item["confidence"] / total

    new_prediction_result = deepcopy(prediction_result)
    new_prediction_result["top_k_predictions"] = reranked
    new_prediction_result["top_prediction"] = (
        reranked[0] if reranked else prediction_result.get("top_prediction")
    )

    old_top1 = original_predictions[0]["disease"] if original_predictions else None
    new_top1 = reranked[0]["disease"] if reranked else None

    old_ranking = [item.get("disease") for item in original_predictions]
    new_ranking = [item.get("disease") for item in reranked]

    return {
        "prediction_result": new_prediction_result,
        "history_changed_top1": old_top1 != new_top1,
        "history_changed_ranking": old_ranking != new_ranking,
        "old_top1": old_top1,
        "new_top1": new_top1,
        "old_ranking": old_ranking,
        "new_ranking": new_ranking,
    }


def build_history_decision(current_result: dict, history_context: dict):
    decision = should_apply_history(current_result, history_context)

    if not decision["apply_history"]:
        return {
            "history_applied": False,
            "history_changed_top1": False,
            "history_changed_ranking": False,
            "history_reason": "History reviewed, but no ranking change was applied.",
            "prediction_result": current_result["prediction_result"],
        }

    rerank_output = rerank_predictions_with_history(
        current_result["prediction_result"],
        history_context
    )

    reason = (
        "History was applied because current confidence was low and similar previous "
        "sessions with repeated symptoms were found."
    )

    return {
        "history_applied": True,
        "history_changed_top1": rerank_output["history_changed_top1"],
        "history_changed_ranking": rerank_output["history_changed_ranking"],
        "history_reason": reason,
        "prediction_result": rerank_output["prediction_result"],
    }