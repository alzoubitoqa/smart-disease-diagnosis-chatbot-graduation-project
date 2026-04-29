from src.ml.inference import predictor
from src.core.groq_client import generate_medical_response
from src.knowledge_base.knowledge_base import (
    get_disease_description,
    get_disease_precautions,
    calculate_severity,
)
from src.services.history_service import (
    create_prediction_session,
    save_session_symptoms,
    save_session_prediction,
    has_previous_history,
    get_recent_history_symptoms,
    merge_history_with_current,
)
from src.services.symptom_parser_service import extract_symptoms_from_text


LOW_CONFIDENCE_THRESHOLD = 0.30
VERY_LOW_CONFIDENCE_THRESHOLD = 0.20

# أعراض طارئة لازم يظهر معها تنبيه واضح
EMERGENCY_SYMPTOMS = {
    "chest_pain",
    "breathlessness",
    "sweating",
    "vomiting",
    "loss_of_balance",
    "unsteadiness",
    "spinning_movements",
    "slurred_speech",
    "weakness_in_limbs",
}

# أمراض لو ظهرت حتى بثقة منخفضة، لا نخفيها بسهولة
HIGH_RISK_DISEASES = {
    "heart_attack",
    "paralysis_(brain_hemorrhage)",
    "paralysis_(brain_haemorrhage)",
    "(vertigo)_paroymsal_positional_vertigo",
    "pneumonia",
    "hypertension",
}


def normalize_disease_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def is_emergency_pattern(symptoms_data: list) -> bool:
    symptom_names = {
        str(item["symptom"]).strip().lower()
        for item in symptoms_data
    }

    # نمط شديد مهم جدًا
    heart_attack_pattern = {
        "chest_pain",
        "breathlessness",
        "sweating",
    }

    severe_vertigo_pattern = {
        "vomiting",
        "nausea",
        "spinning_movements",
        "loss_of_balance",
        "unsteadiness",
    }

    if len(symptom_names.intersection(EMERGENCY_SYMPTOMS)) >= 3:
        return True

    if heart_attack_pattern.issubset(symptom_names):
        return True

    if len(symptom_names.intersection(severe_vertigo_pattern)) >= 4:
        return True

    return False


def enrich_prediction_result(prediction_result: dict):
    top_prediction = prediction_result["top_prediction"]
    predicted_disease = top_prediction["disease"]

    description = get_disease_description(predicted_disease)
    precautions = get_disease_precautions(predicted_disease)

    top_prediction["description"] = description
    top_prediction["precautions"] = precautions

    for item in prediction_result["top_k_predictions"]:
        disease_name = item["disease"]
        item["description"] = get_disease_description(disease_name)
        item["precautions"] = get_disease_precautions(disease_name)

    confidence = float(top_prediction["confidence"])
    is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    is_very_low_confidence = confidence < VERY_LOW_CONFIDENCE_THRESHOLD

    return (
        predicted_disease,
        description,
        precautions,
        confidence,
        is_low_confidence,
        is_very_low_confidence,
    )


def normalize_payload_symptoms(payload):
    """
    Convert payload symptoms into the internal standard format:
    [
        {"symptom": "headache", "severity": 3},
        {"symptom": "vomiting", "severity": 5},
    ]
    """
    if getattr(payload, "symptoms", None):
        normalized = []
        for s in payload.symptoms:
            symptom_name = str(s.symptom).strip().lower().replace(" ", "_")
            severity_value = int(s.severity)

            normalized.append({
                "symptom": symptom_name,
                "severity": severity_value
            })
        return normalized

    extracted = extract_symptoms_from_text(payload.user_text)

    return [
        {
            "symptom": str(symptom).strip().lower().replace(" ", "_"),
            "severity": int(getattr(payload, "default_severity", 4))
        }
        for symptom in extracted
    ]


def append_history_note(ai_response: str, history_symptoms: list, merged_symptoms: list) -> str:
    if not history_symptoms:
        return ai_response

    history_names = [item["symptom"].replace("_", " ") for item in history_symptoms]
    merged_names = [item["symptom"].replace("_", " ") for item in merged_symptoms]

    note = (
        f"History Note: Previous symptom history was combined with the current session. "
        f"Historical symptoms used: {', '.join(history_names)}. "
        f"Final merged sequence used for history-aware prediction: {', '.join(merged_names)}."
    )

    return ai_response + "\n\n" + note


def build_prediction_output(symptoms_data, payload, mode: str = "current_session", history_used: bool = False):
    """
    symptoms_data expected format:
    [
        {"symptom": "vomiting", "severity": 5},
        {"symptom": "headache", "severity": 3},
    ]
    """

    symptom_names = [item["symptom"] for item in symptoms_data]
    severity_values = [int(item["severity"]) for item in symptoms_data]

    prediction_result = predictor.predict(
        symptoms_list=symptom_names,
        severities_list=severity_values,
        top_k=3
    )

    (
        predicted_disease,
        description,
        precautions,
        confidence,
        is_low_confidence,
        is_very_low_confidence,
    ) = enrich_prediction_result(prediction_result)

    user_severities = {
        item["symptom"]: int(item["severity"])
        for item in symptoms_data
    }

    severity_result = calculate_severity(symptom_names, user_severities)

    ai_response = generate_medical_response(
        user_input=payload.user_text,
        prediction_result=prediction_result,
        is_low_confidence=is_low_confidence
    )

    normalized_predicted = normalize_disease_name(predicted_disease)
    emergency_flag = is_emergency_pattern(symptoms_data)
    keep_top_prediction_visible = (
        normalized_predicted in HIGH_RISK_DISEASES
        or emergency_flag
    )

    # بدل إخفاء المرض دائمًا، نخفيه فقط إذا الثقة منخفضة جدًا
    # وما في أي إشارة أن الحالة خطرة أو top1 مهم
    if is_very_low_confidence and not keep_top_prediction_visible:
        predicted_disease = "uncertain_case"
        description = (
            "The model could not determine a reliable condition from the current symptom combination. "
            "Multiple possible conditions may exist."
        )
        precautions = [
            "monitor symptoms carefully",
            "seek medical advice if symptoms persist",
            "do not rely on this result as a final diagnosis"
        ]

    # إذا الحالة خطرة، أضيفي تنبيه واضح
    if emergency_flag:
        emergency_note = (
            "Emergency Warning: The current symptom pattern may indicate a serious condition "
            "that needs urgent medical evaluation."
        )

        if ai_response:
            ai_response = emergency_note + "\n\n" + ai_response
        else:
            ai_response = emergency_note

        precautions = list(precautions or [])
        urgent_items = [
            "seek urgent medical evaluation",
            "do not ignore chest pain or breathing difficulty",
            "go to emergency care if symptoms are severe or worsening"
        ]

        for item in urgent_items:
            if item not in precautions:
                precautions.insert(0, item)

    return {
        "mode": mode,
        "history_used": history_used,
        "prediction_result": prediction_result,
        "predicted_disease": predicted_disease,
        "description": description,
        "precautions": precautions,
        "confidence": confidence,
        "is_low_confidence": is_low_confidence,
        "is_very_low_confidence": is_very_low_confidence,
        "severity_result": severity_result,
        "ai_response": ai_response,
        "emergency_flag": emergency_flag,
    }


def build_response_dict(
    mode,
    result,
    symptoms_data,
    history_available=False,
    used_history_count=0,
    user_id=None,
    message="",
    session_id=None,
    history_symptoms_used=None,
    merged_sequence=None,
):
    return {
        "mode": mode,
        "history_enabled": history_available,
        "history_available": history_available,
        "history_used": result.get("history_used", False),
        "user_id": user_id,
        "session_id": session_id,
        "message": message,
        "used_history_count": used_history_count,
        "current_symptoms": symptoms_data,
        "history_symptoms_used": history_symptoms_used or [],
        "merged_sequence": merged_sequence or symptoms_data,
        "predicted_disease": result["predicted_disease"],
        "confidence": float(result["confidence"]),
        "confidence_percentage": round(float(result["confidence"]) * 100, 2),
        "is_low_confidence": result["is_low_confidence"],
        "is_very_low_confidence": result["is_very_low_confidence"],
        "description": result["description"],
        "precautions": result["precautions"],
        "severity_summary": result["severity_result"],
        "ai_response": result["ai_response"],
        "top_k_predictions": result["prediction_result"]["top_k_predictions"],
        "emergency_flag": result.get("emergency_flag", False),
    }


def run_prediction(payload):
    current_symptoms = normalize_payload_symptoms(payload)

    if not current_symptoms:
        return {
            "mode": "current_session",
            "history_available": False,
            "history_used": False,
            "predicted_disease": None,
            "confidence": 0,
            "confidence_percentage": 0,
            "is_low_confidence": True,
            "is_very_low_confidence": True,
            "description": "",
            "precautions": [],
            "severity_summary": {
                "total": 0,
                "avg": 0,
                "condition": "Mild",
                "weights": {},
                "base_weights": {}
            },
            "ai_response": "No symptoms detected from input.",
            "top_k_predictions": [],
            "current_symptoms": [],
            "history_symptoms_used": [],
            "merged_sequence": [],
            "emergency_flag": False,
        }

    result = build_prediction_output(
        symptoms_data=current_symptoms,
        payload=payload,
        mode="current_session",
        history_used=False
    )

    return build_response_dict(
        mode="current_session",
        result=result,
        symptoms_data=current_symptoms,
        history_available=False,
        used_history_count=0,
        history_symptoms_used=[],
        merged_sequence=current_symptoms,
    )


def run_history_aware_prediction(payload):
    current_symptoms = normalize_payload_symptoms(payload)

    if not current_symptoms:
        return {
            "mode": "history_aware",
            "history_available": False,
            "history_used": False,
            "user_id": int(payload.user_id),
            "message": "No symptoms detected from input.",
            "used_history_count": 0,
            "current_symptoms": [],
            "history_symptoms_used": [],
            "merged_sequence": [],
            "top_k_predictions": [],
            "emergency_flag": False,
        }

    user_id = int(payload.user_id)

    history_exists_before_current = has_previous_history(user_id)

    current_result = build_prediction_output(
        symptoms_data=current_symptoms,
        payload=payload,
        mode="current_session",
        history_used=False
    )

    session_id = create_prediction_session(user_id, payload.user_text)
    save_session_symptoms(session_id, current_symptoms)

    save_session_prediction(
        session_id=session_id,
        mode="current_session",
        predicted_disease=current_result["predicted_disease"],
        confidence=current_result["confidence"],
        ai_response=current_result["ai_response"],
        description=current_result["description"],
        precautions=current_result["precautions"],
        top_k_predictions=current_result["prediction_result"]["top_k_predictions"],
        severity_summary=current_result["severity_result"],
        history_context=None,
    )

    if not history_exists_before_current:
        return build_response_dict(
            mode="history_aware",
            result=current_result,
            symptoms_data=current_symptoms,
            history_available=False,
            used_history_count=0,
            user_id=user_id,
            session_id=session_id,
            message="No previous history found for this user yet. Current session has been saved as the first history record.",
            history_symptoms_used=[],
            merged_sequence=current_symptoms,
        )

    # إذا current prediction واضح نسبيًا أو الحالة خطرة،
    # لا تخلي history يخرب النتيجة الحالية
    if current_result["confidence"] >= 0.35 or current_result.get("emergency_flag", False):
        return build_response_dict(
            mode="history_aware",
            result=current_result,
            symptoms_data=current_symptoms,
            history_available=True,
            used_history_count=0,
            user_id=user_id,
            session_id=session_id,
            message="History was available, but the system kept the current-session prediction because it was already sufficiently clear or medically important.",
            history_symptoms_used=[],
            merged_sequence=current_symptoms,
        )

    history_symptoms = get_recent_history_symptoms(
        user_id=user_id,
        limit_sessions=3,
        max_items=17,
        exclude_session_id=session_id,
    )

    if not history_symptoms:
        return build_response_dict(
            mode="history_aware",
            result=current_result,
            symptoms_data=current_symptoms,
            history_available=False,
            used_history_count=0,
            user_id=user_id,
            session_id=session_id,
            message="No usable previous history found for this user.",
            history_symptoms_used=[],
            merged_sequence=current_symptoms,
        )

    merged_symptoms = merge_history_with_current(
        history_symptoms=history_symptoms,
        current_symptoms=current_symptoms,
        max_len=17
    )

    history_result = build_prediction_output(
        symptoms_data=merged_symptoms,
        payload=payload,
        mode="history_aware",
        history_used=True
    )

    history_result["ai_response"] = append_history_note(
        history_result["ai_response"],
        history_symptoms=history_symptoms,
        merged_symptoms=merged_symptoms
    )

    save_session_prediction(
        session_id=session_id,
        mode="history_aware",
        predicted_disease=history_result["predicted_disease"],
        confidence=history_result["confidence"],
        ai_response=history_result["ai_response"],
        description=history_result["description"],
        precautions=history_result["precautions"],
        top_k_predictions=history_result["prediction_result"]["top_k_predictions"],
        severity_summary=history_result["severity_result"],
        history_context={
            "history_symptoms_used": history_symptoms,
            "merged_sequence": merged_symptoms,
        },
    )

    return build_response_dict(
        mode="history_aware",
        result=history_result,
        symptoms_data=current_symptoms,
        history_available=True,
        used_history_count=len(history_symptoms),
        user_id=user_id,
        session_id=session_id,
        history_symptoms_used=history_symptoms,
        merged_sequence=merged_symptoms,
    )