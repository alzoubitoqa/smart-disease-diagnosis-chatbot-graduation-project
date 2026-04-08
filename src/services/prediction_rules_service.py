from copy import deepcopy


RESPIRATORY_SYMPTOMS = {
    "continuous_sneezing",
    "cough",
    "breathlessness",
    "phlegm",
    "mucoid_sputum",
    "rusty_sputum",
    "throat_irritation",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
}

INFECTIOUS_SYMPTOMS = {
    "high_fever",
    "mild_fever",
    "chills",
    "fatigue",
    "swelled_lymph_nodes",
    "malaise",
    "dehydration",
    "sweating",
    "headache",
}

DIGESTIVE_SYMPTOMS = {
    "abdominal_pain",
    "stomach_pain",
    "vomiting",
    "nausea",
    "diarrhoea",
    "distention_of_abdomen",
    "belly_pain",
    "acidity",
    "indigestion",
    "loss_of_appetite",
    "passage_of_gases",
    "stomach_bleeding",
}

SKIN_SYMPTOMS = {
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "dischromic_patches",
    "skin_peeling",
    "blister",
    "red_spots_over_body",
    "silver_like_dusting",
    "pus_filled_pimples",
    "blackheads",
}

NEURO_SYMPTOMS = {
    "headache",
    "dizziness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "blurred_and_distorted_vision",
    "slurred_speech",
    "weakness_of_one_body_side",
    "stiff_neck",
}

MALARIA_SUPPORT = {
    "high_fever",
    "chills",
    "sweating",
    "headache",
    "vomiting",
    "nausea",
    "muscle_pain",
    "malaise",
}

ASTHMA_SUPPORT = {
    "cough",
    "breathlessness",
    "chest_pain",
    "fatigue",
}

PNEUMONIA_SUPPORT = {
    "cough",
    "high_fever",
    "breathlessness",
    "chest_pain",
    "fatigue",
    "phlegm",
    "rusty_sputum",
}

FUNGAL_SUPPORT = {
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "dischromic_patches",
    "skin_peeling",
}

HEPATITIS_SUPPORT = {
    "abdominal_pain",
    "fatigue",
    "high_fever",
    "joint_pain",
    "yellowish_skin",
    "yellowing_of_eyes",
    "dark_urine",
    "loss_of_appetite",
    "nausea",
    "vomiting",
}

DENGUE_SUPPORT = {
    "high_fever",
    "chills",
    "fatigue",
    "headache",
    "nausea",
    "vomiting",
    "joint_pain",
    "muscle_pain",
    "skin_rash",
}

COMMON_COLD_SUPPORT = {
    "continuous_sneezing",
    "cough",
    "runny_nose",
    "congestion",
    "headache",
    "mild_fever",
    "chills",
}

DISEASE_RULES = {
    "malaria": {
        "support": MALARIA_SUPPORT,
        "penalty_if_has": RESPIRATORY_SYMPTOMS,
        "boost": 0.08,
        "penalty": 0.08,
    },
    "bronchial_asthma": {
        "support": ASTHMA_SUPPORT,
        "penalty_if_has": SKIN_SYMPTOMS | DIGESTIVE_SYMPTOMS,
        "boost": 0.08,
        "penalty": 0.05,
    },
    "pneumonia": {
        "support": PNEUMONIA_SUPPORT,
        "penalty_if_has": SKIN_SYMPTOMS,
        "boost": 0.08,
        "penalty": 0.05,
    },
    "fungal_infection": {
        "support": FUNGAL_SUPPORT,
        "penalty_if_has": RESPIRATORY_SYMPTOMS,
        "boost": 0.08,
        "penalty": 0.08,
    },
    "hepatitis_e": {
        "support": HEPATITIS_SUPPORT,
        "penalty_if_has": RESPIRATORY_SYMPTOMS | SKIN_SYMPTOMS,
        "boost": 0.08,
        "penalty": 0.04,
    },
    "jaundice": {
        "support": HEPATITIS_SUPPORT,
        "penalty_if_has": RESPIRATORY_SYMPTOMS | SKIN_SYMPTOMS,
        "boost": 0.06,
        "penalty": 0.04,
    },
    "hepatitis_d": {
        "support": HEPATITIS_SUPPORT,
        "penalty_if_has": RESPIRATORY_SYMPTOMS | SKIN_SYMPTOMS,
        "boost": 0.06,
        "penalty": 0.04,
    },
    "dengue": {
        "support": DENGUE_SUPPORT,
        "penalty_if_has": RESPIRATORY_SYMPTOMS,
        "boost": 0.06,
        "penalty": 0.06,
    },
    "common_cold": {
        "support": COMMON_COLD_SUPPORT,
        "penalty_if_has": DIGESTIVE_SYMPTOMS | SKIN_SYMPTOMS,
        "boost": 0.07,
        "penalty": 0.04,
    },
}


def _normalize_symptom_name(symptom: str) -> str:
    return str(symptom or "").strip().lower().replace(" ", "_")


def _extract_symptom_names(symptoms_with_severity):
    return [_normalize_symptom_name(item["symptom"]) for item in symptoms_with_severity]


def _count_matches(symptom_names, support_set):
    symptom_set = set(symptom_names)
    return len(symptom_set.intersection(support_set))


def _dominant_category(symptom_names):
    counts = {
        "respiratory": _count_matches(symptom_names, RESPIRATORY_SYMPTOMS),
        "infectious": _count_matches(symptom_names, INFECTIOUS_SYMPTOMS),
        "digestive": _count_matches(symptom_names, DIGESTIVE_SYMPTOMS),
        "skin": _count_matches(symptom_names, SKIN_SYMPTOMS),
        "neuro": _count_matches(symptom_names, NEURO_SYMPTOMS),
    }

    dominant = max(counts, key=counts.get)
    return dominant, counts


def _rule_adjust_score(disease_name, symptom_names):
    disease_key = _normalize_symptom_name(disease_name)
    rule = DISEASE_RULES.get(disease_key)

    if not rule:
        return 0.0, 0, 0

    support_matches = _count_matches(symptom_names, rule["support"])
    penalty_matches = _count_matches(symptom_names, rule["penalty_if_has"])

    score_adjustment = 0.0

    # special malaria fix
    if disease_key == "malaria":
        respiratory_count = _count_matches(symptom_names, RESPIRATORY_SYMPTOMS)

        if respiratory_count >= 1:
            score_adjustment -= 0.10
        elif support_matches >= 2:
            score_adjustment += rule["boost"]
    else:
        if support_matches >= 2:
            score_adjustment += rule["boost"]

    if penalty_matches >= 2 and support_matches < 2:
        score_adjustment -= rule["penalty"]

    # dengue helper
    if disease_key == "dengue":
        if "skin_rash" in symptom_names or "headache" in symptom_names:
            score_adjustment += 0.05

    return score_adjustment, support_matches, penalty_matches


def apply_prediction_rules(symptoms_with_severity, prediction_result):
    """
    لا نغير المودل نفسه.
    فقط نعيد ترتيب top-k بشكل أذكى اعتمادًا على نمط الأعراض.
    """
    if not prediction_result or not prediction_result.get("top_k_predictions"):
        return prediction_result

    symptom_names = _extract_symptom_names(symptoms_with_severity)
    dominant_category, category_counts = _dominant_category(symptom_names)

    explanations = []
    adjusted_predictions = []

    for item in deepcopy(prediction_result["top_k_predictions"]):
        disease = item["disease"]
        base_confidence = float(item.get("confidence", 0.0))
        disease_key = _normalize_symptom_name(disease)

        rule_adjustment, support_matches, penalty_matches = _rule_adjust_score(
            disease, symptom_names
        )
        adjusted_score = base_confidence + rule_adjustment

        notes = []

        if disease_key == "malaria" and _count_matches(symptom_names, RESPIRATORY_SYMPTOMS) >= 1:
            notes.append("penalized because malaria does not fit respiratory symptoms well")

        if dominant_category == "respiratory" and category_counts["respiratory"] >= 2:
            if disease_key in {"bronchial_asthma", "pneumonia", "common_cold"}:
                adjusted_score += 0.04
                notes.append("boosted due to respiratory symptom dominance")
            if disease_key in {"malaria", "dengue"}:
                adjusted_score -= 0.05
                notes.append("penalized because respiratory symptoms dominate")

        if dominant_category == "skin" and category_counts["skin"] >= 2:
            if disease_key in {"fungal_infection", "acne", "impetigo", "psoriasis"}:
                adjusted_score += 0.04
                notes.append("boosted due to skin symptom dominance")
            if disease_key in {"pneumonia", "bronchial_asthma"}:
                adjusted_score -= 0.04
                notes.append("penalized because skin symptoms dominate")

        if dominant_category == "digestive" and category_counts["digestive"] >= 2:
            if disease_key in {"gastroenteritis", "jaundice", "hepatitis_e", "hepatitis_d"}:
                adjusted_score += 0.04
                notes.append("boosted due to digestive symptom dominance")

        item["base_confidence"] = base_confidence
        item["rule_adjustment"] = round(rule_adjustment, 4)
        item["adjusted_score"] = round(adjusted_score, 6)

        explanation = {
            "disease": disease,
            "base_confidence": round(base_confidence, 4),
            "support_matches": support_matches,
            "penalty_matches": penalty_matches,
            "rule_adjustment": round(rule_adjustment, 4),
            "final_score": round(adjusted_score, 4),
            "notes": notes,
        }
        explanations.append(explanation)

        adjusted_predictions.append(item)

    adjusted_predictions.sort(key=lambda x: x["adjusted_score"], reverse=True)

    corrected_result = deepcopy(prediction_result)
    corrected_result["top_k_predictions"] = adjusted_predictions
    corrected_result["top_prediction"] = adjusted_predictions[0]

    corrected_result["rules_summary"] = {
        "dominant_category": dominant_category,
        "category_counts": category_counts,
        "explanations": explanations,
    }

    return corrected_result