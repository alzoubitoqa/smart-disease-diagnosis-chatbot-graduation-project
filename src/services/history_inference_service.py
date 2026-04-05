from src.ml.history_inference import build_history_decision


def apply_history_logic(current_result: dict, history_context: dict):
    return build_history_decision(current_result, history_context)