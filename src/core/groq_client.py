from groq import Groq
from src.core.config import GROQ_API_KEY

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please set it in the .env file.")

client = Groq(api_key=GROQ_API_KEY)


def generate_medical_response(user_input: str, prediction_result: dict, is_low_confidence: bool = False) -> str:
    top_prediction = prediction_result["top_prediction"]
    disease = top_prediction["disease"]
    confidence = round(top_prediction["confidence"] * 100, 2)
    description = top_prediction.get("description", "")
    precautions = top_prediction.get("precautions", [])

    if isinstance(precautions, dict):
        precautions_text = "\n".join([f"- {v}" for v in precautions.values() if v])
    else:
        precautions_text = "\n".join([f"- {v}" for v in precautions if v])

    top_3_text = ""
    for i, item in enumerate(prediction_result.get("top_k_predictions", [])[:3], start=1):
        top_3_text += f"{i}. {item['disease']} ({round(item['confidence'] * 100, 2)}%)\n"

    confidence_note = (
        "The model confidence is low. Be cautious and do not present the result as a likely confirmed condition."
        if is_low_confidence
        else "The model confidence is acceptable for a preliminary AI-based prediction, but this is still not a final diagnosis."
    )

    prompt = f"""
You are a careful medical assistant chatbot.

Rules:
- Do not claim certainty.
- Do not say this is a final medical diagnosis.
- Keep the answer clear, calm, and helpful.
- If confidence is low, clearly say the prediction is uncertain.
- Mention when the user should seek medical care urgently.
- Use the top predictions to reflect uncertainty when needed.
- No long explanations.

User symptoms:
{user_input}

Top model prediction:
Disease: {disease}
Confidence: {confidence}%
Description: {description}

Precautions:
{precautions_text}

Top 3 predictions:
{top_3_text}

Confidence guidance:
{confidence_note}

Write the response in this structure:
1. Likely condition or uncertain result
2. Why this may be the prediction
3. Recommended precautions
4. Safety note
"""

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You explain model-based medical predictions carefully and clearly."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
    )

    return chat_completion.choices[0].message.content