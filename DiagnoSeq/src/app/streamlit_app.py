import streamlit as st
from src.inference.predict import DiseasePredictor


st.set_page_config(page_title="DiagnoSeq", layout="centered")

st.title("DiagnoSeq")
st.subheader("AI-Powered Disease Prediction from Symptoms")

model_name = st.selectbox(
    "Choose model",
    options=["deep_lstm", "vanilla_lstm", "rnn"],
    index=0
)

symptom_text = st.text_area(
    "Enter symptoms separated by commas",
    placeholder="fever, headache, nausea"
)

if st.button("Predict Disease"):
    symptoms = [s.strip() for s in symptom_text.split(",") if s.strip()]

    if not symptoms:
        st.warning("Please enter at least one symptom.")
    else:
        predictor = DiseasePredictor(model_name=model_name)
        results = predictor.predict(symptoms, top_k=3)

        st.success("Prediction completed.")

        for i, item in enumerate(results, start=1):
            st.markdown(f"### {i}. {item['disease']}")
            st.write(f"**Confidence:** {item['probability']:.4f}")
            st.write(f"**Description:** {item['description']}")

            if item["precautions"]:
                st.write("**Precautions:**")
                for p in item["precautions"]:
                    st.write(f"- {p}")

            st.markdown("---")