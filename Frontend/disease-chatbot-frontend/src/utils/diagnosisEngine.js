export function analyzeSymptoms(text) {
  const input = text.toLowerCase()

  if (
    input.includes("fever") ||
    input.includes("cough") ||
    input.includes("sore throat") ||
    input.includes("fatigue")
  ) {
    return {
      disease: "Influenza",
      confidence: 87,
      description:
        "Influenza is a viral respiratory illness commonly associated with fever, fatigue, sore throat, headache, and muscle aches.",
      precautions: [
        "Get enough rest",
        "Drink plenty of fluids",
        "Monitor fever regularly",
        "Seek medical help if symptoms worsen"
      ],
      reply:
        "Your symptoms may be related to a respiratory infection such as influenza. Please monitor the severity of fever, fatigue, and throat discomfort."
    }
  }

  if (
    input.includes("headache") ||
    input.includes("nausea") ||
    input.includes("light")
  ) {
    return {
      disease: "Migraine",
      confidence: 74,
      description:
        "Migraine is a neurological condition that may cause intense headache, nausea, and sensitivity to light or sound.",
      precautions: [
        "Rest in a quiet dark room",
        "Stay hydrated",
        "Avoid strong light and noise",
        "Seek medical advice if attacks become frequent"
      ],
      reply:
        "These symptoms may align with migraine. Please pay attention to headache severity and sensitivity to light."
    }
  }

  if (
    input.includes("runny nose") ||
    input.includes("cold") ||
    input.includes("mild fever")
  ) {
    return {
      disease: "Common Cold",
      confidence: 68,
      description:
        "The common cold is a mild viral infection affecting the upper respiratory tract and usually includes congestion, sore throat, and mild fatigue.",
      precautions: [
        "Rest well",
        "Drink warm fluids",
        "Monitor symptoms for worsening",
        "Use supportive care as needed"
      ],
      reply:
        "This looks closer to a mild upper respiratory issue such as the common cold."
    }
  }

  return {
    disease: "General Medical Review Recommended",
    confidence: 52,
    description:
      "The current symptom pattern is not specific enough for a stronger preliminary match in this frontend simulation.",
    precautions: [
      "Monitor symptoms closely",
      "Record duration and severity",
      "Seek medical advice for persistent symptoms",
      "Add more symptom details for better assessment"
    ],
    reply:
      "I need a bit more detail about your symptoms, their duration, and severity to provide a clearer preliminary assessment."
  }
}