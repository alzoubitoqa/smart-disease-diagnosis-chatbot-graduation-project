export const mockUser = {
  name: "Toqa Al-Zoubi",
  email: "toqa@example.com",
  language: "English",
}

export const mockSessions = [
  {
    id: 1,
    symptoms: ["Fever", "Cough", "Fatigue"],
    prediction: "Influenza",
    confidence: 0.87,
    date: "12 Mar 2026"
  },
  {
    id: 2,
    symptoms: ["Headache", "Nausea", "Light sensitivity"],
    prediction: "Migraine",
    confidence: 0.74,
    date: "10 Mar 2026"
  },
  {
    id: 3,
    symptoms: ["Sore throat", "Mild fever", "Body aches"],
    prediction: "Common Cold",
    confidence: 0.68,
    date: "07 Mar 2026"
  }
]

export const mockStats = {
  totalSessions: 12,
  symptomsTracked: 28,
  latestPrediction: "Influenza",
  healthStatus: "Needs Follow-up"
}