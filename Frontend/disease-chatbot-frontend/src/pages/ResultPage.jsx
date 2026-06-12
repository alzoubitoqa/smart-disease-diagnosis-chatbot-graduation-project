import { Link } from "react-router-dom"
import AppLayout from "../layout/AppLayout"

function ResultPage() {
  const stored = localStorage.getItem("latestResult")
  const result = stored ? JSON.parse(stored) : null

  const formatText = (text) => {
    if (!text) return "Not available"

    return String(text)
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase())
  }

  if (!result) {
    return (
      <AppLayout>
        <div className="result-page">
          <div className="result-empty-box">
            <h2>No result yet</h2>
            <p>Go to the diagnosis chat and make a prediction first.</p>

            <Link to="/chat" className="btn primary-btn">
              Go to Chat
            </Link>
          </div>
        </div>
      </AppLayout>
    )
  }

  const getSeverityClass = (severity) => {
    if (severity === "Severe") return "severity-badge severity-severe"
    if (severity === "Moderate") return "severity-badge severity-moderate"
    return "severity-badge severity-mild"
  }

  const getUrgency = (score) => {
    if (score >= 7) {
      return {
        label: "High Attention Needed",
        className: "urgency-high",
        note: "The severity score is high. Medical attention is recommended if symptoms continue or worsen."
      }
    }

    if (score >= 4) {
      return {
        label: "Monitor Closely",
        className: "urgency-medium",
        note: "The symptoms should be monitored. Follow up if the condition does not improve."
      }
    }

    return {
      label: "Low Urgency",
      className: "urgency-low",
      note: "The severity level appears low. Continue monitoring symptoms."
    }
  }

  const urgency = getUrgency(result.severityScore || 0)

  return (
    <AppLayout>
      <div className="result-page premium-result-page">
        <section className="result-hero-card">
          <div>
            <span className="result-section-badge">AI Prediction Summary</span>
            <h1>Prediction Result</h1>
            <p>
              This result is generated from your backend prediction model based
              on the symptoms and severity levels entered in the chat.
            </p>
          </div>

          <Link to="/chat" className="btn result-ghost-btn">
            Back to Chat
          </Link>
        </section>

        <section className="result-disease-card">
          <div className="result-disease-top">
            <div>
              <span className="result-section-badge">Predicted Disease</span>
              <h2>{formatText(result.disease)}</h2>
            </div>

            <div className={`result-urgency-panel ${urgency.className}`}>
              <span>Urgency Level</span>
              <strong>{urgency.label}</strong>
            </div>
          </div>

          <div className="result-meta-row premium-result-meta">
            <span className="score-chip">Severity Score: {result.severityScore || 0}</span>
            <span className={urgency.className}>{urgency.label}</span>
          </div>

          <div className="confidence-section premium-confidence-section">
            <div className="confidence-header">
              <h3>Confidence Score</h3>
              <strong>{result.confidence || 0}%</strong>
            </div>

            <div className="confidence-bar">
              <div
                className="confidence-bar-fill"
                style={{ width: `${Math.min(Number(result.confidence || 0), 100)}%` }}
              />
            </div>
          </div>

          <div className="result-guidance-box">
            <strong>Guidance</strong>
            <p>{urgency.note}</p>
          </div>
        </section>

        <section className="result-grid premium-result-grid">
          <div className="result-box premium-result-box">
            <div className="result-box-header">
              <span>🩺</span>
              <h3>Symptoms</h3>
            </div>

            <div className="result-symptom-list">
              {(result.symptoms || []).map((s, i) => (
                <div key={i} className="result-symptom-tag">
                  <span>{formatText(s.name)}</span>
                  <strong className={getSeverityClass(s.severity)}>
                    {s.severity}
                  </strong>
                </div>
              ))}
            </div>
          </div>

          <div className="result-box premium-result-box">
            <div className="result-box-header">
              <span>📄</span>
              <h3>Description</h3>
            </div>

            <p>{result.description || "No description available."}</p>
          </div>

          <div className="result-box premium-result-box">
            <div className="result-box-header">
              <span>✅</span>
              <h3>Precautions</h3>
            </div>

            {(result.precautions || []).length > 0 ? (
              <ul>
                {result.precautions.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            ) : (
              <p>No precautions available.</p>
            )}
          </div>
        </section>

        <section className="result-safety-note">
          <div>
            <span className="result-section-badge">Important Note</span>
            <h3>Preliminary support only</h3>
            <p>
              This system is designed to support early understanding of symptoms.
              It does not replace professional medical diagnosis or emergency care.
            </p>
          </div>

          <div className="result-actions">
            <Link to="/chat" className="btn result-main-btn">
              New Diagnosis
            </Link>

            <Link to="/history" className="btn result-ghost-btn">
              View History
            </Link>
          </div>
        </section>
      </div>
    </AppLayout>
  )
}

export default ResultPage