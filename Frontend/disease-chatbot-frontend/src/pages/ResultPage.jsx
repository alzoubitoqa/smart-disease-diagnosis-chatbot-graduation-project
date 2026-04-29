import { Link } from "react-router-dom"
import AppLayout from "../layout/AppLayout"
import PageHeader from "../components/common/PageHeader"

function ResultPage() {
  const stored = localStorage.getItem("latestResult")
  const result = stored ? JSON.parse(stored) : null

  if (!result) {
    return (
      <AppLayout>
        <h2 style={{ padding: "20px" }}>
          No result yet. Go to chat and make a prediction.
        </h2>
      </AppLayout>
    )
  }

  const getSeverityClass = (severity) => {
    if (severity === "Severe") return "severity-badge severity-severe"
    if (severity === "Moderate") return "severity-badge severity-moderate"
    return "severity-badge severity-mild"
  }

  const getUrgency = (score) => {
    if (score >= 7) return { label: "High Attention Needed", className: "urgency-high" }
    if (score >= 4) return { label: "Monitor Closely", className: "urgency-medium" }
    return { label: "Low Urgency", className: "urgency-low" }
  }

  const urgency = getUrgency(result.severityScore || 0)

  return (
    <AppLayout>
      <div className="result-page">

        <PageHeader
          title="Prediction Result"
          description="This result is generated from your real backend model."
        />

        <div className="result-card">

          <div className="result-topbar">
            <div>
              <h2>Predicted Disease</h2>
              <p className="result-main">{result.disease}</p>
            </div>

            <Link to="/chat" className="btn secondary-btn">
              Back to Chat
            </Link>
          </div>

          <div className="result-meta-row">
            <span className="score-chip">Severity Score: {result.severityScore}</span>
            <span className={urgency.className}>{urgency.label}</span>
          </div>

          <div className="confidence-section">
            <div className="confidence-header">
              <h3>Confidence</h3>
              <strong>{result.confidence}%</strong>
            </div>

            <div className="confidence-bar">
              <div
                className="confidence-bar-fill"
                style={{ width: `${result.confidence}%` }}
              />
            </div>
          </div>

          <div className="result-grid">

            <div className="result-box">
              <h3>Symptoms</h3>
              <div className="result-symptom-list">
                {result.symptoms.map((s, i) => (
                  <div key={i} className="result-symptom-tag">
                    <span>{s.name}</span>
                    <strong className={getSeverityClass(s.severity)}>
                      {s.severity}
                    </strong>
                  </div>
                ))}
              </div>
            </div>

            <div className="result-box">
              <h3>Description</h3>
              <p>{result.description}</p>
            </div>

            <div className="result-box">
              <h3>Precautions</h3>
              <ul>
                {result.precautions.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            </div>

          </div>

          <div className="result-actions">
            <Link to="/chat" className="btn primary-btn">
              New Diagnosis
            </Link>
            <Link to="/history" className="btn secondary-btn">
              View History
            </Link>
          </div>

        </div>
      </div>
    </AppLayout>
  )
}

export default ResultPage