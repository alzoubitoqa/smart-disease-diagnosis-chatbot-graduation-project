import { useEffect, useMemo, useState } from "react"
import { useParams } from "react-router-dom"
import AppLayout from "../layout/AppLayout"
import { useAuth } from "../context/AuthContext"
import { getHistoryRequest } from "../services/historyService"

function SessionDetailsPage() {
  const { id } = useParams()
  const { user } = useAuth()

  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadHistory = async () => {
      if (!user?.user_id) {
        setHistory([])
        setLoading(false)
        return
      }

      try {
        const data = await getHistoryRequest(user.user_id)
        setHistory(Array.isArray(data) ? data : [])
      } catch (error) {
        console.error("Failed to load session details:", error)
        setHistory([])
      } finally {
        setLoading(false)
      }
    }

    loadHistory()
  }, [user])

  const session = useMemo(() => {
    return history.find((s) => String(s.id) === String(id))
  }, [history, id])

  const formatSymptomName = (name) => {
    if (!name || name === "uncertain_case") {
      return "Uncertain Case"
    }

    return String(name || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase())
  }

  const getSeverityClass = (severity) => {
    if (severity === "Severe") return "severity-badge severity-severe"
    if (severity === "Moderate") return "severity-badge severity-moderate"
    return "severity-badge severity-mild"
  }

  const normalizeSymptoms = (symptoms) => {
    if (!Array.isArray(symptoms)) return []

    return symptoms.map((item) => ({
      name: item?.name || "Unknown symptom",
      severity: item?.severity || "Moderate",
      severityValue: item?.severityValue ?? 2
    }))
  }

  const renderPredictionCard = (title, predictionData, mode = "current") => {
    if (!predictionData) return null

    const confidencePercentage = Math.round(Number(predictionData.confidence || 0) * 100)
    const isLowConfidence = confidencePercentage < 30

    let historyContext = {}
    if (mode === "history") {
      try {
        historyContext = predictionData.history_context_json
          ? JSON.parse(predictionData.history_context_json)
          : {}
      } catch {
        historyContext = {}
      }
    }

    const repeatedSymptoms = historyContext.repeated_symptoms || []
    const similarSessionsCount = historyContext.similar_sessions_count || 0
    const recentDiseases = historyContext.last_predicted_diseases || []

    return (
      <div className="details-card" style={{ marginTop: "20px" }}>
        <h3>{title}</h3>

        <div style={{ display: "grid", gap: "12px" }}>
          <div>
            {isLowConfidence ? (
              <>
                <strong>Possible Condition:</strong>{" "}
                {formatSymptomName(predictionData.predicted_disease)}
                <span style={{ color: "orange", marginLeft: "8px", fontWeight: "600" }}>
                  (Low Confidence)
                </span>
              </>
            ) : (
              <>
                <strong>Disease:</strong>{" "}
                {formatSymptomName(predictionData.predicted_disease)}
              </>
            )}
          </div>

          <div>
            <strong>Confidence:</strong> {confidencePercentage}%
          </div>

          {isLowConfidence && (
            <div
              style={{
                padding: "10px 12px",
                borderRadius: "10px",
                background: "#fff8e6",
                color: "#8a5a00",
                fontWeight: "500"
              }}
            >
              This prediction has low confidence and should not be treated as a final diagnosis.
            </div>
          )}

          {mode === "history" && (
            <div
              style={{
                padding: "10px 12px",
                borderRadius: "10px",
                background: "#eef6ff",
                color: "#174a7c"
              }}
            >
              <strong>History Insight:</strong>
              <div style={{ marginTop: "6px" }}>
                <div>Similar previous sessions: {similarSessionsCount}</div>

                {repeatedSymptoms.length > 0 && (
                  <div>
                    Repeated symptoms:{" "}
                    {repeatedSymptoms.map((item) => formatSymptomName(item)).join(", ")}
                  </div>
                )}

                {recentDiseases.length > 0 && (
                  <div>
                    Recent related conditions:{" "}
                    {recentDiseases.map((item) => formatSymptomName(item)).join(", ")}
                  </div>
                )}
              </div>
            </div>
          )}

          <div>
            <strong>Description:</strong>
            <p style={{ marginTop: "6px" }}>
              {predictionData.description || "No description available."}
            </p>
          </div>

          <div>
            <strong>Precautions:</strong>
            <ul style={{ marginTop: "6px", paddingLeft: "18px" }}>
              {(() => {
                try {
                  const precautions = predictionData.precautions_json
                    ? JSON.parse(predictionData.precautions_json)
                    : []
                  return precautions.length > 0 ? (
                    precautions.map((item, index) => <li key={index}>{item}</li>)
                  ) : (
                    <li>No precautions available.</li>
                  )
                } catch {
                  return <li>No precautions available.</li>
                }
              })()}
            </ul>
          </div>

          <div>
            <strong>Severity Summary:</strong>
            <p style={{ marginTop: "6px" }}>
              Total: {predictionData.severity_total ?? 0} | Avg:{" "}
              {predictionData.severity_avg ?? 0} | Condition:{" "}
              {predictionData.severity_condition || "N/A"}
            </p>
          </div>

          <div>
            <strong>AI Response:</strong>
            <p style={{ marginTop: "6px", whiteSpace: "pre-wrap" }}>
              {predictionData.ai_response || "No AI response available."}
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <AppLayout>
        <div className="details-page">
          <h2>Loading session details...</h2>
        </div>
      </AppLayout>
    )
  }

  if (!session) {
    return (
      <AppLayout>
        <div className="details-page">
          <h2>Session not found</h2>
        </div>
      </AppLayout>
    )
  }

  const normalizedSymptoms = normalizeSymptoms(session.symptoms)
  const currentPrediction = session.currentPrediction
  const historyAwarePrediction = session.historyAwarePrediction

  return (
    <AppLayout>
      <div className="details-page">
        <h1>Session Details</h1>

        <div className="details-card">
          <h3>Session Overview</h3>
          <p><strong>Session ID:</strong> {session.id}</p>
          <p><strong>Date:</strong> {session.date || session.timestamp || "N/A"}</p>
          <p><strong>Severity Score:</strong> {session.severityScore ?? 0}</p>
          <p>
            <strong>Main Display Prediction:</strong>{" "}
            {formatSymptomName(session.prediction)}
          </p>
        </div>

        <div className="details-card">
          <h3>Symptoms</h3>

          <div className="symptoms-list">
            {normalizedSymptoms.length > 0 ? (
              normalizedSymptoms.map((sym, index) => (
                <div key={index} className="symptom-tag">
                  <span>{formatSymptomName(sym.name)}</span>
                  <strong className={getSeverityClass(sym.severity)}>
                    {sym.severity}
                  </strong>
                </div>
              ))
            ) : (
              <p>No symptoms available.</p>
            )}
          </div>
        </div>

        {renderPredictionCard("Current Session Prediction", currentPrediction, "current")}

        {historyAwarePrediction &&
          renderPredictionCard(
            "History-Aware Prediction",
            historyAwarePrediction,
            "history"
          )}
      </div>
    </AppLayout>
  )
}

export default SessionDetailsPage