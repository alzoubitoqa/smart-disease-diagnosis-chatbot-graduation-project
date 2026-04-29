import { useEffect, useMemo, useState } from "react"
import { Link } from "react-router-dom"
import AppLayout from "../layout/AppLayout"
import PageHeader from "../components/common/PageHeader"
import { useAuth } from "../context/AuthContext"
import { getHistoryRequest } from "../services/historyService"

function HistoryPage() {
  const { user } = useAuth()

  const [search, setSearch] = useState("")
  const [sessions, setSessions] = useState([])
  const [predictionFilter, setPredictionFilter] = useState("All")
  const [severityFilter, setSeverityFilter] = useState("All")
  const [sortOrder, setSortOrder] = useState("Newest")
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadHistory = async () => {
      if (!user?.user_id) {
        setSessions([])
        setLoading(false)
        return
      }

      try {
        const data = await getHistoryRequest(user.user_id)
        setSessions(Array.isArray(data) ? data : [])
      } catch (error) {
        console.error("Failed to load history:", error)
        setSessions([])
      } finally {
        setLoading(false)
      }
    }

    loadHistory()
  }, [user])

  const normalizeSymptoms = (symptoms) => {
    if (Array.isArray(symptoms)) {
      return symptoms.map((item) => {
        if (typeof item === "string") {
          return { name: item, severity: "Moderate", severityValue: 2 }
        }

        return {
          name: item?.name || "Unknown symptom",
          severity: item?.severity || "Moderate",
          severityValue: item?.severityValue ?? 2
        }
      })
    }

    if (typeof symptoms === "string") {
      return symptoms
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => ({
          name: item,
          severity: "Moderate",
          severityValue: 2
        }))
    }

    return []
  }

  const formatSymptomName = (name) => {
    if (!name) return "Unknown"

    return String(name)
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase())
  }

  const getPredictionData = (session) => {
    return session.historyAwarePrediction || session.currentPrediction || null
  }

  const getPredictionName = (session) => {
    const predictionData = getPredictionData(session)
    return predictionData?.predicted_disease || session.prediction || "No prediction"
  }

  const getConfidenceValue = (session) => {
    const predictionData = getPredictionData(session)

    if (predictionData?.confidence !== undefined && predictionData?.confidence !== null) {
      return Math.round(Number(predictionData.confidence) * 100)
    }

    return Number(session.confidence || 0)
  }

  const getSeverityCondition = (session) => {
    const predictionData = getPredictionData(session)
    return (
      predictionData?.severity_condition ||
      predictionData?.severityCondition ||
      session?.severityCondition ||
      null
    )
  }

  const getHistoryStatus = (session) => {
    const historyPrediction = session.historyAwarePrediction

    if (!historyPrediction) {
      return null
    }

    const historyApplied =
      Boolean(historyPrediction.history_applied) ||
      Boolean(historyPrediction.history_changed_top1) ||
      Boolean(historyPrediction.history_changed_ranking)

    if (historyApplied) {
      return {
        label: "History Applied",
        bg: "#d1fae5",
        color: "#065f46",
        title: "Previous sessions influenced this prediction"
      }
    }

    return {
      label: "History Reviewed",
      bg: "#e5e7eb",
      color: "#374151",
      title: "Previous sessions were checked but did not change the result"
    }
  }

  const predictionOptions = useMemo(() => {
    const uniquePredictions = [
      ...new Set(
        sessions
          .map((s) => getPredictionName(s))
          .filter((value) => value && value !== "No prediction")
      ),
    ]

    return ["All", ...uniquePredictions]
  }, [sessions])

  const filteredHistory = useMemo(() => {
    let result = [...sessions]

    if (search.trim()) {
      const query = search.toLowerCase()

      result = result.filter((item) => {
        const normalizedSymptoms = normalizeSymptoms(item.symptoms)

        const symptomNames = normalizedSymptoms
          .map((symptom) => symptom.name)
          .join(" ")
          .toLowerCase()

        const severityLevels = normalizedSymptoms
          .map((symptom) => symptom.severity)
          .join(" ")
          .toLowerCase()

        const predictedDisease = getPredictionName(item).toLowerCase()

        return (
          symptomNames.includes(query) ||
          severityLevels.includes(query) ||
          predictedDisease.includes(query)
        )
      })
    }

    if (predictionFilter !== "All") {
      result = result.filter((item) => getPredictionName(item) === predictionFilter)
    }

    if (severityFilter !== "All") {
      result = result.filter((item) => {
        const normalizedSymptoms = normalizeSymptoms(item.symptoms)
        return normalizedSymptoms.some((symptom) => symptom.severity === severityFilter)
      })
    }

    if (sortOrder === "Newest") {
      result.sort((a, b) => Number(b.id) - Number(a.id))
    } else {
      result.sort((a, b) => Number(a.id) - Number(b.id))
    }

    return result
  }, [sessions, search, predictionFilter, severityFilter, sortOrder])

  const getSeverityClass = (severity) => {
    if (severity === "Severe") return "severity-badge severity-severe"
    if (severity === "Moderate") return "severity-badge severity-moderate"
    return "severity-badge severity-mild"
  }

  return (
    <AppLayout>
      <div className="history-page">
        <PageHeader
          title="Symptom History"
          description="Review your diagnosis sessions, filter them, and inspect symptom severity patterns over time."
        />

        <div className="history-filters">
          <input
            className="history-search"
            type="text"
            placeholder="Search by symptom, prediction, or severity..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />

          <select
            className="history-select"
            value={predictionFilter}
            onChange={(e) => setPredictionFilter(e.target.value)}
          >
            {predictionOptions.map((option) => (
              <option key={option} value={option}>
                {formatSymptomName(option)}
              </option>
            ))}
          </select>

          <select
            className="history-select"
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
          >
            <option value="All">All Severity Levels</option>
            <option value="Mild">Mild</option>
            <option value="Moderate">Moderate</option>
            <option value="Severe">Severe</option>
          </select>

          <select
            className="history-select"
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
          >
            <option value="Newest">Newest First</option>
            <option value="Oldest">Oldest First</option>
          </select>
        </div>

        {loading ? (
          <div className="history-empty">
            <h3>Loading history...</h3>
          </div>
        ) : filteredHistory.length > 0 ? (
          <div className="history-list">
            {filteredHistory.map((session) => {
              const normalizedSymptoms = normalizeSymptoms(session.symptoms)
              const predictionName = getPredictionName(session)
              const confidenceValue = getConfidenceValue(session)
              const severityCondition = getSeverityCondition(session)
              const historyStatus = getHistoryStatus(session)

              return (
                <Link
                  to={`/session/${session.id}`}
                  key={session.id}
                  className="history-link-card"
                >
                  <div className="history-card advanced-history-card">
                    <div className="history-header">
                      <div>
                        <h3>Session #{session.id}</h3>
                        <p className="history-subtitle">{session.date}</p>
                      </div>

                      <div
                        className="history-right-meta"
                        style={{
                          display: "flex",
                          gap: "8px",
                          flexWrap: "wrap",
                          alignItems: "center"
                        }}
                      >
                        <span className="prediction-chip">
                          {formatSymptomName(predictionName)}
                        </span>

                        <span className="score-chip">
                          Score: {session.severityScore ?? 0}
                        </span>

                        {historyStatus && (
                          <span
                            title={historyStatus.title}
                            style={{
                              padding: "6px 12px",
                              borderRadius: "999px",
                              background: historyStatus.bg,
                              color: historyStatus.color,
                              fontSize: "12px",
                              fontWeight: 700
                            }}
                          >
                            {historyStatus.label}
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="history-symptoms-block">
                      <h4>Symptoms</h4>
                      <div className="history-symptom-tags">
                        {normalizedSymptoms.map((item, index) => (
                          <div key={index} className="history-symptom-tag">
                            <span>{formatSymptomName(item.name)}</span>
                            <strong className={getSeverityClass(item.severity)}>
                              {item.severity}
                            </strong>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div
                      className="history-footer-row"
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        gap: "12px",
                        flexWrap: "wrap"
                      }}
                    >
                      <span className="history-footer-text">
                        Confidence: {confidenceValue}%
                        {severityCondition ? ` • ${severityCondition}` : ""}
                      </span>

                      <span className="history-footer-link">
                        View Details
                      </span>
                    </div>
                  </div>
                </Link>
              )
            })}
          </div>
        ) : (
          <div className="history-empty">
            <h3>No history found</h3>
            <p>
              Start a new diagnosis session and your personal history will appear here.
            </p>
          </div>
        )}
      </div>
    </AppLayout>
  )
}

export default HistoryPage