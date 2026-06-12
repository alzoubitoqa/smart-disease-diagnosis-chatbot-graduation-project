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
        className: "history-status-applied",
        title: "Previous sessions influenced this prediction"
      }
    }

    return {
      label: "History Reviewed",
      className: "history-status-reviewed",
      title: "Previous sessions were checked but did not change the result"
    }
  }

  const predictionOptions = useMemo(() => {
    const uniquePredictions = [
      ...new Set(
        sessions
          .map((s) => getPredictionName(s))
          .filter((value) => value && value !== "No prediction")
      )
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

  const totalSessions = sessions.length
  const visibleSessions = filteredHistory.length
  const severeSessions = sessions.filter((session) =>
    normalizeSymptoms(session.symptoms).some((symptom) => symptom.severity === "Severe")
  ).length

  const averageConfidence = sessions.length
    ? Math.round(
        sessions.reduce((sum, session) => sum + getConfidenceValue(session), 0) /
          sessions.length
      )
    : 0

  return (
    <AppLayout>
      <div className="history-page premium-history-page">
        <PageHeader
          title="Symptom History"
          description="Review your diagnosis sessions, filter them, and inspect symptom severity patterns over time."
        />

        <div className="history-overview-grid">
          <div className="history-overview-card">
            <span>Total Sessions</span>
            <strong>{totalSessions}</strong>
          </div>

          <div className="history-overview-card">
            <span>Visible Results</span>
            <strong>{visibleSessions}</strong>
          </div>

          <div className="history-overview-card">
            <span>Severe Cases</span>
            <strong>{severeSessions}</strong>
          </div>

          <div className="history-overview-card">
            <span>Avg. Confidence</span>
            <strong>{averageConfidence}%</strong>
          </div>
        </div>

        <div className="history-filters premium-history-filters">
          <div className="history-filter-main">
            <label>Search Sessions</label>
            <input
              className="history-search"
              type="text"
              placeholder="Search by symptom, prediction, or severity..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <div className="history-filter-field">
            <label>Prediction</label>
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
          </div>

          <div className="history-filter-field">
            <label>Severity</label>
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
          </div>

          <div className="history-filter-field">
            <label>Sort</label>
            <select
              className="history-select"
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value)}
            >
              <option value="Newest">Newest First</option>
              <option value="Oldest">Oldest First</option>
            </select>
          </div>
        </div>

        {loading ? (
          <div className="history-empty premium-history-empty">
            <h3>Loading history...</h3>
          </div>
        ) : filteredHistory.length > 0 ? (
          <div className="history-list premium-history-list">
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
                  className="history-link-card premium-history-link"
                >
                  <div className="history-card advanced-history-card premium-session-card">
                    <div className="premium-session-left">
                      <div className="premium-session-number">
                        #{session.id}
                      </div>

                      <div>
                        <h3>Diagnosis Session</h3>
                        <p>{session.date}</p>
                      </div>
                    </div>

                    <div className="premium-session-main">
                      <div className="premium-session-title-row">
                        <div>
                          <span className="premium-card-label">Predicted Disease</span>
                          <h2>{formatSymptomName(predictionName)}</h2>
                        </div>

                        <div className="premium-session-badges">
                          <span className="score-chip">
                            Score: {session.severityScore ?? 0}
                          </span>

                          {historyStatus && (
                            <span
                              title={historyStatus.title}
                              className={`history-status-chip ${historyStatus.className}`}
                            >
                              {historyStatus.label}
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="history-symptoms-block premium-symptoms-block">
                        <div className="premium-symptoms-title">
                          <h4>Symptoms</h4>
                          <span>{normalizedSymptoms.length} recorded</span>
                        </div>

                        <div className="history-symptom-tags premium-symptom-tags">
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

                      <div className="premium-session-footer">
                        <div className="premium-confidence-wrap">
                          <div className="premium-confidence-text">
                            <span>Confidence</span>
                            <strong>{confidenceValue}%</strong>
                          </div>

                          <div className="premium-confidence-bar">
                            <div
                              className="premium-confidence-fill"
                              style={{ width: `${Math.min(confidenceValue, 100)}%` }}
                            />
                          </div>

                          {severityCondition && (
                            <span className="premium-condition-chip">
                              {severityCondition}
                            </span>
                          )}
                        </div>

                        <span className="history-footer-link">
                          View Details
                        </span>
                      </div>
                    </div>
                  </div>
                </Link>
              )
            })}
          </div>
        ) : (
          <div className="history-empty premium-history-empty">
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