import { useEffect, useMemo, useState } from "react"
import { Link } from "react-router-dom"
import AppLayout from "../layout/AppLayout"
import { useAuth } from "../context/AuthContext"
import { getHistoryRequest } from "../services/historyService"

function DashboardPage() {
  const { user } = useAuth()
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadDashboardHistory = async () => {
      if (!user?.user_id) {
        setSessions([])
        setLoading(false)
        return
      }

      try {
        const data = await getHistoryRequest(user.user_id)
        setSessions(Array.isArray(data) ? data : [])
      } catch (error) {
        console.error("Failed to load dashboard history:", error)
        setSessions([])
      } finally {
        setLoading(false)
      }
    }

    loadDashboardHistory()
  }, [user])

  const formatSymptomName = (name) => {
    return String(name || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase())
  }

  const displayName =
    user?.full_name ||
    user?.name ||
    user?.username ||
    user?.email?.split("@")[0] ||
    "User"

  const dashboardData = useMemo(() => {
    const totalSessions = sessions.length

    const totalSymptoms = sessions.reduce((total, session) => {
      return total + (Array.isArray(session.symptoms) ? session.symptoms.length : 0)
    }, 0)

    const latestSession = sessions.length > 0 ? sessions[0] : null
    const latestPrediction = latestSession ? latestSession.prediction : "No diagnosis yet"
    const latestSeverity = latestSession ? latestSession.severityScore || 0 : 0

    let healthStatus = "Stable"
    let statusClass = "status-good"
    let statusLabel = "Low Risk"
    let guidanceText =
      "Your latest result appears stable. Continue monitoring symptoms and start a new diagnosis session if new symptoms appear."

    if (latestSeverity >= 7) {
      healthStatus = "Needs Medical Attention"
      statusClass = "status-critical"
      statusLabel = "High Priority"
      guidanceText =
        "Your latest severity score is high. The system result is preliminary only, so medical consultation is recommended if symptoms continue, worsen, or feel unusual."
    } else if (latestSeverity >= 4) {
      healthStatus = "Needs Follow-up"
      statusClass = "status-warning"
      statusLabel = "Monitor Case"
      guidanceText =
        "Your latest severity score suggests that symptoms should be monitored. Consider follow-up if the symptoms persist or become stronger."
    }

    return {
      totalSessions,
      totalSymptoms,
      latestPrediction,
      latestSeverity,
      healthStatus,
      statusClass,
      statusLabel,
      guidanceText,
      latestSession
    }
  }, [sessions])

  const latestSymptoms = Array.isArray(dashboardData.latestSession?.symptoms)
    ? dashboardData.latestSession.symptoms
    : []

  return (
    <AppLayout>
      <div className="dashboard-page premium-dashboard-page">
        {loading ? (
          <div className="dashboard-empty-box">Loading dashboard...</div>
        ) : (
          <>
            <section className="dashboard-hero-card">
              <div className="dashboard-hero-left">
                <span className="dashboard-hero-badge">Medical Overview</span>
                <h1>Welcome back, {displayName}</h1>
                <p>
                  Here is a clear overview of your diagnosis activity, latest
                  prediction, tracked symptoms, and health guidance.
                </p>

                <div className="dashboard-hero-actions">
                  <Link to="/chat" className="btn dashboard-main-btn">
                    Start New Diagnosis
                  </Link>

                  <Link to="/history" className="btn dashboard-ghost-btn">
                    View History
                  </Link>
                </div>
              </div>

              <div className="dashboard-hero-right">
                <div className="dashboard-status-panel">
                  <span>Current Health Status</span>
                  <strong className={dashboardData.statusClass}>
                    {dashboardData.healthStatus}
                  </strong>
                  <p>{dashboardData.statusLabel}</p>
                </div>
              </div>
            </section>

            <section className="dashboard-premium-stats">
              <div className="dashboard-stat-box">
                <div className="dashboard-stat-icon">📁</div>
                <span>Total Sessions</span>
                <strong>{dashboardData.totalSessions}</strong>
              </div>

              <div className="dashboard-stat-box">
                <div className="dashboard-stat-icon">🩺</div>
                <span>Symptoms Tracked</span>
                <strong>{dashboardData.totalSymptoms}</strong>
              </div>

              <div className="dashboard-stat-box dashboard-wide-stat">
                <div className="dashboard-stat-icon">🧠</div>
                <span>Latest Prediction</span>
                <strong>{formatSymptomName(dashboardData.latestPrediction)}</strong>
              </div>

              <div className="dashboard-stat-box dashboard-alert-stat">
                <div className="dashboard-stat-icon">⚠️</div>
                <span>Severity Score</span>
                <strong>{dashboardData.latestSeverity}</strong>
              </div>
            </section>

            <section className="dashboard-main-grid">
              <div className="dashboard-feature-card latest-prediction-card">
                <div className="dashboard-card-top">
                  <div>
                    <span className="dashboard-small-label">Latest Result</span>
                    <h2>{formatSymptomName(dashboardData.latestPrediction)}</h2>
                  </div>

                  <span className={`dashboard-status-chip ${dashboardData.statusClass}`}>
                    {dashboardData.statusLabel}
                  </span>
                </div>

                {dashboardData.latestSession ? (
                  <>
                    <p>
                      The latest diagnosis session was created on{" "}
                      <strong>{dashboardData.latestSession.date}</strong> and
                      included <strong>{latestSymptoms.length}</strong> recorded
                      symptoms.
                    </p>

                    <div className="dashboard-mini-pills">
                      <span>Severity Score: {dashboardData.latestSeverity}</span>
                      <span>{latestSymptoms.length} symptoms recorded</span>
                    </div>
                  </>
                ) : (
                  <p>
                    No diagnosis sessions yet. Start your first assessment from
                    the diagnosis chat page.
                  </p>
                )}
              </div>

              <div className="dashboard-feature-card quick-start-card">
                <span className="dashboard-small-label">Quick Start</span>
                <h2>Run a new symptom check</h2>
                <p>
                  Add symptoms, set severity for each one, and receive a
                  preliminary medical prediction with a structured result.
                </p>

                <Link to="/chat" className="btn dashboard-main-btn">
                  Start Diagnosis
                </Link>
              </div>
            </section>

            <section className="dashboard-lower-premium-grid">
              <div className="dashboard-feature-card">
                <div className="dashboard-card-top">
                  <div>
                    <span className="dashboard-small-label">Latest Session</span>
                    <h2>Symptom Summary</h2>
                  </div>

                  {latestSymptoms.length > 0 && (
                    <span className="dashboard-count-chip">
                      {latestSymptoms.length} items
                    </span>
                  )}
                </div>

                {latestSymptoms.length > 0 ? (
                  <div className="dashboard-symptom-list">
                    {latestSymptoms.map((item, index) => (
                      <div key={index} className="dashboard-symptom-row">
                        <span>{formatSymptomName(item.name)}</span>
                        <strong
                          className={`severity-${String(
                            item.severity || "moderate"
                          ).toLowerCase()}`}
                        >
                          {item.severity || "Moderate"}
                        </strong>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="dashboard-empty-box">
                    No session summary available yet. Start a diagnosis from the
                    chat page to generate your first symptom-based result.
                  </div>
                )}
              </div>

              <div className="dashboard-feature-card dashboard-guidance-card">
                <div className="dashboard-card-top">
                  <div>
                    <span className="dashboard-small-label">Recommendation</span>
                    <h2>Health Guidance</h2>
                  </div>

                  <span className={`dashboard-status-chip ${dashboardData.statusClass}`}>
                    {dashboardData.statusLabel}
                  </span>
                </div>

                <p>{dashboardData.guidanceText}</p>

                <div className="dashboard-guidance-note">
                  <strong>Important note</strong>
                  <span>
                    This system provides preliminary support only and does not
                    replace professional medical diagnosis.
                  </span>
                </div>

                <div className="dashboard-mini-pills">
                  <span>Preliminary support</span>
                  <span>Doctor consultation when needed</span>
                </div>
              </div>
            </section>

            <section className="dashboard-insight-premium">
              <div>
                <span className="dashboard-small-label">System Insight</span>
                <h2>Backend-powered diagnosis overview</h2>
                <p>
                  This dashboard reflects real diagnosis sessions, symptom
                  tracking, prediction history, and session summaries using the
                  connected backend workflow.
                </p>
              </div>
            </section>
          </>
        )}
      </div>
    </AppLayout>
  )
}

export default DashboardPage