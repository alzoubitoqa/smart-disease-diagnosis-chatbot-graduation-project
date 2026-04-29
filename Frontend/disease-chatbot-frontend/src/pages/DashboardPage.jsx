import { useEffect, useMemo, useState } from "react"
import { Link } from "react-router-dom"
import AppLayout from "../layout/AppLayout"
import PageHeader from "../components/common/PageHeader"
import StatCard from "../components/common/StatCard"
import InfoCard from "../components/common/InfoCard"
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

    if (latestSeverity >= 7) {
      healthStatus = "Needs Medical Attention"
      statusClass = "status-critical"
    } else if (latestSeverity >= 4) {
      healthStatus = "Needs Follow-up"
      statusClass = "status-warning"
    }

    return {
      totalSessions,
      totalSymptoms,
      latestPrediction,
      healthStatus,
      statusClass,
      latestSession
    }
  }, [sessions])

  const formatSymptomName = (name) => {
    return String(name || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase())
  }

  return (
    <AppLayout>
      <div className="dashboard-page">
        <PageHeader
          title="Dashboard"
          description="Welcome back. Here is your smart medical assistant overview."
        />

        {loading ? (
          <div className="dashboard-empty-box">Loading dashboard...</div>
        ) : (
          <>
            <div className="stats-grid">
              <StatCard title="Total Sessions" value={dashboardData.totalSessions} />
              <StatCard title="Symptoms Tracked" value={dashboardData.totalSymptoms} />
              <StatCard title="Latest Prediction" value={dashboardData.latestPrediction} />
              <StatCard
                title="Health Status"
                value={
                  <span className={dashboardData.statusClass}>
                    {dashboardData.healthStatus}
                  </span>
                }
              />
            </div>

            <div className="dashboard-panels">
              <InfoCard title="Recent Activity">
                {dashboardData.latestSession ? (
                  <>
                    <p style={{ marginBottom: "14px" }}>
                      Your latest session predicted{" "}
                      <strong>{dashboardData.latestSession.prediction}</strong> on{" "}
                      <strong>{dashboardData.latestSession.date}</strong>.
                    </p>

                    <div className="mini-status-row">
                      <span className="status-pill status-info">
                        Severity Score: {dashboardData.latestSession.severityScore || 0}
                      </span>

                      <span className="status-pill status-soft">
                        {dashboardData.latestSession.symptoms.length} symptoms recorded
                      </span>
                    </div>
                  </>
                ) : (
                  <p>No diagnosis sessions yet. Start your first assessment from the chat page.</p>
                )}
              </InfoCard>

              <InfoCard title="Quick Start">
                <p style={{ marginBottom: "16px" }}>
                  Go to Diagnosis Chat to add symptoms, set severity for each one,
                  and receive a preliminary medical prediction.
                </p>

                <Link to="/chat" className="btn primary-btn">
                  Start Diagnosis
                </Link>
              </InfoCard>
            </div>

            <div className="dashboard-lower-grid">
              <div className="panel-card">
                <h2>Latest Session Summary</h2>

                {dashboardData.latestSession ? (
                  <div className="dashboard-summary-list">
                    {dashboardData.latestSession.symptoms.map((item, index) => (
                      <div key={index} className="dashboard-summary-item">
                        <span>{formatSymptomName(item.name)}</span>
                        <strong>{item.severity}</strong>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="dashboard-empty-box">
                    No session summary available yet. Start a diagnosis from the chat page to generate your first symptom-based result.
                  </div>
                )}
              </div>

              <div className="panel-card">
                <h2>System Status</h2>

                <div className="system-list">
                  <div className="system-item">
                    <span>Routing</span>
                    <strong>Ready</strong>
                  </div>

                  <div className="system-item">
                    <span>Authentication</span>
                    <strong>Connected</strong>
                  </div>

                  <div className="system-item">
                    <span>Chat + Result Flow</span>
                    <strong>Connected</strong>
                  </div>

                  <div className="system-item">
                    <span>Backend Connection</span>
                    <strong>Connected</strong>
                  </div>
                </div>
              </div>
            </div>

            <div className="dashboard-insight-card">
              <h2>System Insight</h2>
              <p>
                This dashboard now reflects your real backend-powered diagnosis sessions,
                including prediction history, symptom tracking, and session summaries.
              </p>
            </div>
          </>
        )}
      </div>
    </AppLayout>
  )
}

export default DashboardPage