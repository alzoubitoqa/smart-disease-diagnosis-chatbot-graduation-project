import { Link } from "react-router-dom"

function LandingPage() {
  return (
    <div className="lp4-page">
      <div className="lp4-bg-orb lp4-orb-one" />
      <div className="lp4-bg-orb lp4-orb-two" />
      <div className="lp4-bg-grid" />

      <section className="lp4-hero">
        <div className="lp4-hero-content">
          <div className="lp4-badge">
            <span className="lp4-badge-dot" />
            AI-Powered Medical Assistant
          </div>

          <h1>
            Smart Disease
            <span> Prediction Chatbot</span>
          </h1>

          <p>
            A structured AI medical assistant that helps users enter symptoms,
            assign severity levels, receive a preliminary prediction, review
            confidence scores, and track previous diagnosis sessions.
          </p>

          <div className="lp4-actions">
            <Link to="/register" className="btn lp4-primary-btn">
              Get Started
            </Link>

            <Link to="/login" className="btn lp4-secondary-btn">
              Login
            </Link>
          </div>

          <div className="lp4-stats">
            <div className="lp4-stat-card">
              <strong>41+</strong>
              <span>Disease Classes</span>
            </div>

            <div className="lp4-stat-card">
              <strong>132</strong>
              <span>Symptoms</span>
            </div>

            <div className="lp4-stat-card">
              <strong>1–7</strong>
              <span>Severity Scale</span>
            </div>
          </div>
        </div>

        <div className="lp4-preview-wrap">
          <div className="lp4-preview-card">
            <div className="lp4-preview-top">
              <div className="lp4-window-dots">
                <span />
                <span />
                <span />
              </div>
              <span className="lp4-preview-label">Live Preview</span>
            </div>

            <div className="lp4-preview-body">
              <div className="lp4-chat-bubble lp4-bot-bubble">
                Hello, I’m your AI medical assistant. Please add your symptoms.
              </div>

              <div className="lp4-symptom-row">
                <span>Fever</span>
                <b className="lp4-severity-moderate">Moderate</b>
              </div>

              <div className="lp4-symptom-row">
                <span>Cough</span>
                <b className="lp4-severity-mild">Mild</b>
              </div>

              <div className="lp4-symptom-row">
                <span>Fatigue</span>
                <b className="lp4-severity-severe">Severe</b>
              </div>

              <div className="lp4-send-box">
                Analyze Symptom Set
              </div>

              <div className="lp4-result-card">
                <div className="lp4-result-header">
                  <span>Prediction Result</span>
                  <b>87%</b>
                </div>

                <h3>Influenza</h3>

                <div className="lp4-confidence-track">
                  <div className="lp4-confidence-fill" />
                </div>

                <div className="lp4-top-matches">
                  <div>
                    <span>Influenza</span>
                    <b>87%</b>
                  </div>

                  <div>
                    <span>Common Cold</span>
                    <b>72%</b>
                  </div>

                  <div>
                    <span>Allergy</span>
                    <b>51%</b>
                  </div>
                </div>

                <p>
                  Preliminary result only. Medical consultation is recommended
                  if symptoms continue or worsen.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="lp4-section">
        <div className="lp4-section-header">
          <div className="lp4-mini-badge">How It Works</div>
          <h2>From symptoms to structured prediction</h2>
          <p>
            The system follows a clear workflow to organize user input, evaluate
            severity, generate prediction results, and store diagnosis history.
          </p>
        </div>

        <div className="lp4-workflow">
          <div className="lp4-step-card">
            <div className="lp4-step-icon">01</div>
            <h3>Add Symptoms</h3>
            <p>
              The user enters symptoms manually or selects them from the
              diagnosis interface.
            </p>
          </div>

          <div className="lp4-step-card">
            <div className="lp4-step-icon">02</div>
            <h3>Assign Severity</h3>
            <p>
              Each symptom receives its own severity level to make the case more
              structured.
            </p>
          </div>

          <div className="lp4-step-card">
            <div className="lp4-step-icon">03</div>
            <h3>Generate Prediction</h3>
            <p>
              The prediction engine analyzes the symptom set and produces a
              preliminary result.
            </p>
          </div>

          <div className="lp4-step-card">
            <div className="lp4-step-icon">04</div>
            <h3>Review History</h3>
            <p>
              The user can review previous sessions, predictions, confidence,
              and precautions.
            </p>
          </div>
        </div>
      </section>

      <section className="lp4-section lp4-features-section">
        <div className="lp4-section-header">
          <div className="lp4-mini-badge">Key Features</div>
          <h2>Designed for a smarter medical workflow</h2>
          <p>
            The interface presents the system as organized, explainable, and
            easy to use for preliminary health guidance.
          </p>
        </div>

        <div className="lp4-features-grid">
          <div className="lp4-feature-card">
            <div className="lp4-feature-icon">🩺</div>
            <h3>Per-Symptom Severity</h3>
            <p>
              Each symptom has its own severity level instead of one general
              score for the whole case.
            </p>
          </div>

          <div className="lp4-feature-card">
            <div className="lp4-feature-icon">🧠</div>
            <h3>AI Prediction Flow</h3>
            <p>
              The system shows a clear prediction result supported by a
              confidence score.
            </p>
          </div>

          <div className="lp4-feature-card">
            <div className="lp4-feature-icon">📊</div>
            <h3>Top Matches</h3>
            <p>
              The result can display ranked disease possibilities to make the
              output more informative.
            </p>
          </div>

          <div className="lp4-feature-card">
            <div className="lp4-feature-icon">⚠️</div>
            <h3>Safety-Oriented Output</h3>
            <p>
              The design supports warning messages and safe recommendation
              text for users.
            </p>
          </div>

          <div className="lp4-feature-card">
            <div className="lp4-feature-icon">🕘</div>
            <h3>Diagnosis History</h3>
            <p>
              Previous diagnosis sessions can be stored and reviewed later by
              the user.
            </p>
          </div>

          <div className="lp4-feature-card">
            <div className="lp4-feature-icon">🔐</div>
            <h3>User Flow Ready</h3>
            <p>
              The frontend supports registration, login, protected pages, and
              session-based navigation.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

export default LandingPage