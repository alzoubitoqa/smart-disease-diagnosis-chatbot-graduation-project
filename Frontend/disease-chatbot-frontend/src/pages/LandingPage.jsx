import { Link } from "react-router-dom"

function LandingPage() {
  return (
    <div className="landing-page-v3">
      <div className="landing-overlay" />

      <section className="hero-section-clean">
        <div className="hero-left-clean">
          <div className="landing-badge">
            AI-Powered Medical Assistant
          </div>

          <h1>Smart Disease Prediction Chatbot</h1>

          <p>
            A modern medical frontend experience that helps users describe symptoms,
            assign severity levels, review previous diagnosis sessions, and receive
            preliminary AI-style predictions in a clean and structured workflow.
          </p>

          <div className="landing-actions">
            <Link to="/register" className="btn primary-btn">
              Get Started
            </Link>

            <Link to="/login" className="btn secondary-btn">
              Login
            </Link>
          </div>

          <div className="landing-stats">
            <div className="landing-stat-card">
              <h3>Structured</h3>
              <span>Symptom workflow</span>
            </div>

            <div className="landing-stat-card">
              <h3>Interactive</h3>
              <span>Diagnosis chat UI</span>
            </div>

            <div className="landing-stat-card">
              <h3>Trackable</h3>
              <span>History and sessions</span>
            </div>
          </div>
        </div>

        <div className="hero-right-clean">
          <div className="landing-preview-card">
            <div className="preview-top">
              <span className="preview-dot" />
              <span className="preview-dot" />
              <span className="preview-dot" />
            </div>

            <div className="preview-body">
              <div className="preview-message preview-bot">
                Hello, I’m your AI medical assistant. Please add your symptoms.
              </div>

              <div className="preview-symptoms">
                <div className="preview-chip">Fever - Moderate</div>
                <div className="preview-chip">Cough - Mild</div>
                <div className="preview-chip">Fatigue - Severe</div>
              </div>

              <div className="preview-message preview-user">
                Send symptom set
              </div>

              <div className="preview-result">
                <h4>Prediction Result</h4>
                <p>Influenza</p>
                <span>Confidence: 87%</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="how-it-works-clean">
        <div className="how-it-works-header">
          <div className="landing-badge">How It Works</div>
          <h2>How the system works</h2>
          <p>
            The assistant follows a clear step-by-step workflow to organize symptoms,
            evaluate severity, and generate a preliminary medical-style result.
          </p>
        </div>

        <div className="how-it-works-grid-clean">
          <div className="how-card">
            <div className="how-step-number">1</div>
            <h3>Add Symptoms</h3>
            <p>
              The user starts by entering symptoms manually or selecting them from the
              quick symptom list inside the diagnosis chat.
            </p>
          </div>

          <div className="how-card">
            <div className="how-step-number">2</div>
            <h3>Assign Severity</h3>
            <p>
              Each symptom is given its own severity level such as Mild, Moderate, or
              Severe to make the assessment more structured.
            </p>
          </div>

          <div className="how-card">
            <div className="how-step-number">3</div>
            <h3>Generate Prediction</h3>
            <p>
              The local diagnosis engine processes the symptom set and produces a
              preliminary disease prediction with a confidence score.
            </p>
          </div>

          <div className="how-card">
            <div className="how-step-number">4</div>
            <h3>Review Results & History</h3>
            <p>
              The user can review the result, precautions, urgency level, and save the
              session in the history page for later inspection.
            </p>
          </div>
        </div>
      </section>

      <section className="features-section-clean">
        <div className="how-it-works-header">
          <div className="landing-badge">Key Features</div>
          <h2>What makes this system useful</h2>
          <p>
            The frontend is designed to simulate a complete diagnosis workflow with
            organized interactions, session tracking, and result review.
          </p>
        </div>

        <div className="features-grid-clean">
          <div className="feature-card-clean">
            <h3>Per-Symptom Severity</h3>
            <p>
              Each symptom can be assigned its own severity level instead of using a
              single score for the whole case.
            </p>
          </div>

          <div className="feature-card-clean">
            <h3>Diagnosis History</h3>
            <p>
              Every submitted diagnosis session is stored and can be reviewed later
              through a searchable history page.
            </p>
          </div>

          <div className="feature-card-clean">
            <h3>Session Details View</h3>
            <p>
              Users can open each saved session to inspect symptoms, severity, prediction,
              confidence, and precautions.
            </p>
          </div>

          <div className="feature-card-clean">
            <h3>Guided Workflow</h3>
            <p>
              The interface leads the user step by step, from symptom entry to result
              interpretation and history review.
            </p>
          </div>

          <div className="feature-card-clean">
            <h3>Mock Authentication</h3>
            <p>
              The frontend includes login, registration, logout, protected routes,
              and a forgot password screen for a realistic user flow.
            </p>
          </div>

          <div className="feature-card-clean">
            <h3>Backend-Ready Structure</h3>
            <p>
              The current frontend is prepared to connect later with a real API for
              authentication, prediction, and session persistence.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

export default LandingPage