import { useState } from "react"
import { Link } from "react-router-dom"
import { useToast } from "../context/ToastContext"

function ForgotPasswordPage() {
  const [email, setEmail] = useState("")
  const [error, setError] = useState("")
  const { showToast } = useToast()

  const handleSubmit = (e) => {
    e.preventDefault()

    if (!email.trim()) {
      setError("Please enter your email address.")
      showToast("Please enter your email address.", "error")
      return
    }

    if (!email.includes("@")) {
      setError("Please enter a valid email address.")
      showToast("Please enter a valid email address.", "error")
      return
    }

    setError("")
    showToast("Reset link sent successfully in frontend demo.", "success")
  }

  return (
    <div className="auth-page">
      <div className="auth-card enhanced-auth-card">
        <div className="auth-top-link">
          <Link to="/login">← Back to Login</Link>
        </div>

        <div className="auth-header-block">
          <h2>Forgot Password</h2>
          <p>
            Enter your email address and we will send you a mock password reset
            link for the frontend version of the project.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="auth-field">
            <label>Email Address</label>
            <input
              type="email"
              placeholder="Enter your email address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          {error && <p className="form-error">{error}</p>}

          <button type="submit" className="auth-submit-btn">
            Send Reset Link
          </button>
        </form>

        <p className="auth-footer-text">
          Remembered your password? <Link to="/login">Login</Link>
        </p>
      </div>
    </div>
  )
}

export default ForgotPasswordPage