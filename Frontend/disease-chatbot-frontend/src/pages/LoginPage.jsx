import { useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import { useAuth } from "../context/AuthContext"
import { useToast } from "../context/ToastContext"
import { loginRequest } from "../services/authService"
import { getProfileRequest, isProfileComplete } from "../services/profileService"

function LoginPage() {
  const navigate = useNavigate()
  const { login } = useAuth()
  const { showToast } = useToast()

  const [formData, setFormData] = useState({
    email: "",
    password: ""
  })

  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!formData.email.trim() || !formData.password.trim()) {
      setError("Please fill in all fields.")
      showToast("Please fill in all fields.", "error")
      return
    }

    if (!formData.email.includes("@")) {
      setError("Please enter a valid email address.")
      showToast("Please enter a valid email address.", "error")
      return
    }

    setError("")
    setLoading(true)

    try {
      const response = await loginRequest({
        email: formData.email,
        password: formData.password
      })

      login(response)
      showToast("Login successful.", "success")

      let profile = null

      try {
        profile = await getProfileRequest(response.user_id)
      } catch (profileError) {
        profile = null
      }

      const complete = isProfileComplete(profile)

      setTimeout(() => {
        if (complete) {
          navigate("/dashboard")
        } else {
          navigate("/initial-profile-setup")
        }
      }, 500)
    } catch (err) {
      const message =
        err?.response?.data?.detail || "Login failed. Please try again."
      setError(message)
      showToast(message, "error")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card enhanced-auth-card">
        <div className="auth-top-link">
          <Link to="/">← Back to Home</Link>
        </div>

        <div className="auth-header-block">
          <h2>Welcome Back</h2>
          <p>
            Sign in to continue using your medical assistant dashboard,
            diagnosis chat, and saved session history.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="auth-field">
            <label>Email Address</label>
            <input
              type="email"
              name="email"
              placeholder="Enter your email address"
              value={formData.email}
              onChange={handleChange}
            />
          </div>

          <div className="auth-field">
            <label>Password</label>
            <div className="password-field">
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                placeholder="Enter your password"
                value={formData.password}
                onChange={handleChange}
              />
              <button
                type="button"
                className="toggle-password-btn"
                onClick={() => setShowPassword((prev) => !prev)}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </div>

          <div className="forgot-password-link">
            <Link to="/forgot-password">Forgot password?</Link>
          </div>

          {error && <p className="form-error">{error}</p>}

          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>

        <p className="auth-footer-text">
          Don’t have an account? <Link to="/register">Create one</Link>
        </p>
      </div>
    </div>
  )
}

export default LoginPage