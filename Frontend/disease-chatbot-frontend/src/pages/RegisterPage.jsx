import { useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import { useAuth } from "../context/AuthContext"
import { useToast } from "../context/ToastContext"
import { registerRequest } from "../services/authService"
import { getProfileRequest, isProfileComplete } from "../services/profileService"

function RegisterPage() {
  const navigate = useNavigate()
  const { register } = useAuth()
  const { showToast } = useToast()

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: ""
  })

  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
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

    if (
      !formData.name.trim() ||
      !formData.email.trim() ||
      !formData.password.trim() ||
      !formData.confirmPassword.trim()
    ) {
      setError("Please fill in all fields.")
      showToast("Please fill in all fields.", "error")
      return
    }

    if (!formData.email.includes("@")) {
      setError("Please enter a valid email address.")
      showToast("Please enter a valid email address.", "error")
      return
    }

    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters long.")
      showToast("Password must be at least 6 characters long.", "error")
      return
    }

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.")
      showToast("Passwords do not match.", "error")
      return
    }

    setError("")
    setLoading(true)

    try {
      const response = await registerRequest({
        email: formData.email,
        password: formData.password
      })

      register(response)
      showToast("Account created successfully.", "success")

      // ✅ الجديد
      let profile = null

      try {
        profile = await getProfileRequest(response.user_id)
      } catch (e) {
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
        err?.response?.data?.detail || "Registration failed. Please try again."
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
          <h2>Create Account</h2>
          <p>
            Create your account to access the dashboard,
            diagnosis history, and AI-powered symptom workflow.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="auth-field">
            <label>Full Name</label>
            <input
              type="text"
              name="name"
              placeholder="Enter your full name"
              value={formData.name}
              onChange={handleChange}
            />
          </div>

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
                placeholder="Create a password"
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

          <div className="auth-field">
            <label>Confirm Password</label>
            <div className="password-field">
              <input
                type={showConfirmPassword ? "text" : "password"}
                name="confirmPassword"
                placeholder="Confirm your password"
                value={formData.confirmPassword}
                onChange={handleChange}
              />
              <button
                type="button"
                className="toggle-password-btn"
                onClick={() => setShowConfirmPassword((prev) => !prev)}
              >
                {showConfirmPassword ? "Hide" : "Show"}
              </button>
            </div>
          </div>

          {error && <p className="form-error">{error}</p>}

          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? "Creating account..." : "Register"}
          </button>
        </form>

        <p className="auth-footer-text">
          Already have an account? <Link to="/login">Login</Link>
        </p>
      </div>
    </div>
  )
}

export default RegisterPage