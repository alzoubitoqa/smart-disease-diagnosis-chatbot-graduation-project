import { Link } from "react-router-dom"
import { useAuth } from "../context/AuthContext"

function NotFoundPage() {
  const { user } = useAuth()

  return (
    <div className="notfound-page">
      <div className="notfound-card">
        <div className="notfound-code">404</div>
        <h1>Page not found</h1>
        <p>
          The page you are trying to access does not exist or may have been moved.
        </p>

        <div className="notfound-actions">
          <Link to="/" className="btn secondary-btn">
            Back to Home
          </Link>

          {user && (
            <Link to="/dashboard" className="btn primary-btn">
              Go to Dashboard
            </Link>
          )}

          {!user && (
            <Link to="/login" className="btn primary-btn">
              Go to Login
            </Link>
          )}
        </div>
      </div>
    </div>
  )
}

export default NotFoundPage