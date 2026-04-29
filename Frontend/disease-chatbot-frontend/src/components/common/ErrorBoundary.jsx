import React from "react"
import { Link } from "react-router-dom"

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      hasError: false,
      errorMessage: ""
    }
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      errorMessage: error?.message || "Unexpected application error"
    }
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo)
  }

  handleRefresh = () => {
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="app-error-page">
          <div className="app-error-card">
            <div className="app-error-badge">Application Error</div>
            <h1>Something went wrong</h1>
            <p>
              A frontend error occurred while rendering this page. The interface
              was stopped safely to prevent a blank screen.
            </p>

            <div className="app-error-message-box">
              {this.state.errorMessage}
            </div>

            <div className="app-error-actions">
              <button className="btn primary-btn" onClick={this.handleRefresh}>
                Refresh Page
              </button>

              <Link to="/" className="btn secondary-btn">
                Back to Home
              </Link>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary