import { Link, useLocation, useNavigate } from "react-router-dom"
import { useAuth } from "../context/AuthContext"
import { useToast } from "../context/ToastContext"

function Navbar() {
  const location = useLocation()
  const navigate = useNavigate()
  const { user, logout } = useAuth()
  const { showToast } = useToast()

  const navItems = [
    { label: "Dashboard", path: "/dashboard" },
    { label: "Chat", path: "/chat" },
    { label: "History", path: "/history" },
    { label: "Profile", path: "/profile" }
  ]

  const handleLogout = () => {
    logout()
    showToast("Logged out successfully.", "info")
    navigate("/login")
  }

  return (
    <header className="top-navbar">
      <div className="brand">
        <div className="brand-icon">+</div>
        <div>
          <h2>MedAssist AI</h2>
          <span>Smart Diagnosis Support</span>
        </div>
      </div>

      <div className="navbar-right">
        <nav className="top-nav-links">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={location.pathname === item.path ? "top-active-link" : ""}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        {user && (
          <div className="nav-user-box">
            <span>{user.name}</span>
            <button className="logout-btn" onClick={handleLogout}>
              Logout
            </button>
          </div>
        )}
      </div>
    </header>
  )
}

export default Navbar