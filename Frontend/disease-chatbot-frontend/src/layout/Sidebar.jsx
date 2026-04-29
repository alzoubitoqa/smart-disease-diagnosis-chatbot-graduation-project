import { Link, useLocation } from "react-router-dom"

function Sidebar() {
  const location = useLocation()

  const menuItems = [
    { name: "Dashboard", path: "/dashboard", hint: "Overview and activity" },
    { name: "Diagnosis Chat", path: "/chat", hint: "Start a new session" },
    { name: "Results", path: "/result", hint: "Prediction summary" },
    { name: "History", path: "/history", hint: "Past mock sessions" },
    { name: "Profile", path: "/profile", hint: "User information" }
  ]

  return (
    <aside className="sidebar">
      <h3 className="sidebar-title">Main Menu</h3>

      <div className="sidebar-links">
        {menuItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={location.pathname === item.path ? "active-link" : ""}
          >
            <strong>{item.name}</strong>
            <span className="sidebar-hint">{item.hint}</span>
          </Link>
        ))}
      </div>
    </aside>
  )
}

export default Sidebar