import Navbar from "./Navbar"
import Sidebar from "./Sidebar"

function AppLayout({ children }) {
  return (
    <div className="app-shell">
      <Navbar />
      <div className="app-body">
        <Sidebar />
        <main className="page-content">{children}</main>
      </div>
    </div>
  )
}

export default AppLayout