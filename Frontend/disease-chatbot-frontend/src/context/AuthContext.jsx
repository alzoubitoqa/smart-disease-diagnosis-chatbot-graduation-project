import { createContext, useContext, useEffect, useState } from "react"

const AuthContext = createContext()

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)

  useEffect(() => {
    const savedUser = localStorage.getItem("user")
    const savedToken = localStorage.getItem("token")

    if (savedUser && savedToken) {
      setUser(JSON.parse(savedUser))
    }
  }, [])

  const login = (authData) => {
    const userData = {
      user_id: authData.user_id,
      email: authData.email,
      token: authData.access_token
    }

    localStorage.setItem("user", JSON.stringify(userData))
    localStorage.setItem("token", authData.access_token)
    setUser(userData)
  }

  const register = (authData) => {
    const userData = {
      user_id: authData.user_id,
      email: authData.email,
      token: authData.access_token
    }

    localStorage.setItem("user", JSON.stringify(userData))
    localStorage.setItem("token", authData.access_token)
    setUser(userData)
  }

  const logout = () => {
    localStorage.removeItem("user")
    localStorage.removeItem("token")
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}