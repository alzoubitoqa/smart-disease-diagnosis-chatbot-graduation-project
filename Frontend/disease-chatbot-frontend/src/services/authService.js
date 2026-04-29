import axios from "axios"

const API_BASE_URL = "http://127.0.0.1:8000"

export async function loginRequest(payload) {
  const response = await axios.post(`${API_BASE_URL}/api/auth/login`, payload)
  return response.data
}

export async function registerRequest(payload) {
  const response = await axios.post(`${API_BASE_URL}/api/auth/register`, payload)
  return response.data
}