import axios from "axios"

const API_BASE_URL = "http://127.0.0.1:8000"

export async function getHistoryRequest(userId) {
  const response = await axios.get(`${API_BASE_URL}/api/history/${userId}`)
  return response.data
}