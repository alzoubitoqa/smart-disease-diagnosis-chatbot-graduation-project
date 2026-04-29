import axios from "axios"

const API_BASE_URL = "http://127.0.0.1:8000"

export async function getSymptomsRequest() {
  const response = await axios.get(`${API_BASE_URL}/api/symptoms`)
  return response.data
}