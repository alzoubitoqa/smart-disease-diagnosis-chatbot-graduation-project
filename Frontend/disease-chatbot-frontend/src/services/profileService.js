import axios from "axios"

const API_BASE_URL = "http://127.0.0.1:8000"

export async function getProfileRequest(userId) {
  const response = await axios.get(`${API_BASE_URL}/api/profile/${userId}`)
  return response.data
}

export async function updateProfileRequest(userId, data) {
  const response = await axios.put(`${API_BASE_URL}/api/profile/${userId}`, data)
  return response.data
}

export function isProfileComplete(profile) {
  if (!profile) return false

  const hasAge = profile.age !== null && profile.age !== undefined && String(profile.age).trim() !== ""
  const hasGender = profile.gender !== null && profile.gender !== undefined && String(profile.gender).trim() !== ""

  return hasAge && hasGender
}