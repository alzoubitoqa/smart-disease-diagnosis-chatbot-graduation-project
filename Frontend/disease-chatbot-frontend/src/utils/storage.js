export const STORAGE_KEYS = {
  AUTH_USER: "mockAuthUser",
  DIAGNOSIS_SESSIONS: "diagnosisSessions",
  LATEST_RESULT: "latestDiagnosisResult"
}

// =========================
// AUTH STORAGE
// =========================
export function getStoredUser() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.AUTH_USER)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function setStoredUser(user) {
  localStorage.setItem(STORAGE_KEYS.AUTH_USER, JSON.stringify(user))
}

export function clearStoredUser() {
  localStorage.removeItem(STORAGE_KEYS.AUTH_USER)
}

// =========================
// DIAGNOSIS SESSIONS
// =========================
export function getDiagnosisSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.DIAGNOSIS_SESSIONS)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function setDiagnosisSessions(sessions) {
  localStorage.setItem(
    STORAGE_KEYS.DIAGNOSIS_SESSIONS,
    JSON.stringify(sessions)
  )
}

export function addDiagnosisSession(session) {
  const existing = getDiagnosisSessions()
  const updated = [session, ...existing]
  setDiagnosisSessions(updated)
  return updated
}

export function clearDiagnosisSessions() {
  localStorage.removeItem(STORAGE_KEYS.DIAGNOSIS_SESSIONS)
}

// =========================
// LATEST RESULT
// =========================
export function getLatestDiagnosisResult() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.LATEST_RESULT)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function setLatestDiagnosisResult(result) {
  localStorage.setItem(STORAGE_KEYS.LATEST_RESULT, JSON.stringify(result))
}

export function clearLatestDiagnosisResult() {
  localStorage.removeItem(STORAGE_KEYS.LATEST_RESULT)
}

// =========================
// COMPATIBILITY HELPERS
// =========================
export function getDiagnosisHistory() {
  return getDiagnosisSessions()
}

export function saveDiagnosisSession(session) {
  return addDiagnosisSession(session)
}

// =========================
// CLEAR ALL DIAGNOSIS DATA
// =========================
export function clearAllDiagnosisStorage() {
  clearDiagnosisSessions()
  clearLatestDiagnosisResult()
}