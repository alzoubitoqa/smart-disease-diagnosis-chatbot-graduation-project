import { useEffect, useState } from "react"
import AppLayout from "../layout/AppLayout"
import { useAuth } from "../context/AuthContext"
import { useToast } from "../context/ToastContext"
import {
  getProfileRequest,
  updateProfileRequest
} from "../services/profileService"

function ProfilePage() {
  const { user } = useAuth()
  const { showToast } = useToast()

  const displayName =
    user?.full_name ||
    user?.name ||
    user?.username ||
    user?.email?.split("@")[0] ||
    "User"

  const displayEmail = user?.email || "No email available"

  const [form, setForm] = useState({
    age: "",
    gender: "",
    medical_history: ""
  })

  const [savedProfile, setSavedProfile] = useState({
    age: "",
    gender: "",
    medical_history: ""
  })

  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    const loadProfile = async () => {
      if (!user?.user_id) {
        setLoading(false)
        return
      }

      try {
        const data = await getProfileRequest(user.user_id)

        const profileData = {
          age: data.age || "",
          gender: data.gender || "",
          medical_history: data.medical_history || ""
        }

        setSavedProfile(profileData)
        setForm(profileData)
      } catch (err) {
        showToast("Failed to load profile.", "error")
      } finally {
        setLoading(false)
      }
    }

    loadProfile()
  }, [user, showToast])

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!user?.user_id) return

    if (!form.age || Number(form.age) <= 0) {
      showToast("Please enter a valid age.", "error")
      return
    }

    if (!form.gender) {
      showToast("Please select gender.", "error")
      return
    }

    setSaving(true)

    try {
      const payload = {
        age: Number(form.age),
        gender: form.gender,
        medical_history: form.medical_history?.trim() || "None"
      }

      const updated = await updateProfileRequest(user.user_id, payload)

      const updatedProfile = {
        age: updated.age || "",
        gender: updated.gender || "",
        medical_history: updated.medical_history || ""
      }

      setSavedProfile(updatedProfile)
      setForm(updatedProfile)

      showToast("Profile updated successfully.", "success")
    } catch (err) {
      showToast("Failed to update profile.", "error")
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <AppLayout>
        <div className="profile-page">
          <div className="profile-loading-card">
            <p className="profile-loading">Loading profile...</p>
          </div>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout>
      <div className="profile-page premium-profile-page">
        <div className="profile-hero-card">
          <div className="profile-avatar">
            {displayName.charAt(0).toUpperCase()}
          </div>

          <div className="profile-hero-info">
            <span className="profile-welcome-badge">Medical Profile</span>
            <h1>Welcome, {displayName}</h1>
            <p>
              Manage your personal health information and keep your medical
              profile organized for future diagnosis sessions.
            </p>

            <div className="profile-hero-meta">
              <span>{displayEmail}</span>
              <span>Profile Status: Active</span>
            </div>
          </div>
        </div>

        <div className="profile-content-grid">
          <div className="profile-card modern-profile-card">
            <div className="profile-card-header">
              <div>
                <span className="profile-section-badge">Edit Information</span>
                <h3>User Information</h3>
                <p>Edit your age, gender, and medical history here.</p>
              </div>
            </div>

            <form className="profile-form" onSubmit={handleSubmit}>
              <div className="profile-grid">
                <div className="profile-field">
                  <label htmlFor="age">Age</label>
                  <input
                    id="age"
                    type="number"
                    name="age"
                    placeholder="Enter your age"
                    value={form.age}
                    onChange={handleChange}
                  />
                </div>

                <div className="profile-field">
                  <label htmlFor="gender">Gender</label>
                  <select
                    id="gender"
                    name="gender"
                    value={form.gender}
                    onChange={handleChange}
                  >
                    <option value="">Select gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
              </div>

              <div className="profile-field">
                <label htmlFor="medical_history">Medical History</label>
                <textarea
                  id="medical_history"
                  name="medical_history"
                  rows="5"
                  placeholder="Write any medical history..."
                  value={form.medical_history}
                  onChange={handleChange}
                />
              </div>

              <div className="profile-form-actions">
                <button
                  type="submit"
                  className="profile-save-btn"
                  disabled={saving}
                >
                  {saving ? "Saving..." : "Save Profile"}
                </button>
              </div>
            </form>
          </div>

          <div className="profile-card saved-profile-card">
            <div className="profile-card-header">
              <div>
                <span className="profile-section-badge">Saved Summary</span>
                <h3>Saved Profile Data</h3>
                <p>Your saved information appears here in an organized format.</p>
              </div>
            </div>

            <div className="saved-profile-grid">
              <div className="saved-profile-item">
                <span>Full Name</span>
                <strong>{displayName}</strong>
              </div>

              <div className="saved-profile-item">
                <span>Email</span>
                <strong>{displayEmail}</strong>
              </div>

              <div className="saved-profile-item">
                <span>Age</span>
                <strong>{savedProfile.age || "Not added yet"}</strong>
              </div>

              <div className="saved-profile-item">
                <span>Gender</span>
                <strong>{savedProfile.gender || "Not added yet"}</strong>
              </div>

              <div className="saved-profile-item full-width">
                <span>Medical History</span>
                <strong>{savedProfile.medical_history || "Not added yet"}</strong>
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}

export default ProfilePage