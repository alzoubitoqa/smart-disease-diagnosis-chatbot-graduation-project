import { useState } from "react"
import { useNavigate } from "react-router-dom"
import AppLayout from "../layout/AppLayout"
import { useAuth } from "../context/AuthContext"
import { useToast } from "../context/ToastContext"
import { updateProfileRequest } from "../services/profileService"

function InitialProfileSetupPage() {
  const navigate = useNavigate()
  const { user } = useAuth()
  const { showToast } = useToast()

  const [form, setForm] = useState({
    age: "",
    gender: "",
    medical_history: ""
  })

  const [saving, setSaving] = useState(false)

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!user?.user_id) {
      showToast("User not found. Please login again.", "error")
      return
    }

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
      await updateProfileRequest(user.user_id, {
        age: Number(form.age),
        gender: form.gender,
        medical_history: form.medical_history.trim() || "None"
      })

      showToast("Profile setup completed successfully.", "success")
      navigate("/dashboard")
    } catch (err) {
      showToast("Failed to save profile setup.", "error")
    } finally {
      setSaving(false)
    }
  }

  return (
    <AppLayout>
      <div className="profile-page">
        <div className="page-header">
          <h1>Complete Your Medical Profile</h1>
          <p>
            Before entering the system for the first time, please complete your
            basic medical information.
          </p>
        </div>

        <div className="profile-card modern-profile-card">
          <div className="profile-card-header">
            <h3>Initial User Setup</h3>
            <p>
              This page appears only for new users. After saving, you can enter
              the full system directly.
            </p>
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
                placeholder="Write any medical history, or type None..."
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
                {saving ? "Saving..." : "Enter System"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </AppLayout>
  )
}

export default InitialProfileSetupPage