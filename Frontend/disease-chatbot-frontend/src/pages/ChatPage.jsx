import { useEffect, useMemo, useState } from "react"
import AppLayout from "../layout/AppLayout"
import ChatMessage from "../components/chat/ChatMessage"
import SymptomChip from "../components/chat/SymptomChip"
import { useAuth } from "../context/AuthContext"
import axios from "axios"
import { getSymptomsRequest } from "../services/symptomsService"

const MIN_REQUIRED_SYMPTOMS = 3

const initialMessages = [
  {
    id: 1,
    sender: "bot",
    text: `Hello, I’m your AI medical assistant. Please begin by adding your symptoms. You need at least ${MIN_REQUIRED_SYMPTOMS} symptoms before diagnosis.`
  }
]

const SYMPTOM_ALIAS_MAP = {
  fever: "high_fever",
  "high temperature": "high_fever",
  temperature: "high_fever",
  sneezing: "continuous_sneezing",
  "runny nose": "runny_nose",
  cold: "chills"
}

function ChatPage() {
  const { user } = useAuth()

  const [messages, setMessages] = useState(initialMessages)
  const [input, setInput] = useState("")
  const [symptoms, setSymptoms] = useState([])
  const [availableSymptoms, setAvailableSymptoms] = useState([])
  const [loadingSymptoms, setLoadingSymptoms] = useState(true)
  const [isTyping, setIsTyping] = useState(false)

  const [chatStage, setChatStage] = useState("collecting_symptoms")

  const [currentResult, setCurrentResult] = useState(null)
  const [historyAwareResult, setHistoryAwareResult] = useState(null)
  const [historyStatusMessage, setHistoryStatusMessage] = useState("")

  useEffect(() => {
    const loadSymptoms = async () => {
      try {
        const data = await getSymptomsRequest()
        setAvailableSymptoms(Array.isArray(data) ? data : [])
      } catch (error) {
        console.error("Failed to load symptoms:", error)
      } finally {
        setLoadingSymptoms(false)
      }
    }

    loadSymptoms()
  }, [])

  useEffect(() => {
    if (chatStage === "diagnosed") return

    if (symptoms.length < MIN_REQUIRED_SYMPTOMS) {
      setChatStage("collecting_symptoms")
    }
  }, [symptoms.length, chatStage])

  const remainingSymptoms = useMemo(() => {
    return Math.max(0, MIN_REQUIRED_SYMPTOMS - symptoms.length)
  }, [symptoms.length])

  const progressPercent = useMemo(() => {
    if (chatStage === "diagnosed") return 100
    if (chatStage === "collecting_severity") return 68
    return Math.min(45, (symptoms.length / MIN_REQUIRED_SYMPTOMS) * 45)
  }, [chatStage, symptoms.length])

  const normalizeText = (text) => {
    return String(text || "")
      .toLowerCase()
      .replace(/,/g, " ")
      .replace(/\./g, " ")
      .replace(/_/g, " ")
      .replace(/\s+/g, " ")
      .trim()
  }

  const normalizeSymptomValue = (text) => {
    const normalized = normalizeText(text)
    return SYMPTOM_ALIAS_MAP[normalized] || normalized.replace(/\s+/g, "_")
  }

  const getDisplayLabel = (symptomValue) => {
    const found = availableSymptoms.find((item) => item.value === symptomValue)
    if (found) return found.label

    return symptomValue
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase())
  }

  const extractSymptomsFromSentence = (text) => {
    const normalizedInput = normalizeText(text)
    const found = []
    const addedValues = new Set()

    availableSymptoms.forEach((symptom) => {
      const normalizedValue = normalizeText(symptom.value)
      const normalizedLabel = normalizeText(symptom.label)

      if (
        normalizedInput.includes(normalizedValue) ||
        normalizedInput.includes(normalizedLabel)
      ) {
        if (!addedValues.has(symptom.value)) {
          found.push(symptom)
          addedValues.add(symptom.value)
        }
      }
    })

    Object.entries(SYMPTOM_ALIAS_MAP).forEach(([alias, mappedValue]) => {
      if (normalizedInput.includes(alias) && !addedValues.has(mappedValue)) {
        found.push({
          value: mappedValue,
          label: getDisplayLabel(mappedValue)
        })
        addedValues.add(mappedValue)
      }
    })

    return found
  }

  const addSymptom = (symptomValue, symptomLabel = symptomValue) => {
    const normalizedValue = normalizeSymptomValue(symptomValue)
    const finalLabel = symptomLabel || getDisplayLabel(normalizedValue)

    setSymptoms((prev) => {
      const exists = prev.find(
        (item) => item.name.toLowerCase() === normalizedValue.toLowerCase()
      )

      if (exists) {
        return prev.filter(
          (item) => item.name.toLowerCase() !== normalizedValue.toLowerCase()
        )
      }

      return [
        ...prev,
        {
          name: normalizedValue,
          label: finalLabel,
          severity: 4
        }
      ]
    })
  }

  const addCustomSymptom = () => {
    const trimmed = input.trim()
    if (!trimmed) return

    const foundSymptoms = extractSymptomsFromSentence(trimmed)

    if (foundSymptoms.length === 0) {
      const mappedValue = normalizeSymptomValue(trimmed)

      const existsInAvailable = availableSymptoms.some(
        (item) => item.value === mappedValue
      )

      if (existsInAvailable || mappedValue !== trimmed.toLowerCase()) {
        setSymptoms((prev) => {
          const exists = prev.find(
            (item) => item.name.toLowerCase() === mappedValue.toLowerCase()
          )

          if (exists) return prev

          return [
            ...prev,
            {
              name: mappedValue,
              label: getDisplayLabel(mappedValue),
              severity: 4
            }
          ]
        })
      }

      setInput("")
      return
    }

    setSymptoms((prev) => {
      const updated = [...prev]

      foundSymptoms.forEach((found) => {
        const normalizedValue = normalizeSymptomValue(found.value)

        const exists = updated.find(
          (item) => item.name.toLowerCase() === normalizedValue.toLowerCase()
        )

        if (!exists) {
          updated.push({
            name: normalizedValue,
            label: found.label || getDisplayLabel(normalizedValue),
            severity: 4
          })
        }
      })

      return updated
    })

    setInput("")
  }

  const updateSeverity = (name, value) => {
    setSymptoms((prev) =>
      prev.map((item) =>
        item.name === name ? { ...item, severity: Number(value) } : item
      )
    )
  }

  const removeSymptom = (name) => {
    setSymptoms((prev) => prev.filter((item) => item.name !== name))
  }

  const getSeverityScore = () => {
    return symptoms.reduce((total, item) => total + Number(item.severity || 0), 0)
  }

  const sendMessage = async () => {
    if (symptoms.length < MIN_REQUIRED_SYMPTOMS || !user?.user_id) return

    const formattedSymptoms = symptoms
      .map((item) => `${item.label} (${item.severity})`)
      .join(", ")

    const userMessage = {
      id: Date.now(),
      sender: "user",
      text: formattedSymptoms
    }

    setMessages((prev) => [...prev, userMessage])
    setIsTyping(true)
    setCurrentResult(null)
    setHistoryAwareResult(null)
    setHistoryStatusMessage("")

    try {
      const baseSymptoms = symptoms.map((item) => ({
        symptom: item.name,
        severity: Number(item.severity)
      }))

      const currentPayload = {
        user_text: formattedSymptoms,
        symptoms: baseSymptoms
      }

      const currentResponse = await axios.post(
        "http://127.0.0.1:8000/api/predict",
        currentPayload
      )

      const currentData = currentResponse.data
      setCurrentResult(currentData)

      let historyData = null

      if (user?.user_id) {
        const historyPayload = {
          user_id: String(user.user_id),
          user_text: formattedSymptoms,
          symptoms: baseSymptoms
        }

        const historyResponse = await axios.post(
          "http://127.0.0.1:8000/api/predict/history-aware",
          historyPayload
        )

        const historyResponseData = historyResponse.data

        if (historyResponseData.history_available) {
          historyData = historyResponseData
          setHistoryAwareResult(historyData)
          setHistoryStatusMessage("")
        } else {
          setHistoryAwareResult(null)
          setHistoryStatusMessage(
            historyResponseData.message ||
              "History-aware prediction is not available yet for this user."
          )
        }
      }

      const botMessage = {
        id: Date.now() + 1,
        sender: "bot",
        text:
          historyData?.ai_response ||
          currentData.ai_response ||
          "No response generated."
      }

      setMessages((prev) => [...prev, botMessage])
      setChatStage("diagnosed")

      localStorage.setItem(
        "latestResult",
        JSON.stringify({
          id: Date.now(),
          symptoms: symptoms.map((item) => ({
            name: item.label,
            severity: item.severity
          })),
          disease: currentData.predicted_disease,
          confidence: currentData.confidence_percentage ?? 0,
          description: currentData.description || "",
          precautions: currentData.precautions || [],
          severityScore: currentData.severity_summary?.total || getSeverityScore(),
          date: new Date().toLocaleDateString("en-GB", {
            day: "2-digit",
            month: "short",
            year: "numeric"
          })
        })
      )
    } catch (error) {
      console.error("Prediction request failed:", error)

      const botMessage = {
        id: Date.now() + 1,
        sender: "bot",
        text: "Something went wrong while processing your symptoms."
      }

      setMessages((prev) => [...prev, botMessage])
    } finally {
      setIsTyping(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault()
      addCustomSymptom()
    }
  }

  const handleNewSession = () => {
    setMessages(initialMessages)
    setInput("")
    setSymptoms([])
    setIsTyping(false)
    setCurrentResult(null)
    setHistoryAwareResult(null)
    setHistoryStatusMessage("")
    setChatStage("collecting_symptoms")
    localStorage.removeItem("latestResult")
  }

  const handleAddMoreSymptoms = () => {
    setChatStage("collecting_symptoms")
  }

  const formatSymptomName = (name) => {
    if (!name || name === "uncertain_case") {
      return "Uncertain Case"
    }

    return String(name || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase())
  }

  const renderTopPredictions = (predictions = []) => {
    if (!predictions.length) return null

    return (
      <div className="chat-result-section">
        <strong>Top 3 Possible Conditions</strong>
        <ul>
          {predictions.slice(0, 3).map((item, index) => (
            <li key={index}>
              {formatSymptomName(item.disease)} ({((item.confidence || 0) * 100).toFixed(2)}%)
            </li>
          ))}
        </ul>
      </div>
    )
  }

  const renderResultCard = (title, result) => {
    if (!result) return null

    const isLowConfidence = Number(result.confidence_percentage || 0) < 30
    const historySymptomsUsed = result.history_symptoms_used || []
    const mergedSequence = result.merged_sequence || []

    return (
      <div className="chat-result-card">
        <div className="chat-result-top">
          <div>
            <span className="chat-section-badge">{title}</span>
            <h3>{formatSymptomName(result.predicted_disease)}</h3>
          </div>

          <div className="chat-confidence-badge">
            {result.confidence_percentage ?? 0}%
          </div>
        </div>

        <div className="chat-confidence-bar">
          <div
            className="chat-confidence-fill"
            style={{ width: `${Math.min(Number(result.confidence_percentage || 0), 100)}%` }}
          />
        </div>

        {isLowConfidence && (
          <div className="chat-warning-box">
            This prediction has low confidence. Please consider the top possible
            conditions below and do not treat this result as a final diagnosis.
          </div>
        )}

        {result.mode === "history_aware" && result.history_used && (
          <div className="chat-info-box">
            <strong>History Applied</strong>

            {historySymptomsUsed.length > 0 && (
              <p>
                Previous Symptoms Used:{" "}
                {historySymptomsUsed
                  .map((item) => formatSymptomName(item.symptom))
                  .join(", ")}
              </p>
            )}

            {mergedSequence.length > 0 && (
              <p>
                Final Sequence:{" "}
                {mergedSequence
                  .map((item) => formatSymptomName(item.symptom))
                  .join(", ")}
              </p>
            )}
          </div>
        )}

        {renderTopPredictions(result.top_k_predictions)}

        <div className="chat-result-grid">
          <div className="chat-result-section">
            <strong>Description</strong>
            <p>{result.description}</p>
          </div>

          <div className="chat-result-section">
            <strong>Precautions</strong>
            <ul>
              {(result.precautions || []).map((item, index) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
          </div>
        </div>

        {result.severity_summary && (
          <div className="chat-severity-summary">
            <span>Total: {result.severity_summary.total}</span>
            <span>Avg: {result.severity_summary.avg}</span>
            <span>Condition: {result.severity_summary.condition}</span>
          </div>
        )}
      </div>
    )
  }

  return (
    <AppLayout>
      <div className="chat-page premium-chat-page">
        <div className="chat-hero">
          <div>
            <span className="chat-section-badge">AI Diagnosis Workflow</span>
            <h1>Diagnosis Chat</h1>
            <p>
              Add symptoms, assign severity from 1 to 7, then generate a
              structured preliminary prediction result.
            </p>
          </div>

          <button className="action-btn chat-new-session-btn" onClick={handleNewSession}>
            New Session
          </button>
        </div>

        <div className="chat-progress-card">
          <div className="chat-progress-steps">
            <div className={`chat-progress-step ${chatStage === "collecting_symptoms" ? "active" : ""}`}>
              <span>1</span>
              <strong>Symptoms</strong>
            </div>

            <div className={`chat-progress-step ${chatStage === "collecting_severity" ? "active" : ""}`}>
              <span>2</span>
              <strong>Severity</strong>
            </div>

            <div className={`chat-progress-step ${chatStage === "diagnosed" ? "active" : ""}`}>
              <span>3</span>
              <strong>Result</strong>
            </div>
          </div>

          <div className="chat-progress-track">
            <div
              className="chat-progress-fill"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        <div className="chat-layout">
          <div className="chat-wrapper">
            <div className="chat-header">
              <div>
                <h2>Medical Conversation</h2>
                <span>
                  Stage:{" "}
                  {chatStage === "collecting_symptoms"
                    ? "Collecting Symptoms"
                    : chatStage === "collecting_severity"
                    ? "Setting Severity"
                    : "Diagnosis Completed"}
                </span>
              </div>
            </div>

            <div className="chat-messages conversation-start">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  sender={message.sender}
                  text={message.text}
                />
              ))}

              {isTyping && (
                <div className="typing-indicator">
                  Assistant is typing...
                </div>
              )}
            </div>

            {(chatStage === "collecting_symptoms" || chatStage === "collecting_severity") && (
              <div className="workflow-card premium-selected-card">
                <div className="chat-card-title-row">
                  <div>
                    <span className="chat-section-badge">Selected Input</span>
                    <h3>Collected Symptoms</h3>
                  </div>

                  <span className="chat-count-chip">
                    {symptoms.length}/{MIN_REQUIRED_SYMPTOMS} minimum
                  </span>
                </div>

                {symptoms.length > 0 ? (
                  <div className="selected-symptoms-preview">
                    {symptoms.map((item) => (
                      <div key={item.name} className="selected-symptom-item">
                        <span>{item.label}</span>
                        <strong>Severity {item.severity}</strong>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="chat-note">No symptoms selected yet.</p>
                )}

                {symptoms.length < MIN_REQUIRED_SYMPTOMS ? (
                  <p className="chat-note" style={{ marginTop: "10px" }}>
                    Please add {remainingSymptoms} more symptom(s) before diagnosis.
                  </p>
                ) : (
                  <p className="chat-note" style={{ marginTop: "10px" }}>
                    Enough symptoms collected. Please set the severity for each symptom.
                  </p>
                )}
              </div>
            )}

            {chatStage === "collecting_symptoms" && (
              <div className="workflow-card premium-add-card">
                <div className="chat-card-title-row">
                  <div>
                    <span className="chat-section-badge">Step 1</span>
                    <h3>Add Symptoms</h3>
                  </div>
                </div>

                <div className="symptom-builder-top">
                  <input
                    type="text"
                    placeholder="Describe symptoms in one sentence..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                  />

                  <button className="action-btn" onClick={addCustomSymptom}>
                    Add Symptom
                  </button>
                </div>

                <div className="quick-symptoms inline-symptoms">
                  {loadingSymptoms ? (
                    <p className="chat-note">Loading symptoms...</p>
                  ) : (
                    availableSymptoms.slice(0, 24).map((symptom, index) => {
                      const isSelected = symptoms.some(
                        (item) =>
                          item.name.toLowerCase() ===
                          normalizeSymptomValue(symptom.value).toLowerCase()
                      )

                      return (
                        <SymptomChip
                          key={`${symptom.value}-${symptom.label}-${index}`}
                          label={symptom.label}
                          selected={isSelected}
                          onClick={() => addSymptom(symptom.value, symptom.label)}
                        />
                      )
                    })
                  )}
                </div>

                {symptoms.length >= MIN_REQUIRED_SYMPTOMS && (
                  <div className="chat-step-actions">
                    <button
                      className="action-btn"
                      onClick={() => setChatStage("collecting_severity")}
                    >
                      Continue to Severity
                    </button>
                  </div>
                )}
              </div>
            )}

            {chatStage === "collecting_severity" && (
              <div className="workflow-card premium-severity-card">
                <div className="chat-card-title-row">
                  <div>
                    <span className="chat-section-badge">Step 2</span>
                    <h3>Set Numeric Severity for Each Symptom</h3>
                  </div>

                  <div className="severity-score-box">
                    Score: <strong>{getSeverityScore()}</strong>
                  </div>
                </div>

                {symptoms.length > 0 ? (
                  <div className="symptom-severity-list">
                    {symptoms.map((symptom) => (
                      <div key={symptom.name} className="symptom-row">
                        <div className="symptom-row-left">
                          <span className="symptom-name">{symptom.label}</span>
                        </div>

                        <div className="symptom-row-right">
                          <select
                            value={symptom.severity}
                            onChange={(e) =>
                              updateSeverity(symptom.name, e.target.value)
                            }
                          >
                            <option value={1}>1 - Very Mild</option>
                            <option value={2}>2 - Mild</option>
                            <option value={3}>3 - Low Moderate</option>
                            <option value={4}>4 - Moderate</option>
                            <option value={5}>5 - Noticeable</option>
                            <option value={6}>6 - Strong</option>
                            <option value={7}>7 - Severe</option>
                          </select>

                          <button
                            className="remove-symptom-btn"
                            onClick={() => removeSymptom(symptom.name)}
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-box">
                    No symptoms selected yet.
                  </div>
                )}

                <div className="chat-final-actions premium-final-actions">
                  <button onClick={handleAddMoreSymptoms}>
                    Add More Symptoms
                  </button>

                  <button
                    onClick={sendMessage}
                    disabled={symptoms.length < MIN_REQUIRED_SYMPTOMS}
                  >
                    Confirm and Diagnose
                  </button>
                </div>
              </div>
            )}

            {chatStage === "diagnosed" && (
              <>
                {renderResultCard("Current Session Prediction", currentResult)}

                {historyStatusMessage && (
                  <div className="workflow-card chat-info-box" style={{ marginTop: "20px" }}>
                    <h3>History-Aware Prediction</h3>
                    <p className="chat-note">{historyStatusMessage}</p>
                  </div>
                )}

                {historyAwareResult &&
                  renderResultCard("History-Aware Prediction", historyAwareResult)}
              </>
            )}
          </div>

          <div className="chat-sidebar-card premium-chat-sidebar">
            <span className="chat-section-badge">Workflow</span>
            <h3>Session Progress</h3>

            <div className="sidebar-progress-list">
              <div className={chatStage === "collecting_symptoms" ? "active" : ""}>
                <span>1</span>
                <p>Collect symptoms</p>
              </div>

              <div className={chatStage === "collecting_severity" ? "active" : ""}>
                <span>2</span>
                <p>Assign severity</p>
              </div>

              <div className={chatStage === "diagnosed" ? "active" : ""}>
                <span>3</span>
                <p>Review diagnosis</p>
              </div>
            </div>

            <h3>Current Status</h3>
            <p className="chat-note">
              {chatStage === "collecting_symptoms" &&
                `You are still collecting symptoms. Minimum required: ${MIN_REQUIRED_SYMPTOMS}.`}
              {chatStage === "collecting_severity" &&
                "You now have enough symptoms. Please confirm severity and run diagnosis."}
              {chatStage === "diagnosed" &&
                "Diagnosis completed. You can review results or start a new session."}
            </p>

            <h3>Selected Symptoms</h3>
            <div className="selected-symptoms-preview">
              {symptoms.length > 0 ? (
                symptoms.map((item) => (
                  <div key={item.name} className="selected-symptom-item">
                    <span>{item.label}</span>
                    <strong>{item.severity}</strong>
                  </div>
                ))
              ) : (
                <p className="chat-note">No symptoms selected yet.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}

export default ChatPage