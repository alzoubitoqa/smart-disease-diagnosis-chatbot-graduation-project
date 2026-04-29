function SymptomChip({ label, onClick, selected = false }) {
  return (
    <button
      type="button"
      className={`symptom-chip ${selected ? "active" : ""}`}
      onClick={onClick}
    >
      {label}
    </button>
  )
}

export default SymptomChip