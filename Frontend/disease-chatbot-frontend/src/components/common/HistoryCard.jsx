function HistoryCard({ title, date, symptoms, prediction }) {
  return (
    <div className="history-card">
      <div className="history-top">
        <h3>{title}</h3>
        <span className="history-date">{date}</span>
      </div>

      <p><strong>Symptoms:</strong> {symptoms}</p>
      <p className="history-prediction"><strong>Prediction:</strong> {prediction}</p>
    </div>
  )
}

export default HistoryCard