function InfoCard({ title, children }) {
  return (
    <div className="panel-card">
      <h2>{title}</h2>
      <div>{children}</div>
    </div>
  )
}

export default InfoCard