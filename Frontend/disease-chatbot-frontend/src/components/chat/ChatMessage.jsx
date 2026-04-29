function ChatMessage({ sender, text }) {
  return (
    <div className={`message ${sender === "bot" ? "bot" : "user"}`}>
      {text}
    </div>
  )
}

export default ChatMessage