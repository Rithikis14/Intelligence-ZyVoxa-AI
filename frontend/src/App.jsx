
import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content:
        "Hello! I'm ZyVoxa Intelligence. I'm here to help you learn about ZyVoxa, our services, projects, team and technology.",
      sources: [],
    },
  ])

  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth',
    })
  }, [messages, loading])

  const sendMessage = async (text = question) => {
    const trimmedQuestion = text.trim()

    if (!trimmedQuestion || loading) return

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: trimmedQuestion,
      sources: [],
    }

    setMessages((previous) => [...previous, userMessage])
    setQuestion('')
    setError('')
    setLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: trimmedQuestion,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => null)
        throw new Error(
          errorData?.detail || 'Unable to get a response. Please try again.'
        )
      }

      const data = await response.json()

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
      }

      setMessages((previous) => [...previous, assistantMessage])
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const startNewChat = () => {
    setMessages([
      {
        id: Date.now(),
        role: 'assistant',
        content:
          "Hello! I'm ZyVoxa Intelligence. What would you like to know about ZyVoxa?",
        sources: [],
      },
    ])
    setQuestion('')
    setError('')
    inputRef.current?.focus()
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      sendMessage()
    }
  }

  const suggestions = [
    'What is ZyVoxa?',
    'What services do you offer?',
    'Tell me about VoxaVault',
    'What is ZyVoxa LaunchPad?',
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">Z</div>
          <div>
            <h2>ZyVoxa</h2>
            <span>INTELLIGENCE</span>
          </div>
        </div>

        <button className="new-chat-button" onClick={startNewChat}>
          <span>＋</span> New conversation
        </button>

        <div className="sidebar-section">
          <span className="sidebar-label">YOUR ASSISTANT</span>

          <div className="assistant-card">
            <div className="assistant-avatar">Z</div>
            <div>
              <strong>ZyVoxa Assistant</strong>
              <span>Company knowledge</span>
            </div>
            <span className="online-dot" />
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="privacy-indicator">
            <span>✦</span>
            <div>
              <strong>Knowledge powered</strong>
              <small>Grounded in company data</small>
            </div>
          </div>
          <p>© 2026 ZyVoxa. All rights reserved.</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="topbar-label">WORKSPACE /</span>
            <strong>Company Assistant</strong>
          </div>

          <div className="connection-status">
            <span className="online-dot" />
            System online
          </div>
        </header>

        <section className="chat-area">
          <div className="chat-container">
            <div className="welcome-heading">
              <div className="welcome-icon">Z</div>
              <span className="eyebrow">INTELLIGENCE MEETS IMAGINATION</span>
              <h1>How can I help you <span>today?</span></h1>
              <p>
                Ask me anything about ZyVoxa, our solutions,
                projects, team and technology.
              </p>
            </div>

            <div className="messages">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`message-row ${message.role}`}
                >
                  {message.role === 'assistant' && (
                    <div className="message-avatar">Z</div>
                  )}

                  <div className="message-content">
                    <div className="message-author">
                      {message.role === 'assistant' ? 'ZyVoxa Intelligence' : 'You'}
                    </div>

                    <div className="message-text">
                      {message.content}
                    </div>

                    {message.sources?.length > 0 && (
                      <div className="sources">
                        <div className="sources-heading">
                          <span>✧</span> Sources
                        </div>

                        {message.sources.map((source, index) => (
                          <div className="source-item" key={index}>
                            <span className="source-number">{index + 1}</span>
                            <span className="source-name">
                              {source.source.split(/[\\/]/).pop()}
                            </span>
                            <span className="source-page">
                              Page {source.page}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="message-row assistant">
                  <div className="message-avatar">Z</div>
                  <div className="message-content">
                    <div className="message-author">ZyVoxa Intelligence</div>
                    <div className="typing-indicator">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {messages.length === 1 && !loading && (
              <div className="suggestions">
                <span className="suggestion-label">GET STARTED</span>
                <div className="suggestion-grid">
                  {suggestions.map((suggestion) => (
                    <button
                      key={suggestion}
                      className="suggestion-card"
                      onClick={() => sendMessage(suggestion)}
                    >
                      {suggestion}
                      <span>↗</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {error && (
              <div className="error-message">
                <span>!</span>
                {error}
              </div>
            )}

            <div className="input-section">
              <div className="input-box">
                <textarea
                  ref={inputRef}
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask anything about ZyVoxa..."
                  rows={1}
                  disabled={loading}
                />

                <button
                  className="send-button"
                  onClick={() => sendMessage()}
                  disabled={!question.trim() || loading}
                  aria-label="Send message"
                >
                  ↗
                </button>
              </div>

              <div className="input-footer">
                <span>Enter to send · Shift + Enter for a new line</span>
                <span>Powered by ZyVoxa RAG</span>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
