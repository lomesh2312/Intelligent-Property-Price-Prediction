import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

export default function ChatPage() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hello! I am your Legal Property Advisor. Ask me anything about property law, contracts, and regulations.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endOfMessagesRef = useRef(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', { query: userMsg.text });
      setMessages(prev => [...prev, { role: 'bot', text: response.data.response }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'bot', text: 'Sorry, I am having trouble connecting to the server.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="hero-section" style={{ padding: '2rem 2rem 3rem' }}>
        <h1 className="hero-title">Legal Advisor</h1>
        <p className="hero-subtitle">
          Consult our RAG-powered engine for precise answers to your real estate legal questions.
        </p>
      </div>

      <div className="main-container" style={{ maxWidth: '800px' }}>
        <div className="chat-box">
          <div className="chat-history">
            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-bubble ${msg.role}`}>
                {msg.text}
              </div>
            ))}
            {loading && (
              <div className="chat-bubble bot" style={{ opacity: 0.7 }}>
                Typing...
              </div>
            )}
            <div ref={endOfMessagesRef} />
          </div>

          <form onSubmit={handleSubmit} className="chat-input-row">
            <input 
              type="text" 
              className="form-input chat-input" 
              placeholder="Type your question..." 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
            />
            <button type="submit" className="btn btn-dark" disabled={loading}>
              Send
            </button>
          </form>
        </div>
      </div>
    </>
  );
}
