import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import PredictionPage from './pages/PredictionPage';
import ChatPage from './pages/ChatPage';

function Navbar() {
  const location = useLocation();
  
  return (
    <nav className="navbar">
      <Link to="/" className="nav-link" style={{ fontWeight: location.pathname === '/' ? '700' : '400' }}>
        Home (Predict)
      </Link>
      <Link to="/chat" className="nav-link" style={{ fontWeight: location.pathname === '/chat' ? '700' : '400' }}>
        Legal Advisor
      </Link>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<PredictionPage />} />
        <Route path="/chat" element={<ChatPage />} />
      </Routes>
    </Router>
  );
}

export default App;
