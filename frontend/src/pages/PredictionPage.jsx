import React, { useState } from 'react';
import axios from 'axios';
import { Home, MapPin, Inbox } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function PredictionPage() {
  const [formData, setFormData] = useState({
    area: '',
    bedrooms: '',
    bathrooms: '',
    stories: '',
    parking: '',
    guestroom: 'Select...',
    mainroad: 'Select...',
    prefarea: 'Select...',
    basement: 'Select...',
    airconditioning: 'Select...',
    furnishingstatus: 'Select...'
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    const numericFields = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking'];
    const dropdownFields = ['guestroom', 'mainroad', 'prefarea', 'basement', 'airconditioning', 'furnishingstatus'];

    for (let field of dropdownFields) {
      if (formData[field] === 'Select...') {
        setError('Please select all dropdown options.');
        return;
      }
    }

    for (let field of numericFields) {
      if (!formData[field] || Number(formData[field]) < 0) {
        setError('Please fill all numeric fields with valid numbers.');
        return;
      }
    }

    setLoading(true);
    setResult(null);

    try {
      const payload = {
        area: parseFloat(formData.area),
        bedrooms: parseInt(formData.bedrooms),
        bathrooms: parseInt(formData.bathrooms),
        stories: parseInt(formData.stories),
        parking: parseInt(formData.parking),
        guestroom: formData.guestroom,
        mainroad: formData.mainroad,
        prefarea: formData.prefarea,
        basement: formData.basement,
        airconditioning: formData.airconditioning,
        furnishingstatus: formData.furnishingstatus
      };

      const response = await axios.post('http://localhost:8000/predict', payload);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while predicting.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="hero-section">
        <h1 className="hero-title">Predict Price</h1>
        <p className="hero-subtitle">
          Enter the details of your property to get an intelligent price prediction
          along with an AI-generated advisory review of risks and market trends.
        </p>
      </div>

      <div className="main-container">
        <div className="form-column">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <input type="number" name="area" className="form-input" placeholder="Area (sq ft > 1650)" onChange={handleChange} value={formData.area} min="1650" max="16200" />
              <input type="number" name="bedrooms" className="form-input" placeholder="Bedrooms (1-6)" onChange={handleChange} value={formData.bedrooms} min="1" max="6" />
            </div>

            <div className="form-group">
              <input type="number" name="bathrooms" className="form-input" placeholder="Bathrooms (1-4)" onChange={handleChange} value={formData.bathrooms} min="1" max="4" />
              <input type="number" name="stories" className="form-input" placeholder="Stories (1-4)" onChange={handleChange} value={formData.stories} min="1" max="4" />
            </div>

            <div className="form-group">
              <input type="number" name="parking" className="form-input" placeholder="Parking Spaces (0-3)" onChange={handleChange} value={formData.parking} min="0" max="3" />
              <select name="furnishingstatus" className="form-input" onChange={handleChange} value={formData.furnishingstatus}>
                <option value="Select...">Furnishing Status</option>
                <option value="furnished">Furnished</option>
                <option value="semi-furnished">Semi-Furnished</option>
                <option value="unfurnished">Unfurnished</option>
              </select>
            </div>

            <div className="form-group">
              <select name="guestroom" className="form-input" onChange={handleChange} value={formData.guestroom}>
                <option value="Select...">Guest Room?</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
              <select name="mainroad" className="form-input" onChange={handleChange} value={formData.mainroad}>
                <option value="Select...">Main Road Access?</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div className="form-group">
              <select name="prefarea" className="form-input" onChange={handleChange} value={formData.prefarea}>
                <option value="Select...">Preferred Area?</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
              <select name="basement" className="form-input" onChange={handleChange} value={formData.basement}>
                <option value="Select...">Basement?</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div className="form-group">
              <select name="airconditioning" className="form-input" onChange={handleChange} value={formData.airconditioning}>
                <option value="Select...">Air Conditioning?</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            {error && <p className="error-text">{error}</p>}

            <button type="submit" className="btn btn-dark" disabled={loading} style={{ marginTop: '1rem' }}>
              {loading ? 'Predicting...' : 'Predict Price'}
            </button>
          </form>

          <div className="contact-blocks">
            <div className="contact-block">
              <Home className="contact-block-icon" size={24} color="var(--card-bg-dark)" />
              <div className="contact-block-value">Accurate AI Models</div>
              <div className="contact-block-desc">Trained on real housing data sets</div>
            </div>
            <div className="contact-block">
              <Inbox className="contact-block-icon" size={24} color="var(--card-bg-dark)" />
              <div className="contact-block-value">support@ai.com</div>
              <div className="contact-block-desc">Contact us for API access</div>
            </div>
            <div className="contact-block">
              <MapPin className="contact-block-icon" size={24} color="var(--card-bg-dark)" />
              <div className="contact-block-value">Global Coverage</div>
              <div className="contact-block-desc">Estimates for multiple regions</div>
            </div>
          </div>
        </div>

        <div className="info-column">
          <div className="info-card">
            <div className="info-card-title">AI Property Advisory</div>
            <p>
              Fill out the form to let our intelligent model estimate the market value.
              After the prediction, an AI advisor will give you insights into market trends,
              recommendations, and standard risk disclaimers associated with similar properties.
            </p>

            {loading && (
              <div className="summary-card" style={{ marginTop: '1rem', background: 'rgba(255,255,255,0.1)' }}>
                <p>Analyzing parameters and generating advisory report...</p>
              </div>
            )}

            {result && !loading && (
              <div className="summary-card" style={{ marginTop: '1rem', background: '#e6eced', color: 'var(--primary-text)', display: 'block' }}>
                <h4 style={{ color: 'var(--card-bg-dark)', marginBottom: '1rem', fontSize: '1.5rem' }}>
                  ₹ {result.prediction.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </h4>
                <div className="markdown-body" style={{ fontSize: '1rem', lineHeight: '1.6', whiteSpace: 'pre-line', color: 'var(--primary-text)' }}>
                  <ReactMarkdown>{result.advisory.replace(/\*\*/g, '')}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
