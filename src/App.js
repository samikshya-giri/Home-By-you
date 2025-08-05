import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Gallery from './components/Gallery';
import Scanner from './components/Scanner';
import './App.css';

function App() {
  const [showScanner, setShowScanner] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [formData, setFormData] = useState({
    room_type: '',
    color: '',
    material: '',
    x: '', // Keep as string for input fields
    y: '', // Keep as string for input fields
    z: '', // Keep as string for input fields
    is_seating: false,
    is_table: false,
    is_storage: false,
    loading: false,
    error: null,
    predictionResult: null
  });

  // Premium room templates data (unchanged)
  const premiumTemplates = [
    {
      id: 1,
      name: "Luxury Modern Loft",
      style: "modern",
      colors: ["neutral", "monochrome"],
      image: "https://images.unsplash.com/photo-1583847268964-b28dc8f51f92?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
      likes: 1243
    },
    {
      id: 2,
      name: "Cozy Scandinavian",
      style: "scandinavian",
      colors: ["warm", "neutral"],
      image: "https://images.unsplash.com/photo-1513694203232-719a280e022f?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
      likes: 982
    },
    {
      id: 3,
      name: "Industrial Chic",
      style: "industrial",
      colors: ["cool", "neutral"],
      image: "https://images.unsplash.com/photo-1493809842364-78817add7ffb?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
      likes: 756
    },
    {
      id: 4,
      name: "Minimalist Oasis",
      style: "minimalist",
      colors: ["neutral", "warm"],
      image: "https://images.unsplash.com/photo-1484154218962-a197022b5858?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
      likes: 1120
    },
    {
      id: 5,
      name: "Bohemian Retreat",
      style: "bohemian",
      colors: ["warm", "vibrant"],
      image: "https://images.unsplash.com/photo-1507652313519-d4e9174996dd?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
      likes: 893
    },
    {
      id: 6,
      name: "Mid-Century Classic",
      style: "mid-century",
      colors: ["warm", "neutral"],
      image: "https://images.unsplash.com/photo-1513519245088-0e12902e022f?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
      likes: 678
    }
  ];

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;

    setFormData(prevFormData => ({
      ...prevFormData,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormData(prevFormData => ({ ...prevFormData, loading: true, error: null, predictionResult: null }));

    // Define your Flask API endpoint
    const API_URL = 'http://localhost:5000/predict'; // Make sure this matches your Flask port

    try {
      const payload = {
        room_type: formData.room_type,
        color: formData.color,
        material: formData.material,
        // Convert empty strings to 0 for numerical fields to avoid 'NoneType' errors
        x: formData.x === '' ? 0.0 : parseFloat(formData.x),
        y: formData.y === '' ? 0.0 : parseFloat(formData.y),
        z: formData.z === '' ? 0.0 : parseFloat(formData.z),
        is_seating: formData.is_seating ? 1 : 0,
        is_table: formData.is_table ? 1 : 0,
        is_storage: formData.is_storage ? 1 : 0,
        // Default scale and rotation values if not explicitly in the form
        scale_x: 1.0, // Assuming default or deriving from some other logic if available
        scale_y: 1.0,
        scale_z: 1.0,
        rotation_y: 0.0,
      };

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*' // This is important for CORS, but better configured on Flask
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Something went wrong with the prediction.');
      }

      const result = await response.json();

      setFormData(prevFormData => ({
        ...prevFormData,
        loading: false,
        predictionResult: result.top_predictions // Flask returns 'top_predictions'
      }));
    } catch (error) {
      console.error('Prediction error:', error);
      setFormData(prevFormData => ({
        ...prevFormData,
        loading: false,
        error: error.message || "Failed to get prediction. Please check the server and try again."
      }));
    }
  };

  const filteredTemplates = premiumTemplates.filter(template =>
    template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    template.style.toLowerCase().includes(searchTerm.toLowerCase()) ||
    template.colors.some(color => color.includes(searchTerm.toLowerCase()))
  );

  // Main Content Component (unchanged, except for the handleSubmit call in the button)
  const MainContent = () => (
    <main className="main-content">
      {!showScanner ? (
        <div className="design-container">
          <div className="design-form-container">
            <div className="design-form-header">
              <h2>Design Your Perfect Space</h2>
              <p className="form-subtitle">Craft your dream room with our premium design tools</p>
            </div>

            <div className="form-grid">
              <div className="form-section">
                <h3 className="form-section-title">Basic Information</h3>
                <div className="form-group premium-form-group">
                  <label className="premium-label">Room Type *</label>
                  <select
                    id="room_type"
                    name="room_type"
                    className="premium-select"
                    required
                    aria-required="true"
                    value={formData.room_type}
                    onChange={handleInputChange}
                  >
                    <option value="" disabled>Select room</option>
                    <option value="balcony">Balcony</option>
                    <option value="bathroom">Bathroom</option>
                    <option value="bedroom">Bedroom</option>
                    <option value="classroom">Classroom</option>
                    <option value="diningroom">Dining Room</option>
                    <option value="guestroom">Guest Room</option>
                    <option value="hallway">Hallway</option>
                    <option value="kidsroom">Kids Room</option>
                    <option value="kitchen">Kitchen</option>
                    <option value="livingroom">Living Room</option>
                    <option value="office">Office</option>
                    <option value="studyroom">Study Room</option>
                  </select>
                </div>

                <div className="form-group premium-form-group">
                  <label className="premium-label">Color *</label>
                  <select
                    id="color"
                    name="color"
                    className="premium-select"
                    required
                    aria-required="true"
                    value={formData.color}
                    onChange={handleInputChange}
                  >
                    <option value="" disabled>Select color</option>
                    <option value="beige">Beige</option>
                    <option value="black">Black</option>
                    <option value="blue">Blue</option>
                    <option value="brown">Brown</option>
                    <option value="gold">Gold</option>
                    <option value="gray">Gray</option>
                    <option value="green">Green</option>
                    <option value="orange">Orange</option>
                    <option value="pink">Pink</option>
                    <option value="purple">Purple</option>
                    <option value="red">Red</option>
                    <option value="silver">Silver</option>
                    <option value="turquoise">Turquoise</option>
                    <option value="white">White</option>
                    <option value="yellow">Yellow</option>
                  </select>
                </div>
              </div>

              <div className="form-section">
                <h3 className="form-section-title">Material & Dimensions</h3>
                <div className="form-group premium-form-group">
                  <label className="premium-label">Material *</label>
                  <select
                    id="material"
                    name="material"
                    className="premium-select"
                    required
                    aria-required="true"
                    value={formData.material}
                    onChange={handleInputChange}
                  >
                    <option value="" disabled>Select material</option>
                    <option value="bamboo">Bamboo</option>
                    <option value="ceramic">Ceramic</option>
                    <option value="concrete">Concrete</option>
                    <option value="fabric">Fabric</option>
                    <option value="glass">Glass</option>
                    <option value="leather">Leather</option>
                    <option value="metal">Metal</option>
                    <option value="plastic">Plastic</option>
                    <option value="stone">Stone</option>
                    <option value="wood">Wood</option>
                  </select>
                </div>

                <div className="form-group premium-form-group">
                  <label className="premium-label">Room Dimensions (optional)</label>
                  <div className="dimensions-grid">
                    <div className="dimension-input">
                      <label>Length (X)</label>
                      <input
                        type="number"
                        id="x"
                        name="x"
                        step="0.1"
                        placeholder="e.g., 2.5"
                        value={formData.x}
                        onChange={handleInputChange}
                        className="premium-input"
                      />
                    </div>
                    <div className="dimension-input">
                      <label>Width (Y)</label>
                      <input
                        type="number"
                        id="y"
                        name="y"
                        step="0.1"
                        placeholder="e.g., 0.0"
                        value={formData.y}
                        onChange={handleInputChange}
                        className="premium-input"
                      />
                    </div>
                    <div className="dimension-input">
                      <label>Height (Z)</label>
                      <input
                        type="number"
                        id="z"
                        name="z"
                        step="0.1"
                        placeholder="e.g., 2.5"
                        value={formData.z}
                        onChange={handleInputChange}
                        className="premium-input"
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="form-section">
                <h3 className="form-section-title">Item Properties</h3>
                <div className="form-group premium-form-group">
                  <label className="premium-label">Optional Item Properties</label>
                  <div className="checkbox-group">
                    <label className="checkbox-option">
                      <input
                        type="checkbox"
                        id="is_seating"
                        name="is_seating"
                        checked={formData.is_seating}
                        onChange={handleInputChange}
                      />
                      <span className="checkbox-custom"></span>
                      Is Seating Item?
                    </label>
                    <label className="checkbox-option">
                      <input
                        type="checkbox"
                        id="is_table"
                        name="is_table"
                        checked={formData.is_table}
                        onChange={handleInputChange}
                      />
                      <span className="checkbox-custom"></span>
                      Is Table Item?
                    </label>
                    <label className="checkbox-option">
                      <input
                        type="checkbox"
                        id="is_storage"
                        name="is_storage"
                        checked={formData.is_storage}
                        onChange={handleInputChange}
                      />
                      <span className="checkbox-custom"></span>
                      Is Storage Item?
                    </label>
                  </div>
                </div>
              </div>
            </div>

            <div className="form-actions">
              {/* Changed type to button to prevent default form submission if not desired,
                  and call handleSubmit explicitly */}
              <button
                type="button"
                className="btn premium-button"
                onClick={handleSubmit}
                disabled={formData.loading}
              >
                {formData.loading ? 'Predicting...' : 'Predict Category'}
              </button>
              <button
                onClick={() => setShowScanner(true)}
                className="scan-button premium-button"
              >
                <span>Start 3D Room Scan</span>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" />
                </svg>
              </button>
            </div>

            {formData.loading && (
              <div className="loading" role="alert" aria-live="polite">
                <p>Predicting...</p>
              </div>
            )}

            {formData.error && (
              <div className="error" role="alert" aria-live="assertive">
                {formData.error}
              </div>
            )}

            {formData.predictionResult && (
              <div className="result" role="alert" aria-live="polite">
                <h4>Prediction Results:</h4>
                <ul>
                  {formData.predictionResult.map((item, index) => (
                    <li key={index}>
                      {item.category} ({parseFloat(item.probability * 100).toFixed(1)}
                       confidence)
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="inspiration-section">
  <div className="inspiration-header">
    <h3>Premium Design Templates</h3>
    <div className="search-bar">
      <input
        type="text"
        placeholder="Search templates..."
        value={searchTerm}
        onChange={(e) => {
          e.stopPropagation();
          setSearchTerm(e.target.value);
        }}
        onKeyDown={(e) => e.stopPropagation()}
        className="search-input"
      />
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" strokeWidth="2" />
      </svg>
    </div>
  </div>


            <div className="templates-grid">
              {filteredTemplates.map(template => (
                <div key={template.id} className="template-card">
                  <div className="template-image" style={{ backgroundImage: `url(${template.image})` }}>
                    <div className="template-likes">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M12 21.35L10.55 20.03C5.4 15.36 2 12.28 2 8.5C2 5.42 4.42 3 7.5 3C9.24 3 10.91 3.81 12 5.09C13.09 3.81 14.76 3 16.5 3C19.58 3 22 5.42 22 8.5C22 12.28 18.6 15.36 13.45 20.03L12 21.35Z" fill="currentColor" />
                      </svg>
                      {template.likes.toLocaleString()}
                    </div>
                  </div>
                  <div className="template-info">
                    <h4>{template.name}</h4>
                    <div className="template-tags">
                      <span className="tag">{template.style}</span>
                      {template.colors.map(color => (
                        <span key={color} className="tag">{color}</span>
                      ))}
                    </div>
                    <button
                      className="use-template-btn"
                      onClick={() => {
                        // IMPORTANT: If your Flask backend doesn't recognize "cool", "monochrome", "vibrant",
                        // you will need to map these to colors that it *does* recognize from your
                        // <select id="color"> options. For example:
                        const colorMap = {
                            "cool": "gray", // Or "blue", "white" depending on your model's categories
                            "monochrome": "black", // Or "white", "gray"
                            "vibrant": "red", // Or "orange", "yellow"
                            // Add other mappings if necessary
                        };
                        const selectedColor = template.colors[0];
                        const colorToSend = colorMap[selectedColor] || selectedColor; // Use mapped color or original if no mapping

                        setFormData({
                          ...formData,
                          color: colorToSend
                        });
                      }}
                    >
                      Apply This Style
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <Scanner onBack={() => setShowScanner(false)} />
      )}
    </main>
  );

  // Footer Component (unchanged)
  const Footer = () => (
    <footer className="app-footer">
      <div className="footer-content">
        <div className="footer-logo">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L2 7V17L12 22L22 17V7L12 2Z" stroke="currentColor" strokeWidth="2" />
            <path d="M2 7L12 12" stroke="currentColor" strokeWidth="2" />
            <path d="M12 12L22 7" stroke="currentColor" strokeWidth="2" />
            <path d="M12 12V22" stroke="currentColor" strokeWidth="2" />
          </svg>
          <span>HomeByYou</span>
        </div>
        <div className="footer-links">
          <a href="#">Privacy Policy</a>
          <a href="#">Terms of Service</a>
          <a href="#">Contact Us</a>
        </div>
        <p className="footer-copyright">© {new Date().getFullYear()} HomeByYou. All rights reserved.</p>
      </div>
    </footer>
  );

  return (
    <Router>
      <div className="app">
        <Navbar />

        <Routes>
          <Route path="/" element={
            <>
              <Hero />
              <MainContent />
            </>
          } />
          <Route path="/gallery" element={<Gallery />} />
        </Routes>

        <Footer />
      </div>
    </Router>
  );
}

export default App;