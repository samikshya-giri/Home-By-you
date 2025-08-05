import React from 'react';
import { Link } from 'react-router-dom';

import './Hero.css';

const Hero = () => {
  return (
    <section className="hero">
      <div className="hero-container">
        <div className="hero-content">
          <h1 className="hero-title">
            <span>WHERE STYLE</span>
            <span>MEETS INTELLIGENCE</span>
          </h1>
          <p className="hero-subtitle">
            Transform your space with AI-curated designs
          </p>
          <div className="hero-buttons">
            <button className="btn-primary">Start Designing</button>
<Link to="/gallery" className="btn-secondary">View Gallery</Link>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;