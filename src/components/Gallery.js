// src/components/Gallery/Gallery.js
import React, { useState } from 'react';
import './Gallery.css';

const Gallery = () => {
  const [favorites, setFavorites] = useState([]);

  const toggleFavorite = (id) => {
    if (favorites.includes(id)) {
      setFavorites(favorites.filter(favId => favId !== id));
    } else {
      setFavorites([...favorites, id]);
    }
  };

  const designs = [
    { 
      id: 1, 
      title: 'Modern Living Room', 
      img: 'https://images.unsplash.com/photo-1616486338812-3dadae4b4ace?w=800&auto=format&fit=crop&q=80',
      designer: 'Studio Nova'
    },
    { 
      id: 2, 
      title: 'Minimalist Bedroom', 
      img: 'https://images.unsplash.com/photo-1583847268964-b28dc8f51f92?w=800&auto=format&fit=crop&q=80',
      designer: 'Atelier Luxe'
    },
    { 
      id: 3, 
      title: 'Contemporary Kitchen', 
      img: 'https://images.unsplash.com/photo-1600585152220-90363fe7e115?w=800&auto=format&fit=crop&q=80',
      designer: 'Cuisine Moderne'
    },
    { 
      id: 4, 
      title: 'Minimalist Office', 
      img: 'https://images.unsplash.com/photo-1497366811353-6870744d04b2?w=800&auto=format&fit=crop&q=80',
      designer: 'Workform'
    },
    { 
      id: 5, 
      title: 'Luxury Bathroom', 
      img: 'https://images.unsplash.com/photo-1600566752355-35792bedcfea?w=800&auto=format&fit=crop&q=80',
      designer: 'Aqua Designs'
    },
    { 
      id: 6, 
      title: 'Chic Dining Area', 
      img: 'https://images.unsplash.com/photo-1600210492493-0946911123ea?w=800&auto=format&fit=crop&q=80',
      designer: 'Maison Deluxe'
    }
  ];

  return (
    <div className="gallery-container">
      <div className="gallery-header">
        <h1 className="gallery-title">Design Gallery</h1>
        <p className="gallery-subtitle">Curated Collection of Premium Interiors</p>
        <div className="favorites-indicator">
          <span className="heart-icon">❤</span>
          {favorites.length} Favorites
        </div>
      </div>
      <div className="gallery-grid">
        {designs.map(design => (
          <div key={design.id} className="gallery-card">
            <div className="card-image-container">
              <img 
                src={design.img} 
                alt={design.title} 
                className="card-image"
                loading="lazy"
              />
              <div className="card-overlay"></div>
              <button 
                className={`favorite-button ${favorites.includes(design.id) ? 'active' : ''}`}
                onClick={() => toggleFavorite(design.id)}
                aria-label="Add to favorites"
              >
                ❤
              </button>
              <div className="card-content">
                <div>
                  <h3 className="card-title">{design.title}</h3>
                  <p className="card-designer">by {design.designer}</p>
                </div>
                <div className="card-actions">
                  <button className="card-button">
                    {favorites.includes(design.id) ? 'Remove Favorite' : 'Add to Favorites'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Gallery;