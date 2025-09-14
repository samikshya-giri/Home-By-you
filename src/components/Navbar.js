// src/components/Navbar.js
import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
    const [scrolled, setScrolled] = useState(false);
    const [activeLink, setActiveLink] = useState('/');
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const location = useLocation();

    useEffect(() => {
        setActiveLink(location.pathname);
    }, [location]);

    useEffect(() => {
        const handleScroll = () => {
            if (window.scrollY > 10) {
                setScrolled(true);
            } else {
                setScrolled(false);
            }
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const toggleMobileMenu = () => {
        setIsMobileMenuOpen(!isMobileMenuOpen);
    };

    const closeMobileMenu = () => {
        setIsMobileMenuOpen(false);
    };

    const handleGetStartedClick = (e) => {
        e.preventDefault();
        closeMobileMenu();
        const formSection = document.getElementById('room-form');
        if (formSection) {
            const yOffset = -80;
            const y = formSection.getBoundingClientRect().top + window.pageYOffset + yOffset;
            window.scrollTo({ top: y, behavior: 'smooth' });
        }
    };

    return (
        <nav className={`navbar ${scrolled ? 'scrolled' : ''}`}>
            <div className="navbar-container">
                <div className="navbar-logo">
                    <Link to="/" className="logo-link" onClick={closeMobileMenu}>
                        <div className="logo-icon">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 2L2 7V17L12 22L22 17V7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                <path d="M2 7L12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                <path d="M12 12L22 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                <path d="M12 12V22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                        </div>
                        <span className="logo-text">HomeByYou</span>
                    </Link>
                </div>

                {/* Mobile menu button */}
                <button 
                    className={`mobile-menu-button ${isMobileMenuOpen ? 'active' : ''}`}
                    onClick={toggleMobileMenu}
                    aria-label="Toggle menu"
                >
                    <span></span>
                    <span></span>
                    <span></span>
                </button>

                <div className={`navbar-links ${isMobileMenuOpen ? 'active' : ''}`}>
                    <Link
                        to="/gallery"
                        className={`nav-link ${activeLink === '/gallery' ? 'active' : ''}`}
                        onClick={closeMobileMenu}
                    >
                        <span className="nav-link-text">Gallery</span>
                        <div className="nav-link-underline"></div>
                    </Link>
                    <Link
                        to="/scan"
                        className={`nav-link ${activeLink === '/scan' ? 'active' : ''}`}
                        onClick={closeMobileMenu}
                    >
                        <span className="nav-link-text">Scan Room</span>
                        <div className="nav-link-underline"></div>
                    </Link>

                    <Link
                        to="#"
                        className="generate-link"
                        onClick={handleGetStartedClick}
                    >
                        <span className="generate-text">Get Started</span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                    </Link>
                    
                    {/* Admin Dashboard Link */}
                    <Link to="/admin" className="nav-icon-link" onClick={closeMobileMenu}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            <path d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            <path d="M12 22C14.6667 22 16.6667 20.6667 18 19C16 17 14 17 12 17C10 17 8 17 6 19C7.33333 20.6667 9.33333 22 12 22Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                    </Link>
                </div>

                {/* Mobile menu backdrop */}
                <div 
                    className={`mobile-menu-backdrop ${isMobileMenuOpen ? 'active' : ''}`}
                    onClick={closeMobileMenu}
                ></div>
            </div>
        </nav>
    );
};

export default Navbar;