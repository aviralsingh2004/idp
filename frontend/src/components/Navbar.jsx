import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const navLinks = [
  { to: '/home', label: 'Home' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/simulation', label: 'Simulation Dashboard' }
];

export default function Navbar() {
  const location = useLocation();
  return (
    <nav style={{
      width: '100vw',
      left: 0,
      top: 0,
      background: 'rgba(34, 34, 50, 0.7)',
      color: '#fff',
      padding: '16px 32px',
      display: 'flex',
      alignItems: 'center',
      gap: '32px',
      position: 'sticky',
      zIndex: 1000,
      boxSizing: 'border-box',
      margin: 0,
      height: 60,
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
      boxShadow: '0 4px 24px 0 rgba(0,0,0,0.12)'
    }}>
      {/* Logo/Icon */}
      <div style={{ display: 'flex', alignItems: 'center', marginRight: 32 }}>
        <span style={{ fontSize: 28, fontWeight: 900, color: '#FFEB00', marginRight: 8 }}>🏎️</span>
        <span style={{ fontWeight: 700, fontSize: 20, letterSpacing: 1, color: '#fff' }}>Aero Health</span>
      </div>
      <div style={{ display: 'flex', gap: 24 }}>
        {navLinks.map(link => (
          <Link
            key={link.to}
            to={link.to}
            style={{
              color: location.pathname === link.to ? '#FFEB00' : '#fff',
              textDecoration: 'none',
              fontWeight: 'bold',
              fontSize: '18px',
              position: 'relative',
              transition: 'color 0.2s',
              paddingBottom: 4
            }}
          >
            {link.label}
            {location.pathname === link.to && (
              <span style={{
                position: 'absolute',
                left: 0,
                bottom: 0,
                width: '100%',
                height: 3,
                background: 'linear-gradient(90deg, #FFEB00 0%, #C3002F 100%)',
                borderRadius: 2,
                transition: 'all 0.3s'
              }} />
            )}
          </Link>
        ))}
      </div>
    </nav>
  );
} 