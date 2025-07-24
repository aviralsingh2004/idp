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
      background: 'var(--bg-card)',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      color: 'var(--text-primary)',
      padding: 'var(--spacing-md) var(--spacing-xl)',
      display: 'flex',
      alignItems: 'center',
      gap: 'var(--spacing-xl)',
      position: 'fixed',
      zIndex: 1000,
      boxSizing: 'border-box',
      margin: 0,
      height: 'var(--spacing-3xl)',
      border: 'none',
      borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
      boxShadow: 'var(--shadow-md)'
    }}>
      {/* Logo/Icon */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        marginRight: 'var(--spacing-xl)' 
      }}>
        <span style={{ 
          fontSize: '1.75rem', 
          fontWeight: 900, 
          color: 'var(--color-accent)', 
          marginRight: 'var(--spacing-sm)' 
        }}>🏎️</span>
        <span style={{ 
          fontWeight: 700, 
          fontSize: '1.25rem', 
          letterSpacing: '0.05em', 
          color: 'var(--text-primary)' 
        }}>Aero Health</span>
      </div>
      <div style={{ display: 'flex', gap: 'var(--spacing-lg)' }}>
        {navLinks.map(link => (
          <Link
            key={link.to}
            to={link.to}
            style={{
              color: location.pathname === link.to ? 'var(--color-accent)' : 'var(--text-secondary)',
              textDecoration: 'none',
              fontWeight: 600,
              fontSize: '1rem',
              position: 'relative',
              transition: 'all 0.2s ease',
              padding: 'var(--spacing-sm) var(--spacing-md)',
              borderRadius: 'var(--border-radius-sm)',
              background: location.pathname === link.to ? 'rgba(255, 235, 59, 0.1)' : 'transparent'
            }}
          >
            {link.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
