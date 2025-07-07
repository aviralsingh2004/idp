import React from 'react';
import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();
  return (
    <nav style={{
      width: '100vw',
      left: 0,
      top: 0,
      background: '#222',
      color: '#fff',
      padding: '12px 24px',
      display: 'flex',
      alignItems: 'center',
      gap: '24px',
      position: 'fixed',
      zIndex: 1000,
      boxSizing: 'border-box',
      margin: 0
    }}>
      <Link to="/home" style={{
        color: location.pathname === '/home' ? '#90caf9' : '#fff',
        textDecoration: 'none',
        fontWeight: 'bold',
        fontSize: '18px'
      }}>Home</Link>
      <Link to="/" style={{
        color: location.pathname === '/' ? '#90caf9' : '#fff',
        textDecoration: 'none',
        fontWeight: 'bold',
        fontSize: '18px'
      }}>Dashboard</Link>
      <Link to="/simulation" style={{
        color: location.pathname === '/simulation' ? '#90caf9' : '#fff',
        textDecoration: 'none',
        fontWeight: 'bold',
        fontSize: '18px'
      }}>Simulation Dashboard</Link>
    </nav>
  );
} 