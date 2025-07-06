// src/components/AerodynamicsBackground.jsx
import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere } from '@react-three/drei'; // OrbitControls for development, Sphere for example

// A simple animated object
function AnimatedSphere() {
  const meshRef = useRef();

  useFrame(() => {
    if (meshRef.current) {
      // Rotate the sphere
      meshRef.current.rotation.x += 0.005;
      meshRef.current.rotation.y += 0.005;
    }
  });

  return (
    <Sphere args={[1, 32, 32]} ref={meshRef}>
      <meshStandardMaterial color="#00bcd4" wireframe /> {/* Wireframe for an abstract look */}
    </Sphere>
  );
}

function AerodynamicsBackground() {
  return (
    <Canvas
      camera={{ position: [0, 0, 5], fov: 75 }} // Adjust camera position and field of view
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }} // Position behind everything
    >
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      {/* You can replace AnimatedSphere with more complex aerodynamics visuals */}
      <AnimatedSphere />
      {/* <OrbitControls /> // Uncomment for development to move the camera around */}
    </Canvas>
  );
}

export default AerodynamicsBackground;