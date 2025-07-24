import React, { useEffect, useRef, useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, useGLTF } from '@react-three/drei';
import axios from 'axios';
import * as THREE from 'three';

function CarModel({ position, rotation }) {
  const { scene } = useGLTF('/CAR Model.glb');
  return (
    <primitive
      object={scene}
      position={position}
      scale={[0.1, 0.1, 0.1]}
      rotation={rotation}
      castShadow
    />
  );
}

function lerpColor(a, b, t) {
  // a, b: [r,g,b], t: 0-1
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t
  ];
}

function rgbToHex([r, g, b]) {
  return (
    '#' +
    [r, g, b]
      .map(x => {
        const hex = Math.round(x).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
      })
      .join('')
  );
}

function Track({ points }) {
  if (points.length < 2) return null;
  // Render as a wide tube (asphalt)
  const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(p.X, 0.5, p.Y)));
  const geometry = new THREE.TubeGeometry(curve, 500, 10, 32, false); // wider tube
  return (
    <mesh geometry={geometry} receiveShadow castShadow>
      <meshStandardMaterial color="#222" roughness={0.7} metalness={0.2} />
    </mesh>
  );
}

function TrackScene({ year, gp, session, lap, syncIndex }) {
  const [trackPoints, setTrackPoints] = useState([]);
  const [carPos, setCarPos] = useState([0, 0, 0]);
  const [carRot, setCarRot] = useState([0, Math.PI, 0]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    axios.get('http://localhost:5000/api/track-positions', {
      params: { year, gp, session, driver: 'VER', lap }
    }).then(res => {
      const raw = res.data;
      if (!raw || raw.length === 0) return;
      const xs = raw.map(p => p.X);
      const ys = raw.map(p => p.Y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const norm = raw.map(p => ({
        X: ((p.X - minX) / (maxX - minX)) * 500 - 250,
        Y: ((p.Y - minY) / (maxY - minY)) * 500 - 250,
        Speed: p.Speed
      }));
      setTrackPoints(norm);
      setLoading(false);
    });
  }, [year, gp, session, lap]);

  useEffect(() => {
    if (trackPoints.length > 0 && syncIndex < trackPoints.length) {
      const curr = trackPoints[syncIndex];
      setCarPos([curr.X, 5, curr.Y]);
      // Directional rotation
      let nextIdx = Math.min(syncIndex + 1, trackPoints.length - 1);
      if (nextIdx === syncIndex && syncIndex > 0) nextIdx = syncIndex - 1;
      const next = trackPoints[nextIdx];
      const dx = next.X - curr.X;
      const dz = next.Y - curr.Y;
      const angle = Math.atan2(dx, dz); // Y-axis rotation
      setCarRot([0, angle, 0]);
    }
  }, [syncIndex, trackPoints]);

  // For camera framing
  const center = [0, 50, 0];

  return (
    <Canvas shadows style={{ width: 600, height: 400, background: '#e0e7ef', borderRadius: 12 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[100, 200, 100]} intensity={1} castShadow />
      <PerspectiveCamera makeDefault position={[0, 300, 400]} fov={60} />
      <OrbitControls target={center} enablePan enableZoom enableRotate />
      {/* Grass plane */}
      <mesh receiveShadow position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[1200, 1200]} />
        <meshStandardMaterial color="#2e7d32" />
      </mesh>
      {/* Track */}
      {!loading && <Track points={trackPoints} />}
      {/* Car */}
      {!loading && (
        <Suspense fallback={null}>
          <CarModel position={carPos} rotation={carRot} />
        </Suspense>
      )}
    </Canvas>
  );
}

export default function Track3D(props) {
  return <TrackScene {...props} />;
}

useGLTF.preload('/CAR Model.glb'); 