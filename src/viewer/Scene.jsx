// src/viewer/Scene.jsx
import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { useWebSocket } from './websocket';

export default function Scene() {
  const mountRef = useRef();
  const anomalyScore = useWebSocket();

  useEffect(() => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 1, 3);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444);
    scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    scene.add(dir);

    const loader = new GLTFLoader();
    let model;

    loader.load('/wing.glb', gltf => {
      model = gltf.scene;
      model.traverse(c => {
        if (c.isMesh) c.material = c.material.clone();
      });
      scene.add(model);
    });

    const animate = () => {
      requestAnimationFrame(animate);
      if (model) {
        const color = new THREE.Color().setHSL((1 - anomalyScore) * 0.4, 1, 0.5);
        model.traverse(c => c.isMesh && c.material.color.copy(color));
        model.scale.set(1, 1 + anomalyScore * 0.1, 1);
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
        if (mountRef.current && renderer.domElement) {
          mountRef.current.removeChild(renderer.domElement);
        }
      };
  }, [anomalyScore]);

  return <div ref={mountRef} />;
}
