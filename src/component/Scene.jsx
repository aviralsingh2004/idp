import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { onScoreUpdate } from "../websocket";

const Scene = () => {
  const mountRef = useRef();

  useEffect(() => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 1, 3);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444);
    hemi.position.set(0, 2, 0);
    scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(5, 10, 7.5);
    scene.add(dir);

    let wing = null;
    new GLTFLoader().load("/wing.glb", (gltf) => {
      wing = gltf.scene;
      wing.traverse((c) => c.isMesh && (c.material = c.material.clone()));
      scene.add(wing);
    });

    const getHeatColor = (score) => {
      const hue = (1 - score) * 0.4;
      return new THREE.Color().setHSL(hue, 1, 0.5);
    };

    onScoreUpdate((score) => {
      if (wing) {
        wing.traverse((c) => c.isMesh && c.material.color.copy(getHeatColor(score)));
        wing.scale.set(1, 1 + score * 0.1, 1);
      }
    });

    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      if (mountRef.current && renderer.domElement.parentNode === mountRef.current) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  return <div ref={mountRef} />;
};

export default Scene;
