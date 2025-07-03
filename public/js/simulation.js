// public/js/simulation.js
// import * as THREE         from 'https://unpkg.com/three@0.152.0/build/three.module.js';
// import { GLTFLoader }     from 'https://unpkg.com/three@0.152.0/examples/jsm/loaders/GLTFLoader.js';
// import { latestScore }    from './websocket-client.js';

let scene, camera, renderer, wing;

init();
animate();

function init() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, 100);
  camera.position.set(0, 1, 3);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(innerWidth, innerHeight);
  document.body.appendChild(renderer.domElement);

  // Lights
  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444);
  hemi.position.set(0, 2, 0);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 10, 7.5);
  scene.add(dir);

  // Load wing mesh
  new GLTFLoader().load('/models/wing.glb', gltf => {
    wing = gltf.scene;
    // Clone materials so color changes won’t affect original
    wing.traverse(c => c.isMesh && (c.material = c.material.clone()));
    scene.add(wing);
  });

  window.addEventListener('resize', onResize);
}

function onResize() {
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}

// Map score [0→1] → hue [green→red]
function getHeatColor(score) {
  const hue = (1 - score) * 0.4; // 0.4=green, 0=red
  return new THREE.Color().setHSL(hue, 1, 0.5);
}

function animate() {
  requestAnimationFrame(animate);
  if (wing) {
    // 1) Heat-map color
    wing.traverse(c => {
      if (c.isMesh) c.material.color.copy(getHeatColor(latestScore));
    });
    // 2) Simple deformation: slight bend on Y
    const bend = 1 + latestScore * 0.1;
    wing.scale.set(1, bend, 1);
  }
  renderer.render(scene, camera);
}
