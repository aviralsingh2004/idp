import Scene from './viewer/Scene';

function App() {
  return (
    <>
      <div style={{
        position: 'absolute', top: 10, left: 10, color: 'white',
        background: 'rgba(0,0,0,0.6)', padding: '10px', zIndex: 1
      }}>
        Aero Health Monitor
      </div>
      <Scene />
    </>
  );
}

export default App;
