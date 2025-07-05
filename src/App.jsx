import Scene from "./component/Scene";
import "./App.css";

function App() {
  return (
    <div>
      <h2 style={{ position: "absolute", color: "white", margin: 10, zIndex: 10 }}>
        Aero Health - Real-Time Anomaly
      </h2>
      <Scene />
    </div>
  );
}

export default App;
