# Aero Health IDP – F1 Telemetry & Aerodynamics Simulator

## Overview

Aero Health IDP is a full-stack application for simulating, visualizing, and analyzing Formula 1 car telemetry and aerodynamic data. It enables users to replay real race laps, view live car metrics, and receive AI-powered aerodynamic analysis and suggestions.

## Features

- **Interactive Simulation Dashboard**: Select year, Grand Prix, session, driver, and lap to replay real F1 telemetry.
- **Live Metrics**: Visualizes lap time, wing angle, and body flex in real time.
- **Track Animation**: Animated car movement on a normalized track map.
- **Telemetry Charts**: Plots key metrics (wing angle, body flex) over time.
- **AI-Powered Analysis**: Backend uses a large language model (LLM) to generate expert-level aerodynamic summaries.
- **Feature Importance**: Exposes model feature importances for transparency.
- **Modern UI**: Built with React, Framer Motion, and Recharts for a smooth, interactive experience.

## Architecture

### Frontend (`frontend/`)

- **React + Vite**: Fast, modern SPA.
- **Key Libraries**:  
  - `recharts` for data visualization  
  - `framer-motion` for animation  
  - `@react-three/fiber` and `three` for 3D/track rendering  
  - `axios` for API calls  
  - `tailwindcss` and `styled-components` for styling
- **Main Component**: `SimulationDashboard.jsx` – handles simulation logic, UI, and data visualization.
- **API Integration**: Fetches telemetry, track positions, and analysis from the backend.

### Backend (`backend/`)

- **Flask API**: Serves telemetry, track, and prediction endpoints.
- **Key Endpoints**:
  - `/api/predict`: Returns aerodynamic predictions and LLM-generated analysis.
  - `/api/feature-importance`: Returns model feature importances.
  - `/api/track-positions`: Returns car position and speed data for a lap.
  - `/api/raw-telemetry`: Returns raw telemetry for a lap.
- **ML/AI Integration**:
  - Uses pre-trained scikit-learn and PyTorch models for aerodynamic predictions.
  - Integrates with a large language model (Groq Llama3) for expert analysis.
- **Data Sources**: Uses `fastf1` to fetch real F1 telemetry data.

### Shared/Core (`src/`)

- May contain shared components or logic (expand as needed).

## Installation

### Backend

1. `cd backend`
2. Install dependencies:  
   `pip install -r requirement.txt`
3. Place pre-trained models in `backend/model/` (see `utils.py` for expected files).
4. Set your `GROQ_API_KEY` in a `.env` file.
5. Run the server:  
   `python app.py`

### Frontend

1. `cd frontend`
2. Install dependencies:  
   `npm install`
3. Start the dev server:  
   `npm run dev`

## Usage

- Open the frontend in your browser.
- Select race parameters and start the simulation.
- View live metrics, track animation, and charts.
- Pause, resume, or reset as needed.
- Review AI-generated aerodynamic analysis.

## Technologies

- **Frontend**: React, Vite, Recharts, Framer Motion, Three.js, TailwindCSS
- **Backend**: Flask, FastF1, scikit-learn, PyTorch, Groq LLM API
- **ML Models**: Pre-trained models for aerodynamic prediction and feature importance

## Folder Structure

```
idp/
  backend/    # Flask API, ML models, utilities
  frontend/   # React app, UI components
  src/        # (Optional) Shared logic/components
```

## Notes

- You need real F1 telemetry data and pre-trained models for full functionality.
- The backend requires a valid Groq API key for LLM-powered analysis.
