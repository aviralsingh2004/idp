# F1 Aero Dashboard

## Overview
The F1 Aero Dashboard is a React application designed to visualize and analyze aerodynamic predictions and telemetry data for Formula 1 racing. It provides insights into feature importance and allows for comparison of telemetry data against ideal values.

## Project Structure
```
frontend
├── public
│   └── index.html          # Main HTML file serving as the entry point
├── src
│   ├── App.jsx             # Main application component
│   ├── index.js            # Entry point for the React application
│   ├── components          # Contains all React components
│   │   ├── PredictionCard.jsx         # Displays predictions
│   │   ├── FeatureImportanceChart.jsx  # Visualizes feature importance
│   │   └── TelemetryTable.jsx         # Displays telemetry data
│   └── styles              # Contains CSS styles
│       └── App.css        # Styles for the application
├── package.json            # npm configuration file
└── README.md               # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd frontend
   ```
3. Install the dependencies:
   ```
   npm install
   ```

## Usage
To start the application, run:
```
npm start
```
This will launch the application in your default web browser.

## Components
- **PredictionCard**: Displays predictions related to the F1 Aero Dashboard.
- **FeatureImportanceChart**: Visualizes the importance of different features in the model.
- **TelemetryTable**: Displays telemetry data for comparison against ideal values.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.