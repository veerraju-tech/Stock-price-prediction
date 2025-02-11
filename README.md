# Flask Stock Prediction App

This is a Flask-based web application for stock price prediction using machine learning. It utilizes TensorFlow, pandas, NumPy, scikit-learn, and yFinance for data processing and visualization.

## Features
- Fetch stock data using yFinance
- Preprocess data with MinMaxScaler
- Load and use a trained TensorFlow model for predictions
- Display visualizations using Matplotlib
- Interactive web interface with Flask

## Installation

### 1. Clone the Repository
```sh
git clone <repository-url>
cd <project-directory>
```

### 2. Create a Virtual Environment
```sh
python -m venv venv
```

### 3. Activate the Virtual Environment
- **Windows (cmd/PowerShell)**:
  ```sh
  venv\Scripts\activate
  ```
- **Mac/Linux**:
  ```sh
  source venv/bin/activate
  ```

### 4. Install Dependencies
```sh
pip install -r requirements.txt
```

## Requirements
The required dependencies are listed in `requirements.txt`. They include:
```txt
Flask
pandas
numpy
matplotlib
tensorflow
scikit-learn
yfinance
```

## Running the Application

### 1. Set Flask Environment (Optional)
- **Windows (cmd)**:
  ```sh
  set FLASK_APP=app.py
  ```
- **Mac/Linux**:
  ```sh
  export FLASK_APP=app.py
  ```

### 2. Run Flask Server
```sh
flask run
```

Alternatively, you can run the application directly using:
```sh
python app.py
```

The application will start, and you can access it at `http://127.0.0.1:5000/` in your browser.

## Contributing
Feel free to submit issues or pull requests if you have improvements or bug fixes.

## License
This project is open-source and available under the [MIT License](LICENSE).

