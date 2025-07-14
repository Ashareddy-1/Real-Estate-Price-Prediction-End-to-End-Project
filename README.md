🏡 Real Estate Price Prediction – End-to-End ML Project

This project is a complete, production-ready pipeline for predicting house prices in Bengaluru, India, using machine learning. It covers every stage of a real-world ML project — from data preprocessing to model training, and deployment via a client-server architecture.

🚀 Project Structure

├── client/                     # Frontend (can be CLI or Web UI using HTML/JS)
├── model/                      # Trained ML models and utility scripts
├── server/                     # FastAPI or Flask backend serving predictions
├── bengaluru_house_prices.csv  # Cleaned housing dataset
└── README.md                   # Project documentation

📊 Dataset

File: bengaluru_house_prices.csv

Features: Location, square footage, number of bathrooms, BHK, price, etc.

Objective: Predict the price of a house based on features provided.

🧠 Workflow

Data Cleaning & Feature Engineering

Handling missing values, converting categorical data

Creating dummy variables

Outlier removal and normalization

Modeling

Linear Regression as baseline

Model selection via cross-validation

Exporting the trained model using joblib or pickle

API Deployment

Backend built using FastAPI or Flask

Accepts user input via JSON and returns predicted price

Client Interface

Simple client script or frontend to interact with the backend.

▶️ How to Run

1. Clone the Repo
   
git clone https://github.com/your-username/real-estate-price-prediction.git
cd real-estate-price-prediction

2. Run the Server
   
cd server

uvicorn main:app --reload

3. Run the Client (if CLI-based)

cd client

python predict.py
              
If using a web UI, open the HTML file in a browser or connect it to the API endpoint.

🔍 Sample API Call (JSON)
{
  "location": "Indira Nagar",
  "total_sqft": 1200,
  "bath": 2,
  "bhk": 3
}
Response:

{
  "estimated_price": 125.0
}

✨ Highlights

✅ Clean, modular code structure

✅ Production-ready API with FastAPI/Flask

✅ Fully reproducible with dataset included

✅ Easy to expand or convert to web app

🧰 Technologies Used

Python, Pandas, NumPy, Scikit-learn

FastAPI / Flask

Joblib / Pickle

HTML/CSS/JS (for optional frontend)

📌 Future Enhancements

📈 Model comparison and ensemble learning

📊 Interactive visualizations (Plotly, Dash)

☁️ Deployment on cloud platforms like AWS, Azure, or Streamlit Cloud

"Prediction is not magic — it's clean data, strong logic, and smart deployment."


