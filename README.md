#  Machine Learning-Based Traffic Route Optimization System

## Overview

The Machine Learning-Based Traffic Route Optimization System is an intelligent route planning application that combines Machine Learning and Graph Algorithms to determine the fastest route between two locations under varying traffic conditions.
Unlike traditional shortest-path systems that rely only on distance, this project predicts the average speed of travel on road segments using a trained Machine Learning model. The predicted speeds are then converted into travel times, which are used by Dijkstra's Algorithm to identify the route with the minimum estimated travel time.
The system provides a user-friendly Streamlit interface where users can select source and destination locations and simulate different traffic, weather, and road conditions.

---

# Problem Statement

Travel time is affected by multiple factors such as traffic density, weather conditions, road type, and time of day. Routes with the shortest distance are not always the fastest.
The objective of this project is to use Machine Learning to predict traffic speeds under different conditions and use those predictions to compute the fastest route rather than simply the shortest route.

---

<img width="1923" height="1078" alt="Screenshot 2026-06-05 190024" src="https://github.com/user-attachments/assets/e5c796b5-ccf1-4ed6-802d-637c176f1caa" />

---

# How the System Works

The project operates in two major phases:

## Phase 1: Machine Learning Model Training

### Step 1: Data Collection

The dataset contains information about:

* Start Area
* End Area
* Distance (km)
* Day of Week
* Weather Condition
* Road Type
* Time of Day
* Traffic Density
* Average Speed (Target Variable)

---

### Step 2: Data Preprocessing

The dataset undergoes several preprocessing steps:

* Removal of unnecessary columns
* One-Hot Encoding of categorical features
* Duplicate removal
* Missing value handling
* Outlier removal using the IQR method
* Log transformation of speed and distance

These steps improve model performance and reduce noise in the data.

---

### Step 3: Feature Engineering

Polynomial Features are generated to capture nonlinear relationships between traffic conditions and vehicle speed.

The features are then standardized using StandardScaler.

---

### Step 4: Model Training

The project uses:

**Ridge Regression**

Ridge Regression was selected because it:

* Handles multicollinearity effectively
* Reduces overfitting
* Produces stable predictions
* Performs well on structured tabular data

The model learns the relationship between traffic conditions and vehicle speed.

---

### Step 5: Speed Prediction

The trained model predicts:

**Average Vehicle Speed (km/h)**

based on:

* Distance
* Traffic Density
* Weather Condition
* Road Type
* Day of Week
* Time of Day

---

# Phase 2: Route Optimization

After the model is trained, the route optimization process begins.

## Step 1: User Input

The user selects:

* Start Location
* Destination
* Weather Condition
* Traffic Density
* Road Type

The current day and time are automatically detected.

---

## Step 2: Travel Time Calculation

For every road segment:

The Machine Learning model predicts:

Predicted Speed

Travel time is then calculated using:

Travel Time = Distance / Predicted Speed

The calculated travel time becomes the weight of that road segment.

---

## Step 3: Graph Construction

Each location is represented as a node.

Each road segment is represented as an edge.

Example:

Delhi_A ---- Delhi_B ---- Delhi_C

The weight of every edge is the predicted travel time rather than physical distance.

---

## Step 4: Dijkstra's Algorithm

Dijkstra's Algorithm is applied to the graph.

The algorithm searches for the route with the minimum cumulative travel time.

Instead of finding the shortest distance, it finds the fastest route based on predicted traffic conditions.

---

## Step 5: Route Recommendation

The system displays:

* Optimal Route
* Estimated Travel Time
* Route Statistics
* Route Visualization

---

# System Workflow

Traffic Dataset

↓

Data Preprocessing

↓

Feature Engineering

↓

Ridge Regression Model

↓

Speed Prediction

↓

Travel Time Estimation

↓

Graph Construction

↓

Dijkstra's Algorithm

↓

Optimal Route

↓

User Interface (Streamlit)

---

# Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-Learn
* Ridge Regression
* Polynomial Features
* StandardScaler

### Data Processing

* Pandas
* NumPy

### Route Optimization

* NetworkX
* Dijkstra's Algorithm

### Visualization

* Matplotlib

### Web Application

* Streamlit

---

# Key Features

* Traffic-aware route planning
* Machine Learning-based speed prediction
* Fastest route computation
* Interactive web interface
* Dynamic travel time estimation
* Real-time condition simulation
* Graph-based route optimization

---

**#Project Structure**
```
traffic-route-optimizer/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│── delhi_traffic_features.csv
│── traffic_prediction_model.pkl
├── delhi_traffic_weights.csv
├── model_intercept.txt
└── screenshots/
```

# Future Enhancements

* Real-time traffic API integration
* Live weather updates
* GPS navigation support
* Multi-city deployment
* Deep Learning-based traffic prediction
* Real-time route recalculation
* Mobile application support

---

# Conclusion

This project demonstrates how Machine Learning and Graph Algorithms can be combined to create an intelligent traffic route optimization system. By predicting traffic speeds under different conditions and using those predictions within Dijkstra's Algorithm, the system identifies the fastest route between locations, providing a practical example of AI-driven transportation optimization.
