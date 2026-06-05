import joblib
import numpy as np
import pandas as pd
import heapq
from collections import defaultdict

# LOAD TRAINED MODEL
bundle = joblib.load("ml/traffic_predicition/traffic_prediction_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
poly = bundle["poly"]
feature_order = bundle["original_features"]

print("Model Loaded Successfully!\n")

#  SPEED PREDICTION FUNCTION

def predict_speed(distance_km, extra_features):
   
    data = extra_features.copy()
    data["distance_km_LOG"] = np.log(distance_km)

    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_order, fill_value=0)

    df_poly = poly.transform(df)
    df_scaled = scaler.transform(df_poly)

    pred_log = model.predict(df_scaled)
    return np.exp(pred_log)[0]  # convert from log



# 3️⃣ TRAVEL TIME CALCULATION

def travel_time(distance, speed):
    """
    Time = Distance / Speed
    """
    return distance / speed



# 4️⃣ DIJKSTRA ALGORITHM

def dijkstra(graph, start, end):
    pq = [(0, start, [])]
    visited = set()

    while pq:
        current_time, node, path = heapq.heappop(pq)

        if node in visited:
            continue

        path = path + [node]
        visited.add(node)

        if node == end:
            return current_time, path

        for neighbor, weight in graph[node]:
            heapq.heappush(pq, (current_time + weight, neighbor, path))

    return float("inf"), []

# Format: (Start, End, Distance_km)
df=pd.read_csv("ml/traffic_predicition/delhi_traffic_features.csv")
edges =[]
for row in df.itertuples(index=False):
    start = row.start_area
    end = row.end_area
    distance = row.distance_km
    edges.append((start,end,distance))
    
# Example traffic features (static for demo)
extra_features = {
    "day_of_week_Monday": 1,
    "weather_condition_Rain": 1,
    "road_type_Highway": 0,
    "time_of_day_Evening": 0,
    "traffic_density_level_High": 1
}


#  BUILD GRAPH WITH TIME WEIGHTS

graph = defaultdict(list)

for start, end, distance in edges:

    speed = predict_speed(distance, extra_features)
    time = travel_time(distance, speed)

    graph[start].append((end, time))
    graph[end].append((start, time))  # Remove if one-way road

print("Graph Built with Predicted Travel Times.\n")

#  USER INPUT

start_node = input("Enter Start Location: ")
end_node = input("Enter End Location: ")


#  RUN DIJKSTRA

min_time, path = dijkstra(graph, start_node, end_node)

if path:
    print("\nShortest Time Path Found!")
    print("Route:", " → ".join(path))
    print(f"Estimated Travel Time: {min_time:.2f} hours")
else:
    print("No path found.")