import streamlit as st
import joblib
import numpy as np
import pandas as pd
import heapq
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------
# PAGE CONFIG & CSS
# -------------------------------
st.set_page_config(
    page_title="Traffic Optimizer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
div.stButton > button:first-child {
    background-color: #0068C9;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-weight: 600;
    border: none;
}
div[data-testid="stMetricValue"] {
    color: #0068C9;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("traffic_predicition/traffic_prediction_model.pkl")

try:
    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    poly = bundle["poly"]
    feature_order = bundle["original_features"]
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()

# -------------------------------
# FUNCTIONS
# -------------------------------
def predict_speed(distance_km, extra_features):
    data = extra_features.copy()
    data["distance_km_LOG"] = np.log(distance_km)

    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_order, fill_value=0)

    df_poly = poly.transform(df)
    df_scaled = scaler.transform(df_poly)

    pred_log = model.predict(df_scaled)
    return np.exp(pred_log)[0]

def travel_time(distance, speed):
    return distance / speed

def dijkstra(graph, start, end):
    pq = [(0, start, [])]
    dist = {node: float("inf") for node in graph}
    dist[start] = 0

    while pq:
        current_time, node, path = heapq.heappop(pq)

        if current_time > dist[node]:
            continue

        path = path + [node]

        if node == end:
            return current_time, path

        for neighbor, weight in graph[node]:
            new_dist = current_time + weight

            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor, path))

    return float("inf"), []

def format_time(hours):
    total_minutes = int(round(hours * 60))
    h = total_minutes // 60
    m = total_minutes % 60

    if h == 0:
        return f"{m} min"
    elif m == 0:
        return f"{h} hr"
    return f"{h} hr {m} min"

def get_current_day_time():
    now = datetime.now()

    day = "Weekday" if now.weekday() < 5 else "Weekend"
    hour = now.hour

    if 8 <= hour < 11:
        time_of_day = "Morning Peak"
    elif 11 <= hour < 16:
        time_of_day = "Afternoon"
    elif 16 <= hour < 21:
        time_of_day = "Evening Peak"
    else:
        time_of_day = "Night"

    return day, time_of_day

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("traffic_predicition/delhi_traffic_features.csv")

try:
    df = load_data()
    edges = [(row.start_area, row.end_area, row.distance_km)
             for row in df.itertuples(index=False)]

    locations = sorted(set(df["start_area"]).union(set(df["end_area"])))
    road_types = df["road_type"].unique().tolist()
except FileNotFoundError:
    st.error("Data file not found.")
    st.stop()

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("Settings")

    weather = st.selectbox(
        "Weather",
        ["Clear", "Rain", "Fog", "Heatwave"]
    )

    traffic = st.selectbox(
        "Traffic Density",
        ["Low", "Medium", "High", "Very High"]
    )

    road_type_input = st.selectbox(
        "Road Type",
        road_types
    )

    st.divider()

    day, time_of_day = get_current_day_time()

    st.subheader("Current Context")
    st.info(f"Day: {day}\n\nTime: {time_of_day}")

# -------------------------------
# MAIN UI
# -------------------------------
st.title("🚗 Smart Traffic Route Optimizer")

st.markdown(
    "<p style='font-size:1.1rem;'>Find the fastest route using Machine Learning and Dijkstra's Algorithm.</p>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    start_node = st.selectbox("📍 Start Location", locations)

with col2:
    end_node = st.selectbox("🏁 Destination", locations)

# -------------------------------
# BUTTON ACTION
# -------------------------------
if st.button("🚀 Find Best Route", use_container_width=True):

    extra_features = {
        "day_of_week_Weekday": 1 if day == "Weekday" else 0,
        "day_of_week_Weekend": 1 if day == "Weekend" else 0,
    }

    for w in ["Clear", "Rain", "Fog", "Heatwave"]:
        extra_features[f"weather_condition_{w}"] = 1 if weather == w else 0

    for t in ["Low", "Medium", "High", "Very High"]:
        extra_features[f"traffic_density_level_{t}"] = 1 if traffic == t else 0

    for rt in road_types:
        extra_features[f"road_type_{rt}"] = 1 if road_type_input == rt else 0

    for td in ["Morning Peak", "Afternoon", "Evening Peak", "Night"]:
        extra_features[f"time_of_day_{td}"] = 1 if time_of_day == td else 0

    graph = defaultdict(list)

    for start, end, distance in edges:
        speed = predict_speed(distance, extra_features)
        time = travel_time(distance, speed)

        graph[start].append((end, time))
        graph[end].append((start, time))

    min_time, path = dijkstra(graph, start_node, end_node)

    if path:
        st.success("✅ Fastest route successfully calculated!")

        c1, c2 = st.columns([2, 1])

        with c1:
            st.markdown("### 📍 Suggested Route")
            st.markdown(f"## {' ➔ '.join(path)}")

        with c2:
            st.metric("⏱ Estimated Travel Time", format_time(min_time))

        st.divider()

        route_col1, route_col2 = st.columns([2, 1])

        with route_col1:
            st.markdown("### Route Sequence")

            for i, node in enumerate(path):
                if i < len(path) - 1:
                    st.write(f"{i+1}. {node} ➔")
                else:
                    st.write(f"{i+1}. {node} 🏁")

        with route_col2:
            st.metric("📍 Stops", len(path))
            st.metric("🛣 Segments", max(len(path)-1, 0))

        st.divider()

        st.markdown("## 🌐 Route Graph")

        route_graph = nx.Graph()

        for i in range(len(path) - 1):
            route_graph.add_edge(path[i], path[i + 1])

        fig, ax = plt.subplots(figsize=(10, 4))

        pos = nx.spring_layout(route_graph, seed=42)

        nx.draw(
            route_graph,
            pos,
            with_labels=True,
            node_size=3000,
            font_size=10,
            font_weight="bold",
            width=2,
            ax=ax
        )

        ax.axis("off")
        st.pyplot(fig)

        st.info(
            "Route generated using Dijkstra's Algorithm. "
            "Edge weights are travel times predicted by the ML model."
        )

    else:
        st.error("❌ No route found.")
