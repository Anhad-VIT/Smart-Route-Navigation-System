"""Streamlit dashboard for the smart car navigation prototype."""

from datetime import datetime

import pandas as pd
import streamlit as st

from traffic_analysis import build_demo_graph, geocode_landmark, load_osm_graph, load_traffic_data, nearest_node, predict_volume, recommend_routes, route_intercity_osrm
from route_visualization import build_route_deck


st.set_page_config(page_title="SmartRoute AI", page_icon="S", layout="wide")


@st.cache_data(ttl=86_400, max_entries=4)
def cached_osm_graph(place: str):
    return load_osm_graph(place)


@st.cache_data(ttl=86_400, max_entries=128)
def cached_geocode(landmark: str, place: str):
    return geocode_landmark(landmark, place)


@st.cache_data(ttl=300, max_entries=32)
def cached_intercity_route(origin_coordinates, destination_coordinates, volume, weather, incident, departure):
    frame = load_traffic_data()
    return route_intercity_osrm(
        origin_coordinates,
        destination_coordinates,
        volume,
        weather,
        incident,
        departure=departure,
        frame=frame,
    )


st.title("SmartRoute AI")
st.caption("Predictive traffic intelligence and congestion-aware route recommendations")

with st.sidebar:
    st.header("Trip settings")
    departure = st.datetime_input("Departure", value=datetime.now().replace(minute=0, second=0, microsecond=0))
    weather_penalty = st.slider("Weather impact (minutes / edge)", 0.0, 8.0, 0.0, 0.5)
    incident_penalty = st.slider("Incident impact (minutes / edge)", 0.0, 15.0, 0.0, 0.5)
    use_ml = st.toggle("Use XGBoost when installed", value=True)
    travel_mode = st.radio("Travel mode", ["Within city", "Intercity / long travel"], index=0)
    use_live_map = travel_mode == "Within city" and st.toggle("Use live OpenStreetMap graph", value=True)
    if travel_mode == "Intercity / long travel":
        origin_place = st.text_input("Origin search area", "Mumbai, Maharashtra, India").strip()
        destination_place = st.text_input("Destination search area", "Pune, Maharashtra, India").strip()
        origin = st.text_input("Origin landmark", "Mumbai Central Railway Station").strip()
        destination = st.text_input("Destination landmark", "Pune Junction Railway Station").strip()
        place = ""
    else:
        place = st.text_input("Search area", "Vellore, Tamil Nadu, India").strip()
        origin = st.text_input("Origin landmark", "VIT University Main Gate").strip()
        destination = st.text_input("Destination landmark", "Katpadi Junction Railway Station").strip()
        origin_place = destination_place = place
    st.caption("Landmarks are geocoded through OpenStreetMap. Long travel uses OSRM so the entire highway corridor is covered.")

graph = build_demo_graph()
if travel_mode == "Intercity / long travel":
    try:
        origin_coordinates = cached_geocode(origin, origin_place)
        destination_coordinates = cached_geocode(destination, destination_place)
        frame = load_traffic_data()
        volume = predict_volume(frame, pd.Timestamp(departure), use_ml=use_ml)
        result = cached_intercity_route(
            origin_coordinates,
            destination_coordinates,
            volume,
            weather_penalty,
            incident_penalty,
            departure,
        )
        result["predicted_volume"] = volume
        result["forecast_metrics"] = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
        st.success("Intercity route loaded through OSRM")
    except Exception as error:
        st.error(f"Intercity route unavailable: {error}")
        st.stop()
elif use_live_map:
    try:
        if not place:
            raise ValueError("OSM place is empty. Enter a city or region name.")
        graph = cached_osm_graph(place)
        origin_lat, origin_lon = cached_geocode(origin, place)
        destination_lat, destination_lon = cached_geocode(destination, place)
        start_node = nearest_node(graph, origin_lat, origin_lon)
        goal_node = nearest_node(graph, destination_lat, destination_lon)
        result = recommend_routes(pd.Timestamp(departure), weather_penalty, incident_penalty, graph, use_ml, start_node, goal_node)
        st.success(f"Live OSM graph loaded: {len(graph):,} road nodes")
    except Exception as error:
        st.warning(f"Live graph unavailable ({error}). Showing offline demo graph instead.")
        result = recommend_routes(pd.Timestamp(departure), weather_penalty, incident_penalty, graph, use_ml)
else:
    result = recommend_routes(pd.Timestamp(departure), weather_penalty, incident_penalty, graph, use_ml)
primary = result["primary"]
alternative = result["alternative"]

metric_columns = st.columns(4)
metric_columns[0].metric("Predicted traffic", f"{result['predicted_volume']:,.0f} veh/hr")
metric_columns[1].metric("Primary ETA", f"{primary['travel_minutes']:.1f} min")
metric_columns[2].metric("Alternative ETA", f"{alternative['travel_minutes']:.1f} min")
metric_columns[3].metric("Congestion index", f"{result['primary']['congestion_ratio']:.2f}")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Recommended routes")
    primary_label = "Primary A*" if travel_mode == "Within city" else "Primary OSRM"
    route_table = pd.DataFrame(
        [
            {"Route": primary_label, "Stops": " -> ".join(primary["path"]), "Distance (km)": primary["distance_km"], "ETA (min)": primary["travel_minutes"]},
            {"Route": "Alternative", "Stops": " -> ".join(alternative["path"]), "Distance (km)": alternative["distance_km"], "ETA (min)": alternative["travel_minutes"]},
        ]
    )
    st.dataframe(route_table, hide_index=True, use_container_width=True)

    st.subheader("Route corridor")
    graph = result["graph"]
    evolution = result.get("route_evolution", {})
    reroute = evolution.get("reroute")
    if reroute:
        st.info(reroute["reason"])
        route_deck = build_route_deck(graph, [
            {"label": "Previous route", "path": primary["path"], "color": [148, 163, 184]},
            {"label": "Recommended reroute", "path": reroute["path"], "color": [239, 131, 84]},
        ])
    else:
        route_deck = build_route_deck(graph, [
            {"label": primary_label, "path": primary["path"], "color": [239, 131, 84]},
            {"label": "Alternative", "path": alternative["path"], "color": [56, 189, 248]},
        ])
    if route_deck is not None:
        st.pydeck_chart(route_deck, use_container_width=True, height=480)

    checkpoints = evolution.get("checkpoints", [])
    if checkpoints:
        st.subheader("Traffic checkpoints")
        for checkpoint in checkpoints:
            label = "High congestion" if checkpoint["status"] == "high" else "Stable traffic"
            st.caption(f"{label} at {checkpoint['hour_label']} — {checkpoint['reason']}")

with right:
    st.subheader("Forecast quality")
    metrics = pd.DataFrame([result["forecast_metrics"]]).round(2)
    if travel_mode == "Intercity / long travel":
        st.caption("Forecast metrics are not applicable to OSRM's external route service.")
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    st.subheader("Hourly traffic profile")
    data = load_traffic_data()
    profile = data.groupby("hour", as_index=True)["traffic_volume"].mean()
    st.line_chart(profile)
    st.info("Within-city mode uses admissible A*. Intercity mode uses OSRM to cover long highway corridors; the historical traffic model supplies the congestion context.")