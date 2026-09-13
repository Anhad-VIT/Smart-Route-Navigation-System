"""Mobile-ready frontend shell for SmartRoute AI."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

st.set_page_config(page_title="SmartRoute Mobile", page_icon="🚗", layout="centered")

st.title("SmartRoute AI")
st.caption("Predictive route guidance for city and long-distance travel")

with st.form("route_form"):
    mode = st.radio("Travel mode", ["within_city", "intercity"], horizontal=True, index=1)
    if mode == "intercity":
        origin = st.text_input("Origin landmark", "Mumbai Central Railway Station", key="mobile_origin")
        destination = st.text_input("Destination landmark", "Pune Junction Railway Station", key="mobile_destination")
        origin_place = st.text_input("Origin search area", "Mumbai, Maharashtra, India", key="mobile_origin_place")
        destination_place = st.text_input("Destination search area", "Pune, Maharashtra, India", key="mobile_destination_place")
        place = ""
    else:
        origin = st.text_input("Origin", "VIT University Main Gate", key="mobile_origin_city")
        destination = st.text_input("Destination", "Katpadi Junction Railway Station", key="mobile_destination_city")
        place = st.text_input("Place / search area", "Vellore, Tamil Nadu, India", key="mobile_place")
        origin_place = ""
        destination_place = ""
    departure = st.datetime_input("Departure time", value=datetime.now().replace(minute=0, second=0, microsecond=0))
    use_ml = st.toggle("Use AI forecast", value=True)
    submitted = st.form_submit_button("Find route")

if submitted:
    try:
        from api_server import build_route_plan

        result = build_route_plan(
            origin=origin,
            destination=destination,
            place=place,
            departure=departure,
            mode=mode,
            use_ml=use_ml,
            origin_place=origin_place,
            destination_place=destination_place,
        )
        if result["status"] != "ok":
            st.error(result.get("message", "Unable to generate route."))
        else:
            primary = result["primary"]
            alternative = result["alternative"]
            st.metric("Predicted traffic", f"{result.get('predicted_volume', 0):,.0f} veh/hr")
            col1, col2 = st.columns(2)
            col1.metric("Primary ETA", f"{primary['travel_minutes']:.1f} min")
            col2.metric("Alternative ETA", f"{alternative['travel_minutes']:.1f} min")
            st.metric("Distance", f"{primary['distance_km']:.1f} km")
            st.caption(f"Route: {' -> '.join(primary['path'])}")

            graph = result.get("graph", {})
            route_points = []
            for sequence, node in enumerate(primary.get("path", [])):
                point = graph.get(node)
                if point is None:
                    continue
                route_points.append({
                    "sequence": sequence,
                    "lat": float(point["lat"]),
                    "lon": float(point["lon"]),
                    "label": node,
                })
            if route_points:
                st.subheader("Map view")
                st.map(route_points, latitude="lat", longitude="lon")
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Route generation failed: {exc}")
