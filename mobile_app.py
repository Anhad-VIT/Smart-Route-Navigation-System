"""Mobile-ready frontend shell for SmartRoute AI."""

from __future__ import annotations

from datetime import datetime

import streamlit as st
from route_visualization import build_route_deck

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
            route_deck = build_route_deck(graph, [
                {"label": "Recommended route", "path": primary.get("path", []), "color": [239, 131, 84]},
                {"label": "Alternative", "path": alternative.get("path", []), "color": [56, 189, 248]},
            ])
            if route_deck is not None:
                st.subheader("Map view")
                st.pydeck_chart(route_deck, use_container_width=True, height=480)
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Route generation failed: {exc}")
