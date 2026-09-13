"""Lightweight API layer for the SmartRoute AI route engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from traffic_analysis import (
    build_demo_graph,
    geocode_landmark,
    load_osm_graph,
    nearest_node,
    predict_volume,
    recommend_routes,
    route_intercity_osrm,
    load_traffic_data,
)

app = FastAPI(title="SmartRoute AI API", version="1.0.0")


class RouteRequest(BaseModel):
    origin: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)
    place: str = Field(default="Vellore, Tamil Nadu, India")
    origin_place: str = Field(default="")
    destination_place: str = Field(default="")
    departure: str | None = None
    mode: str = "within_city"
    weather_penalty: float = 0.0
    incident_penalty: float = 0.0
    use_ml: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/route")
def route(payload: RouteRequest) -> dict[str, Any]:
    departure = pd.Timestamp(payload.departure) if payload.departure else pd.Timestamp(datetime.now())
    mode = payload.mode.lower()
    try:
        if mode == "intercity":
            origin_search = (payload.origin_place or payload.place).strip() or "Mumbai, Maharashtra, India"
            destination_search = (payload.destination_place or payload.place).strip() or "Pune, Maharashtra, India"
            origin_coordinates = geocode_landmark(payload.origin, origin_search)
            destination_coordinates = geocode_landmark(payload.destination, destination_search)
            frame = load_traffic_data()
            volume = predict_volume(frame, departure, use_ml=payload.use_ml)
            result = route_intercity_osrm(
                origin_coordinates,
                destination_coordinates,
                volume,
                payload.weather_penalty,
                payload.incident_penalty,
                departure=departure,
                frame=frame,
            )
            result["predicted_volume"] = volume
            result["forecast_metrics"] = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
            return {"status": "ok", "travel_mode": "intercity", **result}

        graph = build_demo_graph()
        try:
            graph = load_osm_graph(payload.place)
            origin_lat, origin_lon = geocode_landmark(payload.origin, payload.place)
            destination_lat, destination_lon = geocode_landmark(payload.destination, payload.place)
            start_node = nearest_node(graph, origin_lat, origin_lon)
            goal_node = nearest_node(graph, destination_lat, destination_lon)
            result = recommend_routes(departure, payload.weather_penalty, payload.incident_penalty, graph, payload.use_ml, start_node, goal_node)
        except Exception:
            result = recommend_routes(departure, payload.weather_penalty, payload.incident_penalty, graph, payload.use_ml)
        result["travel_mode"] = "within_city"
        return {"status": "ok", **result}
    except Exception as error:
        return {"status": "error", "message": str(error)}


def build_route_plan(
    origin: str,
    destination: str,
    place: str,
    departure: datetime,
    mode: str = "within_city",
    weather_penalty: float = 0.0,
    incident_penalty: float = 0.0,
    use_ml: bool = True,
    origin_place: str | None = None,
    destination_place: str | None = None,
) -> dict[str, Any]:
    request = RouteRequest(
        origin=origin,
        destination=destination,
        place=place,
        origin_place=origin_place or "",
        destination_place=destination_place or "",
        departure=departure.isoformat(),
        mode=mode,
        weather_penalty=weather_penalty,
        incident_penalty=incident_penalty,
        use_ml=use_ml,
    )
    return route(request)
