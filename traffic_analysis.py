"""Core forecasting and routing engine for the smart navigation prototype."""

from __future__ import annotations

import heapq
import json
import math
from functools import lru_cache
from threading import Lock
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

try:
	from xgboost import XGBRegressor
except ImportError:
	XGBRegressor = None


DATA_PATH = Path(__file__).parent / "data" / "Metro_Interstate_Traffic_Volume.csv"
VOLUME_CAPACITY = 7_000.0
GEOCODING_REQUEST_TIMEOUT_SECONDS = 20.0
_xgb_model_cache: dict[int, tuple[pd.DataFrame, Any]] = {}
_osmnx_request_lock = Lock()


@lru_cache(maxsize=4)
def _load_traffic_data_cached(path: str, modified_ns: int, file_size: int) -> pd.DataFrame:
	"""Load the UCI records and add cyclic temporal features."""
	frame = pd.read_csv(path)
	frame["date_time"] = pd.to_datetime(frame["date_time"])
	frame["hour"] = frame["date_time"].dt.hour
	frame["day_of_week"] = frame["date_time"].dt.dayofweek
	frame["month"] = frame["date_time"].dt.month
	frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
	frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
	frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
	return frame


def load_traffic_data(path: Path = DATA_PATH) -> pd.DataFrame:
	"""Load and cache traffic records until the source file changes."""
	path = path.resolve()
	stat = path.stat()
	return _load_traffic_data_cached(str(path), stat.st_mtime_ns, stat.st_size)


def predict_volume(frame: pd.DataFrame, departure: pd.Timestamp, use_ml: bool = True) -> float:
	"""Predict volume with XGBoost when available, otherwise use a robust profile."""
	if use_ml and XGBRegressor is not None:
		model_entry = _xgb_model_cache.get(id(frame))
		if model_entry is None or model_entry[0] is not frame:
			features = ["hour_sin", "hour_cos", "day_of_week", "month", "temp", "rain_1h", "snow_1h", "clouds_all"]
			train = frame.dropna(subset=features + ["traffic_volume"])
			model = XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, objective="reg:squarederror", n_jobs=2, tree_method="hist")
			model.fit(train[features], train["traffic_volume"])
			_xgb_model_cache[id(frame)] = (frame, model)
		else:
			model = model_entry[1]
		timestamp = pd.DataFrame([{
			"hour_sin": math.sin(2 * math.pi * departure.hour / 24),
			"hour_cos": math.cos(2 * math.pi * departure.hour / 24),
			"day_of_week": departure.dayofweek,
			"month": departure.month,
			"temp": float(frame["temp"].median()),
			"rain_1h": 0.0,
			"snow_1h": 0.0,
			"clouds_all": float(frame["clouds_all"].median()),
		}])
		return float(max(0, model.predict(timestamp)[0]))
	same_hour = frame[frame["hour"] == departure.hour]
	same_day_type = same_hour[same_hour["is_weekend"] == int(departure.dayofweek >= 5)]
	if not same_day_type.empty:
		return float(same_day_type["traffic_volume"].median())
	return float(same_hour["traffic_volume"].median())


def forecast_metrics(frame: pd.DataFrame) -> dict[str, float]:
	"""Evaluate a chronological historical-profile predictor."""
	split = int(len(frame) * 0.8)
	train, test = frame.iloc[:split], frame.iloc[split:]
	profiles = train.groupby(["hour", "is_weekend"])["traffic_volume"].median()
	profile_keys = pd.MultiIndex.from_arrays([test["hour"], test["is_weekend"]])
	fallback = float(train["traffic_volume"].median())
	predictions = profiles.reindex(profile_keys).fillna(fallback).to_numpy(dtype=float)
	actual = test["traffic_volume"].to_numpy(dtype=float)
	predicted = predictions
	error = actual - predicted
	ss_total = np.sum((actual - actual.mean()) ** 2)
	return {
		"MAE": float(np.mean(np.abs(error))),
		"RMSE": float(np.sqrt(np.mean(error**2))),
		"R2": float(1 - np.sum(error**2) / ss_total),
	}


def build_demo_graph() -> dict[str, dict[str, Any]]:
	"""Create a small routable Twin Cities corridor graph for offline demos."""
	graph: dict[str, dict[str, Any]] = {}
	origin_lat, origin_lon = 44.94, -93.18
	for row in range(4):
		for column in range(4):
			node = f"N{row}{column}"
			graph[node] = {"lat": origin_lat + row * 0.012, "lon": origin_lon + column * 0.018, "edges": {}}

	def connect(start: str, end: str, road_type: str) -> None:
		start_data, end_data = graph[start], graph[end]
		distance_km = haversine_km(start_data, end_data)
		speed = 95.0 if road_type == "highway" else 48.0
		capacity = 2_600.0 if road_type == "highway" else 1_200.0
		edge = {"distance_km": distance_km, "speed_kph": speed, "capacity": capacity, "road_type": road_type}
		start_data["edges"][end] = edge
		end_data["edges"][start] = dict(edge)

	for row in range(4):
		for column in range(3):
			connect(f"N{row}{column}", f"N{row}{column + 1}", "highway" if row in (1, 2) else "arterial")
	for row in range(3):
		for column in range(4):
			connect(f"N{row}{column}", f"N{row + 1}{column}", "arterial")
	return graph


def load_osm_graph(place: str, network_type: str = "drive") -> dict[str, dict[str, Any]]:
	"""Download and normalize a drivable OSM graph when OSMnx is installed."""
	place = place.strip()
	if not place:
		raise ValueError("Enter an OSM place, such as 'Vellore, Tamil Nadu, India'.")
	try:
		import osmnx as ox
	except ImportError as error:
		raise RuntimeError("Install osmnx to use live OpenStreetMap routing.") from error
	with _osmnx_request_lock:
		network = ox.graph_from_place(place, network_type=network_type, simplify=True)
	normalized: dict[str, dict[str, Any]] = {}
	for node, data in network.nodes(data=True):
		normalized[str(node)] = {"lat": float(data["y"]), "lon": float(data["x"]), "edges": {}}
	for start, end, data in network.edges(data=True):
		start, end = str(start), str(end)
		distance_km = float(data.get("length", 100)) / 1000
		speed = float(data.get("speed_kph", 45)) if str(data.get("speed_kph", "45")).replace(".", "", 1).isdigit() else 45.0
		normalized[start]["edges"][end] = {"distance_km": distance_km, "speed_kph": speed, "capacity": 1_500.0, "road_type": "osm"}
	return normalized


def geocode_landmark(landmark: str, place: str) -> tuple[float, float]:
	"""Resolve a landmark name within a place using OpenStreetMap Nominatim."""
	landmark = landmark.strip()
	place = place.strip()
	if not landmark:
		raise ValueError("Enter a landmark name.")
	if not place:
		raise ValueError("Enter an OSM place before searching for landmarks.")
	try:
		import osmnx as ox
	except ImportError as error:
		raise RuntimeError("Install osmnx to search landmarks.") from error
	candidates = [landmark]
	for suffix in (" Main Gate", " Railway Station", " Station"):
		if landmark.lower().endswith(suffix.lower()):
			candidates.append(landmark[: -len(suffix)].strip())
	last_error: Exception | None = None
	for candidate in candidates:
		try:
			with _osmnx_request_lock:
				previous_timeout = ox.settings.requests_timeout
				try:
					ox.settings.requests_timeout = GEOCODING_REQUEST_TIMEOUT_SECONDS
					latitude, longitude = ox.geocode(f"{candidate}, {place}")
				finally:
					ox.settings.requests_timeout = previous_timeout
			return float(latitude), float(longitude)
		except requests.exceptions.RequestException:
			raise
		except Exception as error:
			last_error = error
	raise ValueError(f"Could not find '{landmark}' in {place}.") from last_error


def segment_travel_time(
	distance_km: float,
	volume: float,
	base_speed_kph: float = 80.0,
	capacity: float = VOLUME_CAPACITY,
	weather_penalty: float = 0.0,
	incident_penalty: float = 0.0,
) -> float:
	"""Estimate the time for one road segment using a volume-adjusted BPR-style curve."""
	if distance_km <= 0:
		return 0.0
	free_flow_hours = distance_km / max(base_speed_kph, 1e-6)
	congestion_ratio = min(volume / max(capacity, 1e-6), 3.0)
	travel_hours = free_flow_hours * (1 + 0.15 * congestion_ratio**4)
	return max(0.0, (travel_hours * 60) + weather_penalty + incident_penalty)


def estimate_volume_by_hour(frame: pd.DataFrame, departure: pd.Timestamp, horizon_hours: int = 12) -> dict[int, float]:
	"""Build hourly traffic expectations around the departure time for long-distance trip scheduling."""
	profile: dict[int, float] = {}
	for offset in range(horizon_hours):
		hour = (departure.hour + offset) % 24
		hour_frame = frame[frame["hour"] == hour]
		if hour_frame.empty:
			profile[hour] = float(frame["traffic_volume"].median())
		else:
			profile[hour] = float(hour_frame["traffic_volume"].median())
	return profile


def route_intercity_osrm(
	origin: tuple[float, float],
	destination: tuple[float, float],
	volume: float,
	weather_penalty: float = 0.0,
	incident_penalty: float = 0.0,
	departure: pd.Timestamp | datetime | None = None,
	frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
	"""Route between distant landmarks through the public OSRM driving service, with time-dependent congestion around departure."""
	coordinates = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
	query = urlencode({"alternatives": "true", "overview": "full", "geometries": "geojson"})
	request = Request(
		f"https://router.project-osrm.org/route/v1/driving/{coordinates}?{query}",
		headers={"User-Agent": "SmartRouteAI/1.0"},
	)
	with urlopen(request, timeout=60) as response:
		payload = json.loads(response.read().decode("utf-8"))
	if payload.get("code") != "Ok" or not payload.get("routes"):
		raise RuntimeError(payload.get("message", "OSRM could not find a route."))

	if departure is None:
		departure = pd.Timestamp.now().floor("h")
	if frame is None:
		frame = load_traffic_data()
	hourly_profile = estimate_volume_by_hour(frame, pd.Timestamp(departure), horizon_hours=24)

	graph: dict[str, dict[str, Any]] = {}
	routes: list[dict[str, Any]] = []
	for route_number, route in enumerate(payload["routes"][:2], start=1):
		path: list[str] = []
		points = route["geometry"]["coordinates"]
		segment_time_minutes = 0.0
		time_points: list[dict[str, float]] = []
		current_time = pd.Timestamp(departure)
		cumulated_distance = 0.0
		for point_number, (longitude, latitude) in enumerate(points):
			node = f"R{route_number}_{point_number}"
			graph[node] = {"lat": latitude, "lon": longitude, "edges": {}}
			path.append(node)
			if point_number > 0:
				previous_lon, previous_lat = points[point_number - 1]
				segment_distance = haversine_km({"lat": previous_lat, "lon": previous_lon}, {"lat": latitude, "lon": longitude})
				cumulated_distance += segment_distance
				hour_key = current_time.hour
				segment_volume = hourly_profile.get(hour_key, volume)
				segment_minutes = segment_travel_time(
					segment_distance,
					segment_volume,
					base_speed_kph=80.0,
					capacity=VOLUME_CAPACITY,
					weather_penalty=weather_penalty / max(len(points) - 1, 1),
					incident_penalty=incident_penalty / max(len(points) - 1, 1),
				)
				segment_time_minutes += segment_minutes
				current_time = current_time + pd.Timedelta(minutes=segment_minutes)
				time_points.append({"progress_km": cumulated_distance, "arrival_minutes": segment_time_minutes})
		penalty = weather_penalty + incident_penalty
		routes.append({
			"path": path,
			"distance_km": route["distance"] / 1000,
			"travel_minutes": max(route["duration"] / 60, segment_time_minutes) + penalty,
			"congestion_ratio": volume / VOLUME_CAPACITY,
			"time_profile": time_points,
		})
	if len(routes) == 1:
		routes.append(dict(routes[0]))
	return {"graph": graph, "primary": routes[0], "alternative": routes[1]}


def haversine_km(first: dict[str, float], second: dict[str, float]) -> float:
	radius = 6_371.0
	lat1, lat2 = math.radians(first["lat"]), math.radians(second["lat"])
	delta_lat = lat2 - lat1
	delta_lon = math.radians(second["lon"] - first["lon"])
	value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
	return 2 * radius * math.asin(math.sqrt(value))


def nearest_node(graph: dict[str, dict[str, Any]], latitude: float, longitude: float) -> str:
	"""Return the graph node closest to a latitude/longitude pair."""
	return min(graph, key=lambda node: haversine_km(graph[node], {"lat": latitude, "lon": longitude}))


def edge_cost(edge: dict[str, float], volume: float, weather_penalty: float = 0.0, incident_penalty: float = 0.0, multiplier: float = 1.0) -> float:
	free_flow_hours = edge["distance_km"] / edge["speed_kph"]
	congestion_ratio = volume / edge["capacity"]
	travel_hours = free_flow_hours * (1 + 0.15 * congestion_ratio**4)
	return multiplier * (travel_hours * 60 + weather_penalty + incident_penalty)


def astar_route(graph: dict[str, dict[str, Any]], start: str, goal: str, volume: float, weather_penalty: float = 0.0, incident_penalty: float = 0.0, penalized_edges: set[tuple[str, str]] | None = None) -> tuple[list[str], float]:
	"""Find a minimum-cost route; straight-line time is an admissible heuristic."""
	penalized_edges = penalized_edges or set()
	maximum_speed = 95.0

	def heuristic(node: str) -> float:
		return haversine_km(graph[node], graph[goal]) / maximum_speed * 60

	queue: list[tuple[float, float, str]] = [(heuristic(start), 0.0, start)]
	best_cost = {start: 0.0}
	previous: dict[str, str | None] = {start: None}
	while queue:
		_, cost, node = heapq.heappop(queue)
		if node == goal:
			path: list[str] = []
			current: str | None = goal
			while current is not None:
				path.append(current)
				current = previous[current]
			path.reverse()
			return path, cost
		if cost > best_cost.get(node, float("inf")):
			continue
		for neighbor, edge in graph[node]["edges"].items():
			multiplier = 1.35 if (node, neighbor) in penalized_edges else 1.0
			next_cost = cost + edge_cost(edge, volume, weather_penalty, incident_penalty, multiplier)
			if next_cost < best_cost.get(neighbor, float("inf")):
				best_cost[neighbor] = next_cost
				previous[neighbor] = node
				heapq.heappush(queue, (next_cost + heuristic(neighbor), next_cost, neighbor))
	raise ValueError(f"No route exists between {start} and {goal}")


def route_summary(graph: dict[str, dict[str, Any]], path: list[str], volume: float, weather_penalty: float = 0.0, incident_penalty: float = 0.0) -> dict[str, float]:
	edges = [graph[first]["edges"][second] for first, second in zip(path, path[1:])]
	return {
		"distance_km": sum(edge["distance_km"] for edge in edges),
		"travel_minutes": sum(edge_cost(edge, volume, weather_penalty, incident_penalty) for edge in edges),
		"congestion_ratio": volume / VOLUME_CAPACITY,
	}


def evaluate_route_checkpoints(
	graph: dict[str, dict[str, Any]],
	path: list[str],
	departure: pd.Timestamp,
	weather_penalty: float = 0.0,
	incident_penalty: float = 0.0,
	use_ml: bool = True,
	checkpoint_interval_minutes: int = 60,
	congestion_threshold: float = 0.75,
	frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
	"""Flag congested checkpoints and re-search a route after an interval if conditions worsen."""
	if len(path) < 2:
		return {"checkpoints": [], "reroute": None, "reason": "Route is too short to evaluate checkpoints."}
	if frame is None:
		frame = load_traffic_data()
	current_time = pd.Timestamp(departure)
	accumulated_minutes = 0.0
	checkpoints: list[dict[str, Any]] = []
	reroute: dict[str, Any] | None = None
	for index in range(1, len(path)):
		edge = graph[path[index - 1]]["edges"].get(path[index])
		if edge is None:
			continue
		segment_volume = predict_volume(frame, current_time, use_ml=use_ml)
		segment_minutes = edge_cost(edge, segment_volume, weather_penalty, incident_penalty)
		accumulated_minutes += segment_minutes
		if accumulated_minutes >= checkpoint_interval_minutes:
			congestion_ratio = segment_volume / VOLUME_CAPACITY
			status = "high" if congestion_ratio >= congestion_threshold else "normal"
			reason = (
				f"Historical traffic at {current_time.strftime('%H:%M')} is high "
				f"({congestion_ratio:.2f} of capacity), so the route should be re-evaluated."
				if status == "high"
				else f"Traffic at {current_time.strftime('%H:%M')} is acceptable, so the route remains stable."
			)
			checkpoint = {
				"checkpoint_index": index,
				"node": path[index],
				"time_minutes": round(accumulated_minutes, 1),
				"hour_label": current_time.strftime("%H:%M"),
				"congestion_ratio": round(congestion_ratio, 3),
				"status": status,
				"reason": reason,
			}
			checkpoints.append(checkpoint)
			if status == "high":
				try:
					replanned_path, _ = astar_route(graph, path[index], path[-1], segment_volume, weather_penalty, incident_penalty)
					reroute = {
						"from_node": path[index],
						"to_node": path[-1],
						"path": replanned_path,
						"reason": reason,
					}
				except Exception:
					reroute = {"from_node": path[index], "to_node": path[-1], "path": path[index:], "reason": reason}
			accumulated_minutes = 0.0
		current_time += pd.Timedelta(minutes=segment_minutes)
	return {"checkpoints": checkpoints, "reroute": reroute, "old_path": path, "highlighted_path": reroute["path"] if reroute else path}


def shortest_distance_path(graph: dict[str, dict[str, Any]], start: str, goal: str) -> tuple[list[str], float]:
	"""Compute a pure geometry-based shortest route by road distance."""
	queue: list[tuple[float, str]] = [(0.0, start)]
	best_cost: dict[str, float] = {start: 0.0}
	previous: dict[str, str | None] = {start: None}
	while queue:
		current_cost, node = heapq.heappop(queue)
		if node == goal:
			break
		if current_cost > best_cost.get(node, float("inf")):
			continue
		for neighbor, edge in graph[node]["edges"].items():
			next_cost = current_cost + float(edge["distance_km"])
			if next_cost < best_cost.get(neighbor, float("inf")):
				best_cost[neighbor] = next_cost
				previous[neighbor] = node
				heapq.heappush(queue, (next_cost, neighbor))
	if goal not in best_cost:
		raise ValueError(f"No route exists between {start} and {goal}")
	path: list[str] = []
	node: str | None = goal
	while node is not None:
		path.append(node)
		node = previous[node]
	path.reverse()
	return path, best_cost[goal]


def route_score(graph: dict[str, dict[str, Any]], path: list[str], volume: float, weather_penalty: float = 0.0, incident_penalty: float = 0.0, reference_distance_km: float | None = None) -> tuple[float, dict[str, float]]:
	"""Score a route using geometry, time, and congestion while keeping A* heuristics in the search layer."""
	metrics = route_summary(graph, path, volume, weather_penalty, incident_penalty)
	if reference_distance_km is None:
		_, reference_distance_km = shortest_distance_path(graph, path[0], path[-1])
	geometry_only_distance = reference_distance_km
	geometry_only_distance = max(geometry_only_distance, 1e-6)
	travel_weight = 0.45
	distance_weight = 0.35
	congestion_weight = 0.20
	normalized_time = metrics["travel_minutes"] / max(metrics["travel_minutes"] + 1.0, 1.0)
	normalized_distance = metrics["distance_km"] / max(geometry_only_distance, 1e-6)
	normalized_congestion = min(1.0, metrics["congestion_ratio"] / 1.5)
	operational_penalty = (weather_penalty + incident_penalty) / 20.0
	score = (
		travel_weight * normalized_time
		+ distance_weight * normalized_distance
		+ congestion_weight * normalized_congestion
		+ operational_penalty
	)
	return score, metrics


def recommend_routes(departure: pd.Timestamp, weather_penalty: float = 0.0, incident_penalty: float = 0.0, graph: dict[str, dict[str, Any]] | None = None, use_ml: bool = True, start_node: str | None = None, goal_node: str | None = None) -> dict[str, Any]:
	frame = load_traffic_data()
	graph = graph or build_demo_graph()
	volume = predict_volume(frame, departure, use_ml=use_ml)
	start = start_node or ("N00" if "N00" in graph else next(iter(graph)))
	goal = goal_node or ("N33" if "N33" in graph else list(graph)[-1])

	primary, _ = astar_route(graph, start, goal, volume, weather_penalty, incident_penalty)
	alternative, _ = astar_route(graph, start, goal, volume, weather_penalty, incident_penalty, set(zip(primary, primary[1:])))
	geometry_path, geometry_distance = shortest_distance_path(graph, start, goal)
	long_distance_evolution = evaluate_route_checkpoints(graph, primary, pd.Timestamp(departure), weather_penalty, incident_penalty, use_ml=use_ml, frame=frame)

	candidates: list[dict[str, Any]] = []
	route_results: dict[str, dict[str, Any]] = {}
	for label, path in (("primary", primary), ("alternative", alternative), ("shortest_distance", geometry_path)):
		score, metrics = route_score(graph, path, volume, weather_penalty, incident_penalty, geometry_distance)
		route = {"path": path, **metrics, "score": score}
		route_results[label] = route
		candidate = {"route_name": label, **route}
		candidates.append(candidate)

	selected = min(candidates, key=lambda item: item["score"])
	selected_path = selected["path"]
	selected_metrics = {key: value for key, value in selected.items() if key not in {"route_name", "path", "score"}}
	return {
		"predicted_volume": volume,
		"forecast_metrics": forecast_metrics(frame),
		"graph": graph,
		"primary": route_results["primary"],
		"alternative": route_results["alternative"],
		"shortest_distance": route_results["shortest_distance"],
		"selected_route": {"path": selected_path, **selected_metrics, "score": selected["score"]},
		"route_evolution": long_distance_evolution,
	}


if __name__ == "__main__":
	result = recommend_routes(pd.Timestamp.now().floor("h"))
	print(f"Loaded {len(load_traffic_data()):,} traffic observations")
	print("Forecast metrics:", {key: round(value, 2) for key, value in result["forecast_metrics"].items()})
	print(f"Predicted volume: {result['predicted_volume']:,.0f} vehicles/hour")
	print("Primary route:", " -> ".join(result["primary"]["path"]))
	print("Alternative route:", " -> ".join(result["alternative"]["path"]))