"""Shared map rendering helpers for SmartRoute route geometries."""

from __future__ import annotations

import math
from typing import Any

import pydeck as pdk


def build_route_deck(
	graph: dict[str, dict[str, Any]],
	routes: list[dict[str, Any]],
) -> pdk.Deck | None:
	"""Build a fitted map with colored route lines instead of node markers."""
	path_data: list[dict[str, Any]] = []
	coordinates: list[list[float]] = []
	for route in routes:
		path = [
			[float(graph[node]["lon"]), float(graph[node]["lat"])]
			for node in route["path"]
			if node in graph
		]
		if len(path) < 2:
			continue
		path_data.append({
			"route": route["label"],
			"path": path,
			"color": route["color"],
		})
		coordinates.extend(path)
	if not path_data:
		return None

	longitudes = [point[0] for point in coordinates]
	latitudes = [point[1] for point in coordinates]
	span = max(max(longitudes) - min(longitudes), max(latitudes) - min(latitudes))
	zoom = max(2, min(15, math.log2(360 / max(span * 1.5, 0.01)) - 1))
	layer = pdk.Layer(
		"PathLayer",
		data=path_data,
		get_path="path",
		get_color="color",
		get_width=5,
		width_min_pixels=4,
		width_scale=1,
		pickable=True,
		auto_highlight=True,
	)
	return pdk.Deck(
		layers=[layer],
		initial_view_state=pdk.ViewState(
			latitude=(min(latitudes) + max(latitudes)) / 2,
			longitude=(min(longitudes) + max(longitudes)) / 2,
			zoom=zoom,
		),
		map_style=pdk.map_styles.DARK,
		tooltip={"text": "{route}"},
	)