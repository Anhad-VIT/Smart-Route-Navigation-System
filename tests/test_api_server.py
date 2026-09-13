from datetime import datetime

from api_server import build_route_plan


def test_build_route_plan_uses_traffic_prediction():
    result = build_route_plan(
        origin="VIT University Main Gate",
        destination="Katpadi Junction Railway Station",
        place="Vellore, Tamil Nadu, India",
        departure=datetime(2026, 9, 13, 8, 0),
        mode="within_city",
        weather_penalty=1.0,
        incident_penalty=2.0,
        use_ml=False,
    )

    assert result["status"] == "ok"
    assert result["predicted_volume"] > 0
    assert result["primary"]["travel_minutes"] > 0
    assert result["alternative"]["travel_minutes"] > 0
    assert "selected_route" in result
    assert result["selected_route"]["distance_km"] > 0
    assert "graph" in result


def test_build_route_plan_supports_intercity_origin_and_destination_areas():
    result = build_route_plan(
        origin="Mumbai Central Railway Station",
        destination="Pune Junction Railway Station",
        place="Mumbai, Maharashtra, India",
        origin_place="Mumbai, Maharashtra, India",
        destination_place="Pune, Maharashtra, India",
        departure=datetime(2026, 9, 13, 8, 0),
        mode="intercity",
        weather_penalty=1.0,
        incident_penalty=2.0,
        use_ml=False,
    )

    assert result["status"] == "ok"
    assert result["travel_mode"] == "intercity"
    assert result["primary"]["distance_km"] > 0
    assert result["alternative"]["distance_km"] > 0
