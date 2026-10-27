import pytest
import requests

import traffic_analysis
from traffic_analysis import GEOCODING_REQUEST_TIMEOUT_SECONDS, geocode_landmark


def test_geocode_landmark_uses_bounded_nominatim_request(monkeypatch):
    observed_requests = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"lat": "12.9", "lon": "79.1"}]

    def get(url, **kwargs):
        observed_requests.append((url, kwargs))
        return Response()

    monkeypatch.setattr(traffic_analysis.requests, "get", get)
    traffic_analysis._geocode_nominatim_query.cache_clear()
    assert geocode_landmark("Katpadi Junction Railway Station", "Vellore") == (
        12.9,
        79.1,
    )
    assert len(observed_requests) == 1
    url, kwargs = observed_requests[0]
    assert url == traffic_analysis.NOMINATIM_SEARCH_URL
    assert kwargs["timeout"] == (5.0, GEOCODING_REQUEST_TIMEOUT_SECONDS)
    assert kwargs["params"]["q"] == "Katpadi Junction Railway Station, Vellore"
    assert "SmartRouteAI" in kwargs["headers"]["User-Agent"]


def test_geocode_landmark_does_not_retry_after_http_failure(monkeypatch):
    observed_requests = []

    class Response:
        def raise_for_status(self):
            raise requests.exceptions.HTTPError("simulated 429 response")

    def get(url, **kwargs):
        observed_requests.append(kwargs)
        return Response()

    monkeypatch.setattr(traffic_analysis.requests, "get", get)
    traffic_analysis._geocode_nominatim_query.cache_clear()
    with pytest.raises(requests.exceptions.HTTPError):
        geocode_landmark("Katpadi Junction Railway Station", "Vellore")

    assert len(observed_requests) == 1
