import sys
from types import SimpleNamespace

import pytest
import requests

from traffic_analysis import GEOCODING_REQUEST_TIMEOUT_SECONDS, geocode_landmark


def test_geocode_landmark_uses_bounded_timeout_and_restores_osmnx_setting(monkeypatch):
    observed_timeouts = []
    settings = SimpleNamespace(requests_timeout=180.0)

    def geocode(query):
        observed_timeouts.append(settings.requests_timeout)
        return 12.9, 79.1

    monkeypatch.setitem(
        sys.modules,
        "osmnx",
        SimpleNamespace(settings=settings, geocode=geocode),
    )

    assert geocode_landmark("Katpadi Junction Railway Station", "Vellore") == (
        12.9,
        79.1,
    )
    assert observed_timeouts == [GEOCODING_REQUEST_TIMEOUT_SECONDS]
    assert settings.requests_timeout == 180.0


def test_geocode_landmark_does_not_retry_alternate_query_after_network_timeout(monkeypatch):
    observed_queries = []
    settings = SimpleNamespace(requests_timeout=180.0)

    def geocode(query):
        observed_queries.append((query, settings.requests_timeout))
        raise requests.exceptions.Timeout("simulated timeout")

    monkeypatch.setitem(
        sys.modules,
        "osmnx",
        SimpleNamespace(settings=settings, geocode=geocode),
    )

    with pytest.raises(requests.exceptions.Timeout):
        geocode_landmark("Katpadi Junction Railway Station", "Vellore")

    assert observed_queries == [
        (
            "Katpadi Junction Railway Station, Vellore",
            GEOCODING_REQUEST_TIMEOUT_SECONDS,
        )
    ]
    assert settings.requests_timeout == 180.0
