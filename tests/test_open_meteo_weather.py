from datetime import datetime, timedelta, timezone
import os
from unittest.mock import MagicMock, patch
import httpx
import pytest

from app.config.settings import settings
from app.tools.weather import (
    MockWeatherProvider,
    OpenMeteoWeatherProvider,
    WeatherContext,
    WeatherForecast,
    WeatherToolError,
    get_weather_forecast,
    get_weather_provider,
    map_wmo_code_to_condition,
    set_weather_provider,
)


@pytest.fixture(autouse=True)
def reset_provider():
    set_weather_provider(None)
    yield
    set_weather_provider(None)


def test_wmo_code_mapping():
    assert map_wmo_code_to_condition(0) == "Clear"
    assert map_wmo_code_to_condition(1) == "Mainly Clear"
    assert map_wmo_code_to_condition(3) == "Overcast"
    assert map_wmo_code_to_condition(63) == "Moderate Rain"
    assert map_wmo_code_to_condition(95) == "Thunderstorm"
    assert map_wmo_code_to_condition(9999) == "Clear"  # Default fallback


def test_invalid_coordinates():
    provider = OpenMeteoWeatherProvider()
    from_t = datetime.now(timezone.utc)
    to_t = from_t + timedelta(hours=6)

    with pytest.raises(WeatherToolError) as exc_info:
        provider.get_weather_forecast(95.0, 0.0, from_t, to_t)
    assert "Invalid latitude" in str(exc_info.value)

    with pytest.raises(WeatherToolError) as exc_info:
        provider.get_weather_forecast(0.0, -200.0, from_t, to_t)
    assert "Invalid longitude" in str(exc_info.value)


def test_naive_datetime_validation():
    provider = OpenMeteoWeatherProvider()
    naive_from = datetime(2026, 8, 15, 12, 0)
    aware_to = datetime(2026, 8, 15, 18, 0, tzinfo=timezone.utc)

    with pytest.raises(WeatherToolError) as exc_info:
        provider.get_weather_forecast(30.0, 31.0, naive_from, aware_to)
    assert "timezone-aware" in str(exc_info.value)


def test_valid_open_meteo_response_mapping():
    provider = OpenMeteoWeatherProvider(base_url="https://api.open-meteo.com/v1/forecast")

    from_t = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    to_t = datetime(2026, 8, 15, 15, 0, tzinfo=timezone.utc)

    mock_json = {
        "hourly": {
            "time": ["2026-08-15T12:00", "2026-08-15T13:00", "2026-08-15T14:00"],
            "temperature_2m": [25.5, 26.0, 24.8],
            "precipitation_probability": [0, 85, 120],  # 120 should be clamped to 1.0
            "weather_code": [0, 63, 95],
        }
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_json

    with patch("httpx.Client.get", return_value=mock_response) as mock_get:
        result = provider.get_weather_forecast(30.0, 31.0, from_t, to_t)

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        url = call_kwargs[0][0]
        params = call_kwargs[1]["params"]

        assert url == "https://api.open-meteo.com/v1/forecast"
        assert params["latitude"] == 30.0
        assert params["longitude"] == 31.0
        assert params["start_hour"] == "2026-08-15T12:00"
        assert params["end_hour"] == "2026-08-15T15:00"
        assert params["timezone"] == "UTC"
        assert "apikey" not in params  # Dev endpoint without API key

        assert len(result.forecast) == 3
        assert result.forecast[0].condition == "Clear"
        assert result.forecast[0].temperature == 25.5
        assert result.forecast[0].precipitation_probability == 0.0

        assert result.forecast[1].condition == "Moderate Rain"
        assert result.forecast[1].temperature == 26.0
        assert result.forecast[1].precipitation_probability == 0.85

        assert result.forecast[2].condition == "Thunderstorm"
        assert result.forecast[2].precipitation_probability == 1.0  # Clamped


def test_api_key_added_when_configured():
    provider = OpenMeteoWeatherProvider(
        base_url="https://customer-api.open-meteo.com/v1/forecast",
        api_key="secret-customer-key-123",
    )
    from_t = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    to_t = datetime(2026, 8, 15, 13, 0, tzinfo=timezone.utc)

    mock_json = {
        "hourly": {
            "time": ["2026-08-15T12:00"],
            "temperature_2m": [20.0],
            "precipitation_probability": [10],
            "weather_code": [1],
        }
    }
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = mock_json

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        provider.get_weather_forecast(30.0, 31.0, from_t, to_t)
        params = mock_get.call_args[1]["params"]
        assert params["apikey"] == "secret-customer-key-123"


def test_open_meteo_http_error():
    provider = OpenMeteoWeatherProvider()
    from_t = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    to_t = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Internal Error", request=MagicMock(), response=mock_resp
    )

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(WeatherToolError) as exc_info:
            provider.get_weather_forecast(30.0, 31.0, from_t, to_t)
        assert "HTTP error 500" in str(exc_info.value)


def test_open_meteo_timeout_error():
    provider = OpenMeteoWeatherProvider()
    from_t = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    to_t = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)

    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Timed out")):
        with pytest.raises(WeatherToolError) as exc_info:
            provider.get_weather_forecast(30.0, 31.0, from_t, to_t)
        assert "timed out" in str(exc_info.value)


def test_open_meteo_malformed_response():
    provider = OpenMeteoWeatherProvider()
    from_t = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    to_t = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)

    # Missing hourly key
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"invalid": "data"}

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(WeatherToolError) as exc_info:
            provider.get_weather_forecast(30.0, 31.0, from_t, to_t)
        assert "missing or invalid 'hourly' section" in str(exc_info.value)


def test_provider_selection_by_settings():
    with patch.object(settings, "WEATHER_PROVIDER", "open_meteo"):
        prov = get_weather_provider()
        assert isinstance(prov, OpenMeteoWeatherProvider)

    with patch.object(settings, "WEATHER_PROVIDER", "mock"):
        set_weather_provider(None)
        prov_mock = get_weather_provider()
        assert isinstance(prov_mock, MockWeatherProvider)


@pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_INTEGRATION_TESTS") != "true",
    reason="Integration test requires RUN_EXTERNAL_INTEGRATION_TESTS=true",
)
def test_real_open_meteo_live_integration():
    """Optional live integration test calling the real Open-Meteo API when explicitly enabled."""
    provider = OpenMeteoWeatherProvider(base_url="https://api.open-meteo.com/v1/forecast")
    from_t = datetime.now(timezone.utc)
    to_t = from_t + timedelta(hours=12)

    result = provider.get_weather_forecast(30.0444, 31.2357, from_t, to_t)
    assert isinstance(result, WeatherContext)
    assert isinstance(result.forecast, list)
    if len(result.forecast) > 0:
        first = result.forecast[0]
        assert isinstance(first.timestamp, datetime)
        assert isinstance(first.condition, str)
        assert 0.0 <= first.precipitation_probability <= 1.0
