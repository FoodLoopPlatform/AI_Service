from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any
import httpx
from pydantic import BaseModel, Field

from app.config.settings import settings


class WeatherToolError(Exception):
    """Raised when the weather context retrieval tool fails."""

    pass


class WeatherForecast(BaseModel):
    timestamp: datetime
    condition: str
    precipitation_probability: float = Field(..., ge=0.0, le=1.0)
    temperature: float


class WeatherContext(BaseModel):
    forecast: list[WeatherForecast]


WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing Rime Fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Dense Drizzle",
    56: "Light Freezing Drizzle",
    57: "Dense Freezing Drizzle",
    61: "Slight Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    66: "Light Freezing Rain",
    67: "Heavy Freezing Rain",
    71: "Slight Snow",
    73: "Moderate Snow",
    75: "Heavy Snow",
    77: "Snow Grains",
    80: "Slight Rain Showers",
    81: "Moderate Rain Showers",
    82: "Violent Rain Showers",
    85: "Slight Snow Showers",
    86: "Heavy Snow Showers",
    95: "Thunderstorm",
    96: "Thunderstorm with Light Hail",
    99: "Thunderstorm with Heavy Hail",
}


def map_wmo_code_to_condition(code: int) -> str:
    """Maps numeric WMO weather interpretation code to a human-readable condition string."""
    return WMO_WEATHER_CODES.get(code, "Clear")


class WeatherProvider(ABC):
    """Abstract interface for weather forecast providers."""

    @abstractmethod
    def get_weather_forecast(
        self,
        latitude: float,
        longitude: float,
        from_time: datetime,
        to_time: datetime,
    ) -> WeatherContext:
        """Retrieves weather context for a given location and time window."""
        pass


class MockWeatherProvider(WeatherProvider):
    """Deterministic mock weather provider for offline testing and development."""

    def get_weather_forecast(
        self,
        latitude: float,
        longitude: float,
        from_time: datetime,
        to_time: datetime,
    ) -> WeatherContext:
        if not (-90.0 <= latitude <= 90.0):
            raise WeatherToolError(
                f"Invalid latitude value: {latitude}. Must be between -90 and 90."
            )
        if not (-180.0 <= longitude <= 180.0):
            raise WeatherToolError(
                f"Invalid longitude value: {longitude}. Must be between -180 and 180."
            )

        if from_time.tzinfo is None or to_time.tzinfo is None:
            raise WeatherToolError("from_time and to_time must be timezone-aware datetimes.")

        if from_time >= to_time:
            return WeatherContext(forecast=[])

        forecast_items: list[WeatherForecast] = []
        current = from_time
        interval = timedelta(hours=6)

        while current < to_time:
            hour = current.hour
            if hour in (6, 12):
                condition = "Sunny"
                precip = 0.05
                temp = 22.5
            elif hour == 18:
                condition = "Partly Cloudy"
                precip = 0.15
                temp = 19.0
            else:
                condition = "Clear"
                precip = 0.0
                temp = 15.0

            forecast_items.append(
                WeatherForecast(
                    timestamp=current,
                    condition=condition,
                    precipitation_probability=precip,
                    temperature=temp,
                )
            )
            current += interval

        return WeatherContext(forecast=forecast_items)


class OpenMeteoWeatherProvider(WeatherProvider):
    """Production Weather Provider communicating with Open-Meteo Forecast API."""

    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com/v1/forecast",
        api_key: str = "",
        timeout: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def get_weather_forecast(
        self,
        latitude: float,
        longitude: float,
        from_time: datetime,
        to_time: datetime,
    ) -> WeatherContext:
        if not (-90.0 <= latitude <= 90.0):
            raise WeatherToolError(
                f"Invalid latitude value: {latitude}. Must be between -90 and 90."
            )
        if not (-180.0 <= longitude <= 180.0):
            raise WeatherToolError(
                f"Invalid longitude value: {longitude}. Must be between -180 and 180."
            )

        if from_time.tzinfo is None or to_time.tzinfo is None:
            raise WeatherToolError("from_time and to_time must be timezone-aware UTC datetimes.")

        # Ensure UTC timezone conversion
        from_time_utc = from_time.astimezone(timezone.utc)
        to_time_utc = to_time.astimezone(timezone.utc)

        if from_time_utc >= to_time_utc:
            return WeatherContext(forecast=[])

        start_hour = from_time_utc.strftime("%Y-%m-%dT%H:00")
        end_hour = to_time_utc.strftime("%Y-%m-%dT%H:00")

        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,precipitation_probability,weather_code",
            "start_hour": start_hour,
            "end_hour": end_hour,
            "timezone": "UTC",
        }
        if self.api_key:
            params["apikey"] = self.api_key

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise WeatherToolError(f"Open-Meteo API call timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise WeatherToolError(
                f"Open-Meteo API returned HTTP error {exc.response.status_code}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise WeatherToolError(f"Open-Meteo API network error: {exc}") from exc
        except Exception as exc:
            raise WeatherToolError(f"Failed to query Open-Meteo API: {exc}") from exc

        return self._map_response(data, from_time_utc, to_time_utc)

    def _map_response(
        self, data: dict[str, Any], from_time_utc: datetime, to_time_utc: datetime
    ) -> WeatherContext:
        if not isinstance(data, dict):
            raise WeatherToolError("Malformed Open-Meteo response: top-level JSON is not an object.")

        hourly = data.get("hourly")
        if not isinstance(hourly, dict):
            raise WeatherToolError("Malformed Open-Meteo response: missing or invalid 'hourly' section.")

        times = hourly.get("time")
        temps = hourly.get("temperature_2m")
        precips = hourly.get("precipitation_probability")
        codes = hourly.get("weather_code")

        if (
            not isinstance(times, list)
            or not isinstance(temps, list)
            or not isinstance(precips, list)
            or not isinstance(codes, list)
        ):
            raise WeatherToolError(
                "Malformed Open-Meteo response: missing required forecast arrays in 'hourly'."
            )

        if not (len(times) == len(temps) == len(precips) == len(codes)):
            raise WeatherToolError(
                "Malformed Open-Meteo response: forecast arrays have mismatched lengths."
            )

        forecast_items: list[WeatherForecast] = []

        for i in range(len(times)):
            try:
                time_str = times[i]
                # Open-Meteo returns timestamps like "2026-08-15T12:00"
                dt = datetime.fromisoformat(time_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)

                # Include item if within requested window [from_time_utc, to_time_utc)
                if dt < from_time_utc or dt >= to_time_utc:
                    continue

                temp = float(temps[i])
                raw_precip = float(precips[i])
                precip_prob = max(0.0, min(1.0, raw_precip / 100.0))
                code = int(codes[i])
                condition = map_wmo_code_to_condition(code)

                forecast_items.append(
                    WeatherForecast(
                        timestamp=dt,
                        condition=condition,
                        precipitation_probability=precip_prob,
                        temperature=temp,
                    )
                )
            except (ValueError, TypeError) as exc:
                raise WeatherToolError(
                    f"Malformed Open-Meteo response: invalid values at index {i}: {exc}"
                ) from exc

        return WeatherContext(forecast=forecast_items)


_weather_provider_instance: WeatherProvider | None = None


def get_weather_provider() -> WeatherProvider:
    """Returns the configured weather provider instance."""
    global _weather_provider_instance
    if _weather_provider_instance is not None:
        return _weather_provider_instance

    provider_type = settings.WEATHER_PROVIDER.lower()
    if provider_type == "open_meteo":
        return OpenMeteoWeatherProvider(
            base_url=settings.WEATHER_API_BASE_URL,
            api_key=settings.WEATHER_API_KEY,
            timeout=settings.WEATHER_API_TIMEOUT_SECONDS,
        )
    return MockWeatherProvider()


def set_weather_provider(provider: WeatherProvider | None) -> None:
    """Overrides the global weather provider instance (useful for testing)."""
    global _weather_provider_instance
    _weather_provider_instance = provider


def get_weather_forecast(
    latitude: float,
    longitude: float,
    from_time: datetime,
    to_time: datetime,
) -> WeatherContext:
    """Retrieves weather context for a given location and time window using the configured provider."""
    provider = get_weather_provider()
    return provider.get_weather_forecast(
        latitude=latitude,
        longitude=longitude,
        from_time=from_time,
        to_time=to_time,
    )
