from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from typing import Any
import httpx
from pydantic import BaseModel, Field

from app.config.settings import settings


class LocalEventsToolError(Exception):
    """Raised when the holiday context retrieval tool fails."""

    pass


class Holiday(BaseModel):
    """Domain model representing a public / national holiday or official occasion."""

    date: date
    name: str
    country_code: str
    national_holiday: bool
    holiday_types: list[str] = Field(default_factory=list)


class HolidayContext(BaseModel):
    """Domain context containing a list of holidays for a store/country context."""

    holidays: list[Holiday] = Field(default_factory=list)


# Aliases for backward compatibility across existing schemas and agent contracts
LocalEventsContext = HolidayContext
LocalEvent = Holiday


class HolidayProvider(ABC):
    """Abstract interface for holiday context providers."""

    @abstractmethod
    def get_holidays(
        self,
        country_code: str,
        year: int,
    ) -> list[Holiday]:
        """Retrieves public holidays for a given country code and calendar year."""
        pass

    def get_holidays_for_range(
        self,
        country_code: str,
        from_time: datetime,
        to_time: datetime,
    ) -> HolidayContext:
        """Retrieves and filters holidays for a given datetime window."""
        if from_time.tzinfo is None or to_time.tzinfo is None:
            raise LocalEventsToolError("from_time and to_time must be timezone-aware datetimes.")

        from_utc = from_time.astimezone(timezone.utc)
        to_utc = to_time.astimezone(timezone.utc)

        if from_utc >= to_utc:
            return HolidayContext(holidays=[])

        start_year = from_utc.year
        end_year = to_utc.year

        all_holidays: list[Holiday] = []
        for year in range(start_year, end_year + 1):
            year_holidays = self.get_holidays(country_code=country_code, year=year)
            all_holidays.extend(year_holidays)

        from_date = from_utc.date()
        to_date = to_utc.date()

        filtered = [
            h for h in all_holidays
            if from_date <= h.date <= to_date
        ]
        filtered.sort(key=lambda h: h.date)
        return HolidayContext(holidays=filtered)


class MockHolidayProvider(HolidayProvider):
    """Deterministic mock holiday provider for offline unit testing and development."""

    def get_holidays(
        self,
        country_code: str,
        year: int,
    ) -> list[Holiday]:
        if not country_code or not isinstance(country_code, str) or len(country_code.strip()) == 0:
            raise LocalEventsToolError("country_code must be a non-empty string.")

        code = country_code.strip().upper()

        return [
            Holiday(
                date=date(year, 1, 7),
                name="Coptic Christmas Day",
                country_code=code,
                national_holiday=True,
                holiday_types=["Public"],
            ),
            Holiday(
                date=date(year, 1, 25),
                name="Revolution Day January 25",
                country_code=code,
                national_holiday=True,
                holiday_types=["Public"],
            ),
            Holiday(
                date=date(year, 6, 30),
                name="30 June Revolution",
                country_code=code,
                national_holiday=True,
                holiday_types=["Public"],
            ),
            Holiday(
                date=date(year, 7, 23),
                name="Revolution Day July 23",
                country_code=code,
                national_holiday=True,
                holiday_types=["Public"],
            ),
            Holiday(
                date=date(year, 10, 6),
                name="Armed Forces Day",
                country_code=code,
                national_holiday=True,
                holiday_types=["Public"],
            ),
        ]


class NagerDateHolidayProvider(HolidayProvider):
    """Production provider integrating with Nager.Date Public Holidays API v4."""

    def __init__(
        self,
        base_url: str = "https://date.nager.at/api/v4",
        timeout: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache: dict[tuple[str, int], list[Holiday]] = {}

    def get_holidays(
        self,
        country_code: str,
        year: int,
    ) -> list[Holiday]:
        if not country_code or not isinstance(country_code, str) or len(country_code.strip()) == 0:
            raise LocalEventsToolError("country_code must be a non-empty string.")

        code = country_code.strip().upper()
        cache_key = (code, year)
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{self.base_url}/Holidays/{code}/{year}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise LocalEventsToolError(f"Nager.Date API call timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LocalEventsToolError(
                f"Nager.Date API returned HTTP status {exc.response.status_code}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise LocalEventsToolError(f"Nager.Date API network error: {exc}") from exc
        except Exception as exc:
            raise LocalEventsToolError(f"Failed to query Nager.Date API: {exc}") from exc

        holidays = self._map_response(data, code, year)
        self._cache[cache_key] = holidays
        return holidays

    def _map_response(self, data: Any, expected_country: str, expected_year: int) -> list[Holiday]:
        if not isinstance(data, list):
            raise LocalEventsToolError("Malformed Nager.Date response: expected a JSON list of holidays.")

        holidays: list[Holiday] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise LocalEventsToolError(
                    f"Malformed Nager.Date response: item at index {idx} is not an object."
                )

            date_str = item.get("date")
            name = item.get("name") or item.get("localName")
            c_code = item.get("countryCode") or expected_country

            if not date_str or not name:
                raise LocalEventsToolError(
                    f"Malformed Nager.Date response: item at index {idx} missing required fields."
                )

            try:
                h_date = date.fromisoformat(date_str)
            except ValueError as exc:
                raise LocalEventsToolError(
                    f"Malformed Nager.Date response: invalid date format '{date_str}': {exc}"
                ) from exc

            types_list = item.get("holidayTypes") or item.get("types") or ["Public"]
            if not isinstance(types_list, list):
                types_list = [str(types_list)]

            is_national = item.get("nationalHoliday")
            if is_national is None:
                is_national = bool(item.get("global", True) or ("Public" in types_list))
            else:
                is_national = bool(is_national)

            holidays.append(
                Holiday(
                    date=h_date,
                    name=str(name),
                    country_code=str(c_code).upper(),
                    national_holiday=is_national,
                    holiday_types=[str(t) for t in types_list],
                )
            )

        return holidays


_holiday_provider_instance: HolidayProvider | None = None


def get_holiday_provider() -> HolidayProvider:
    """Returns the configured holiday provider instance."""
    global _holiday_provider_instance
    if _holiday_provider_instance is not None:
        return _holiday_provider_instance

    provider_type = settings.EVENTS_PROVIDER.lower()
    if provider_type in ("nager", "nager_date"):
        return NagerDateHolidayProvider(

            base_url=settings.HOLIDAY_API_BASE_URL,
            timeout=settings.HOLIDAY_API_TIMEOUT_SECONDS,
        )
    return MockHolidayProvider()


def set_holiday_provider(provider: HolidayProvider | None) -> None:
    """Overrides the global holiday provider instance (useful for testing)."""
    global _holiday_provider_instance
    _holiday_provider_instance = provider


def get_local_events(
    latitude: float,
    longitude: float,
    from_time: datetime,
    to_time: datetime,
    country_code: str | None = None,
) -> HolidayContext:
    """Retrieves deterministic holiday/occasions context for a country and time window using the configured provider.
    
    Accepts latitude and longitude for signature compatibility with location context.
    """
    code = country_code or settings.DEFAULT_COUNTRY_CODE
    provider = get_holiday_provider()
    return provider.get_holidays_for_range(
        country_code=code,
        from_time=from_time,
        to_time=to_time,
    )
