from datetime import date, datetime, timedelta, timezone
import os
from unittest.mock import MagicMock, patch
import httpx
import pytest

from app.config.settings import settings
from app.tools.events import (
    Holiday,
    HolidayContext,
    LocalEventsToolError,
    MockHolidayProvider,
    NagerDateHolidayProvider,
    get_holiday_provider,
    set_holiday_provider,
)


@pytest.fixture(autouse=True)
def reset_events_provider():
    set_holiday_provider(None)
    yield
    set_holiday_provider(None)


def test_1_valid_response_mapping():
    provider = NagerDateHolidayProvider(base_url="https://date.nager.at/api/v4")

    mock_data = [
        {
            "date": "2026-01-07",
            "localName": "Coptic Christmas Day",
            "name": "Coptic Christmas Day",
            "countryCode": "EG",
            "fixed": True,
            "global": True,
            "types": ["Public"],
        }
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_data

    with patch("httpx.Client.get", return_value=mock_resp):
        holidays = provider.get_holidays("EG", 2026)
        assert len(holidays) == 1
        h = holidays[0]
        assert h.date == date(2026, 1, 7)
        assert h.name == "Coptic Christmas Day"
        assert h.country_code == "EG"
        assert h.national_holiday is True
        assert h.holiday_types == ["Public"]


def test_2_multiple_holidays():
    provider = NagerDateHolidayProvider()

    mock_data = [
        {"date": "2026-01-07", "name": "Coptic Christmas Day", "countryCode": "EG", "types": ["Public"]},
        {"date": "2026-01-25", "name": "Revolution Day", "countryCode": "EG", "types": ["Public"]},
    ]
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = mock_data

    with patch("httpx.Client.get", return_value=mock_resp):
        holidays = provider.get_holidays("EG", 2026)
        assert len(holidays) == 2
        assert holidays[0].name == "Coptic Christmas Day"
        assert holidays[1].name == "Revolution Day"


def test_3_country_code_propagation():
    provider = NagerDateHolidayProvider(base_url="https://date.nager.at/api/v4")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = []

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        provider.get_holidays("eg", 2026)
        url = mock_get.call_args[0][0]
        assert url == "https://date.nager.at/api/v4/Holidays/EG/2026"


def test_4_year_propagation():
    provider = NagerDateHolidayProvider(base_url="https://date.nager.at/api/v4")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = []

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        provider.get_holidays("EG", 2027)
        url = mock_get.call_args[0][0]
        assert url == "https://date.nager.at/api/v4/Holidays/EG/2027"


def test_5_same_year_date_window():
    provider = NagerDateHolidayProvider()
    from_t = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    to_t = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = [
        {"date": "2026-01-07", "name": "Holiday 1", "countryCode": "EG", "types": ["Public"]}
    ]

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        result = provider.get_holidays_for_range("EG", from_t, to_t)
        assert mock_get.call_count == 1
        assert len(result.holidays) == 1


def test_6_cross_year_date_window_fetches_both_years():
    provider = NagerDateHolidayProvider()
    from_t = datetime(2025, 12, 25, 0, 0, tzinfo=timezone.utc)
    to_t = datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc)

    mock_resp_2025 = MagicMock(status_code=200)
    mock_resp_2025.json.return_value = [
        {"date": "2025-12-31", "name": "New Year's Eve", "countryCode": "EG", "types": ["Public"]}
    ]
    mock_resp_2026 = MagicMock(status_code=200)
    mock_resp_2026.json.return_value = [
        {"date": "2026-01-07", "name": "Coptic Christmas Day", "countryCode": "EG", "types": ["Public"]}
    ]

    def mock_get(url, **kwargs):
        if "2025" in url:
            return mock_resp_2025
        return mock_resp_2026

    with patch("httpx.Client.get", side_effect=mock_get):
        result = provider.get_holidays_for_range("EG", from_t, to_t)
        assert len(result.holidays) == 2
        assert result.holidays[0].date == date(2025, 12, 31)
        assert result.holidays[1].date == date(2026, 1, 7)


def test_7_local_filtering_by_requested_date_range():
    provider = NagerDateHolidayProvider()
    from_t = datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc)
    to_t = datetime(2026, 1, 20, 0, 0, tzinfo=timezone.utc)

    mock_data = [
        {"date": "2026-01-07", "name": "Coptic Christmas Day", "countryCode": "EG"},
        {"date": "2026-01-15", "name": "Mid-Month Holiday", "countryCode": "EG"},
        {"date": "2026-01-25", "name": "Revolution Day", "countryCode": "EG"},
    ]
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = mock_data

    with patch("httpx.Client.get", return_value=mock_resp):
        result = provider.get_holidays_for_range("EG", from_t, to_t)
        assert len(result.holidays) == 1
        assert result.holidays[0].name == "Mid-Month Holiday"


def test_8_empty_holiday_result():
    provider = NagerDateHolidayProvider()
    from_t = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    to_t = datetime(2026, 2, 10, 0, 0, tzinfo=timezone.utc)

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = []

    with patch("httpx.Client.get", return_value=mock_resp):
        result = provider.get_holidays_for_range("EG", from_t, to_t)
        assert isinstance(result, HolidayContext)
        assert result.holidays == []


def test_9_http_4xx_raises_tool_error():
    provider = NagerDateHolidayProvider()
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found", request=MagicMock(), response=mock_resp
    )

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(LocalEventsToolError, match="HTTP status 404"):
            provider.get_holidays("EG", 2026)


def test_10_http_5xx_raises_tool_error():
    provider = NagerDateHolidayProvider()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Server Error", request=MagicMock(), response=mock_resp
    )

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(LocalEventsToolError, match="HTTP status 500"):
            provider.get_holidays("EG", 2026)


def test_11_timeout_raises_tool_error():
    provider = NagerDateHolidayProvider()

    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Connection timeout")):
        with pytest.raises(LocalEventsToolError, match="timed out"):
            provider.get_holidays("EG", 2026)


def test_12_malformed_json_raises_tool_error():
    provider = NagerDateHolidayProvider()
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"not_a": "list"}

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(LocalEventsToolError, match="expected a JSON list"):
            provider.get_holidays("EG", 2026)


def test_13_missing_required_fields_raises_tool_error():
    provider = NagerDateHolidayProvider()
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = [{"countryCode": "EG"}]  # Missing date and name

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(LocalEventsToolError, match="missing required fields"):
            provider.get_holidays("EG", 2026)


def test_14_invalid_holiday_date_raises_tool_error():
    provider = NagerDateHolidayProvider()
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = [{"date": "invalid-date", "name": "Test", "countryCode": "EG"}]

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(LocalEventsToolError, match="invalid date format"):
            provider.get_holidays("EG", 2026)


def test_15_provider_factory_selects_nager_date_provider():
    with patch.object(settings, "EVENTS_PROVIDER", "nager_date"):
        prov = get_holiday_provider()
        assert isinstance(prov, NagerDateHolidayProvider)


def test_16_mock_provider_remains_available():
    with patch.object(settings, "EVENTS_PROVIDER", "mock"):
        prov = get_holiday_provider()
        assert isinstance(prov, MockHolidayProvider)


@pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_INTEGRATION_TESTS") != "true",
    reason="Live integration test requires RUN_EXTERNAL_INTEGRATION_TESTS=true",
)
def test_real_nager_date_live_integration():
    """Opt-in live integration test against real Nager.Date API."""
    provider = NagerDateHolidayProvider(base_url="https://date.nager.at/api/v4")
    current_year = datetime.now(timezone.utc).year

    holidays = provider.get_holidays("EG", current_year)
    assert isinstance(holidays, list)
    if len(holidays) > 0:
        h = holidays[0]
        assert isinstance(h.date, date)
        assert isinstance(h.name, str)
        assert h.country_code == "EG"
