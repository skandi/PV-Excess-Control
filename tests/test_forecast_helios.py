"""Helios daily totals use local calendar days without truncating the timeline."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from custom_components.pv_excess_control.forecast import (
    AggregatingForecastProvider,
    HeliosForecastProvider,
)

LOCAL = ZoneInfo("Europe/Zurich")
ENTITY = "sensor.helios_forecast_power_now"


def states(points):
    return {
        ENTITY: {
            "state": "0",
            "attributes": {
                "unit_of_measurement": "W",
                "forecast": [
                    {"datetime": start.isoformat(), "watts": watts}
                    for start, watts in points
                ],
            },
        }
    }


@pytest.mark.parametrize("aggregate", [False, True])
def test_daily_totals_exclude_other_days_but_keep_full_timeline(aggregate):
    now = datetime(2026, 10, 21, 20, tzinfo=LOCAL)
    points = [
        (now - timedelta(minutes=15), 9000),
        (now, 2000),
        (datetime(2026, 10, 22, 12, tzinfo=LOCAL), 3000),
        (datetime(2026, 10, 23, 12, tzinfo=LOCAL), 4000),
    ]
    provider = (
        AggregatingForecastProvider("helios_forecast", [ENTITY])
        if aggregate else HeliosForecastProvider(ENTITY)
    )
    data = provider.get_forecast(states(points), now=now)

    assert data.remaining_today_kwh == pytest.approx(0.5)
    assert data.tomorrow_total_kwh == pytest.approx(0.75)
    assert [row.expected_watts for row in data.hourly_breakdown] == [
        watts for _, watts in points
    ]
    assert [row.start for row in data.hourly_breakdown] == [
        start.astimezone(timezone.utc) for start, _ in points
    ]
    assert all(
        row.end - row.start == timedelta(minutes=15)
        for row in data.hourly_breakdown
    )


def test_after_sunset_remaining_today_is_zero_despite_future_production():
    now = datetime(2026, 10, 21, 20, tzinfo=LOCAL)
    points = [
        (now, 0),
        (now + timedelta(minutes=15), 0),
        (now + timedelta(days=1), 2000),
    ]
    data = HeliosForecastProvider(ENTITY).get_forecast(states(points), now=now)

    assert data.remaining_today_kwh == 0
    assert data.tomorrow_total_kwh == pytest.approx(0.5)


def test_current_interval_counts_only_its_remaining_fraction():
    start = datetime(2026, 10, 21, 12, tzinfo=LOCAL)
    data = HeliosForecastProvider(ENTITY).get_forecast(
        states([(start, 2000)]), now=start + timedelta(minutes=7, seconds=30)
    )

    assert data.remaining_today_kwh == pytest.approx(0.25)
    assert data.hourly_breakdown[0].expected_kwh == pytest.approx(0.5)


def test_interval_crossing_midnight_is_split_between_daily_totals():
    start = datetime(2026, 10, 21, 23, 55, tzinfo=LOCAL)
    data = HeliosForecastProvider(ENTITY).get_forecast(
        states([(start, 2000)]), now=start
    )

    assert data.remaining_today_kwh == pytest.approx(2000 * 5 / 60000)
    assert data.tomorrow_total_kwh == pytest.approx(2000 * 10 / 60000)
    assert len(data.hourly_breakdown) == 1
    assert data.hourly_breakdown[0].expected_kwh == pytest.approx(0.5)


def test_local_date_is_used_when_utc_is_still_on_previous_day():
    now = datetime(2026, 10, 22, 0, 30, tzinfo=LOCAL)
    points = [
        (now, 1000),
        (now + timedelta(minutes=15), 0),
        (datetime(2026, 10, 23, 0, tzinfo=LOCAL), 2000),
        (datetime(2026, 10, 24, 0, tzinfo=LOCAL), 9000),
    ]
    data = HeliosForecastProvider(ENTITY).get_forecast(states(points), now=now)

    assert data.remaining_today_kwh == pytest.approx(0.25)
    assert data.tomorrow_total_kwh == pytest.approx(0.5)


def test_tomorrow_total_clips_interval_crossing_its_end():
    now = datetime(2026, 10, 21, 12, tzinfo=LOCAL)
    points = [
        (now, 0),
        (now + timedelta(minutes=15), 0),
        (datetime(2026, 10, 22, 23, 55, tzinfo=LOCAL), 2000),
    ]
    data = HeliosForecastProvider(ENTITY).get_forecast(states(points), now=now)

    assert data.remaining_today_kwh == 0
    assert data.tomorrow_total_kwh == pytest.approx(2000 * 5 / 60000)


@pytest.mark.parametrize(
    "day,hours", [(datetime(2026, 3, 29), 23), (datetime(2026, 10, 25), 25)]
)
@pytest.mark.parametrize("tomorrow", [False, True])
def test_dst_calendar_days_use_actual_elapsed_hours(day, hours, tomorrow):
    start = day.replace(tzinfo=LOCAL)
    end = (day + timedelta(days=1)).replace(tzinfo=LOCAL)
    first = start.astimezone(timezone.utc)
    last = end.astimezone(timezone.utc)
    points = [
        (first + timedelta(minutes=15 * index), 1000)
        for index in range(int((last - first).total_seconds() / 900))
    ]
    # Out-of-day production must not leak into the daily total.
    points.append((last, 9000))
    now = (day - timedelta(days=1)).replace(hour=12, tzinfo=LOCAL) if tomorrow else start
    data = HeliosForecastProvider(ENTITY).get_forecast(states(points), now=now)

    total = data.tomorrow_total_kwh if tomorrow else data.remaining_today_kwh
    assert total == pytest.approx(hours)


@pytest.mark.parametrize("has_tomorrow", [False, True])
def test_zero_tomorrow_is_distinct_from_missing_tomorrow(has_tomorrow):
    now = datetime(2026, 10, 21, 12, tzinfo=LOCAL)
    points = [(now, 0)]
    if has_tomorrow:
        points.append((now + timedelta(days=1), 0))
    data = HeliosForecastProvider(ENTITY).get_forecast(states(points), now=now)

    assert data.remaining_today_kwh == 0
    assert data.tomorrow_total_kwh == (0 if has_tomorrow else None)


def test_naive_now_keeps_legacy_utc_interpretation():
    now = datetime(2026, 10, 21, 12)
    points = [
        (now, 2000),
        (now + timedelta(minutes=15), 0),
        (now + timedelta(days=1), 4000),
    ]
    data = HeliosForecastProvider(ENTITY).get_forecast(states(points), now=now)

    assert data.remaining_today_kwh == pytest.approx(0.5)
    assert data.tomorrow_total_kwh == pytest.approx(1)
