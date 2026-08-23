"""
pipelines/backfill_data.py

One time historical backfill for the city level base
pipeline. Pulls the full available range for each citys base station chain
(live + dead, when available) and pushes it into hopsworks.


"""

from datetime import datetime, timezone
from feature_pipeline import run_feature_pipeline

# Earliest real data across our base stations is StateAir's 2019-05-22.
# Set this to that date (or later, if you don't want the dead-station depth).
BACKFILL_START = datetime(2023, 11, 1, tzinfo=timezone.utc)


def run_backfill(include_dead_station=False):
    end_dt = datetime.now(timezone.utc)
    start_dt = BACKFILL_START

    print(f" starting backfill: {start_dt.date()} -> {end_dt.date()}")
    print(f"   Include dead/historical stations: {include_dead_station}")

    features = run_feature_pipeline(
        start_dt=start_dt,
        end_dt=end_dt,
        include_dead_station=include_dead_station,
        push=True,
    )

    if features.empty:
        print("\n Backfill failed — no data produced.")
    else:
        print(f"\n Backfill complete: {features.shape[0]} rows ingested.")


if __name__ == "__main__":
    run_backfill()