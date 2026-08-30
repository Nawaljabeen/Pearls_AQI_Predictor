"""
pipelines/backfill_data.py

Runs feature_pipeline.py over the full historical range and pushes the
result to Hopsworks as a feature group 

"""

from datetime import datetime, timezone
from feature_pipeline import run_feature_pipeline

BACKFILL_START = datetime(2023, 11, 1, tzinfo=timezone.utc)


def run_backfill():
    end_dt = datetime.now(timezone.utc)
    start_dt = BACKFILL_START

    print(f"Starting backfill: {start_dt.date()} -> {end_dt.date()}")

    features = run_feature_pipeline(
        start_dt=start_dt,
        end_dt=end_dt,
        push=True,
    )
    
    if features.empty:
        print("\nBackfill failed — no data produced.")
    else:
        print(f"\nBackfill complete: {features.shape[0]} rows ingested.")


if __name__ == "__main__":
    run_backfill()