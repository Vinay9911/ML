"""CLI for M20: run the model once and write the output as JSON.

    uv run python -m src.run --as-of "2027-08-02T06:00:00+05:30" --out outputs/
    uv run python -m src.run --scenario S02 --out outputs/

Used by ``scripts/refresh_samples.py`` to regenerate the sample outputs downstream models
read, and by a human who wants to look at the JSON without starting a server.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from twin_common.contracts import PredictRequest, ScenarioRequest
from twin_common.contracts.models import to_ist

from .model import Model

FOLDER = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.run", description="Run M20.")
    parser.add_argument("--as-of", default=None, help="ISO timestamp; defaults to demo_now")
    parser.add_argument("--scenario", default=None, help="scenario ID, e.g. S02")
    parser.add_argument("--horizon-min", type=int, default=None)
    parser.add_argument("--entity", action="append", default=None, help="repeatable")
    parser.add_argument("--out", type=Path, default=None, help="directory to write JSON into")
    args = parser.parse_args(argv)

    model = Model.from_folder(FOLDER)
    common = {
        "as_of": to_ist(datetime.fromisoformat(args.as_of)) if args.as_of else None,
        "horizon_min": args.horizon_min,
        "entity_ids": args.entity,
    }
    if args.scenario:
        output = model.scenario(ScenarioRequest(scenario_id=args.scenario, **common))
        name = f"sample_scenario_{args.scenario}.json"
    else:
        output = model.predict(PredictRequest(**common))
        name = "sample_output.json"

    payload = output.model_dump(mode="json")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{output.model_id}: {len(output.results)} records -> {path}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
