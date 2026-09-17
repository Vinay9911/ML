"""Shared fixtures for the twin_common test suite."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from twin_common.contracts import IST, Metadata, ModelOutput, PredictRequest, ScenarioRequest
from twin_common.contracts.enums import EntityType, State
from twin_common.model import TwinModel
from twin_common.output import OutputBuilder

#: The demo instant from docs/04 section 3.
DEMO_NOW = datetime(2027, 8, 2, 6, 0, tzinfo=IST)

#: A dummy model that owns exactly the M21 weather KPIs, used to test the API factory
#: without depending on Phase 3 having built M21 yet.
DUMMY_CONFIG: dict[str, Any] = {
    "model_id": "M21",
    "model_name": "Weather Impact (test dummy)",
    "model_version": "0.1.0",
    "port": 8021,
    "engine": "formula",
    "data_source": "synthetic",
    "seed": 42,
    "demo_now": "2027-08-02T06:00:00+05:30",
    "horizons_min": [15, 60, 180],
    "default_horizon_min": 180,
    "upstream": [],
    "inputs": ["weather_hourly"],
    "kpis": ["temperature", "heat_index", "rainfall_intensity", "waterlogging_probability"],
    "scenarios_supported": ["S01", "S03", "S04"],
    "upstream_timeout_s": 5,
    "params": {
        "base_temperature_c": 31.0,
        "base_humidity_pct": 55.0,
        "base_rain_mm_hr": 0.0,
        "waterlogging_base_pct": 5.0,
    },
}


class DummyModel(TwinModel):
    """Deterministic stand-in: constant values driven entirely by params."""

    model_id = "M21"

    #: Low-lying zones get the elevated waterlogging probability.
    ZONES = ("Z01", "Z02", "Z03")
    LOW_LYING = frozenset({"Z01", "Z02"})

    def metadata(self) -> Metadata:
        from twin_common.contracts import registry as reg

        return Metadata(
            model_id=self.model_id,
            model_name=self.model_name,
            model_version=self.model_version,
            question="What is the weather now and next hours (test dummy)?",
            engine=self.engine,
            method_summary="Constant values from config params; no real computation.",
            kpis=[
                {"kpi": k, "unit": reg.kpi_unit(k), "description": reg.kpi_info(k).definition}
                for k in self.owned_kpis
            ],
            upstream=self.upstream_ids,
            inputs=self.input_tables,
            horizons_min=self.horizons_min,
            scenarios_supported=self.scenarios_supported,
            limitations=["Test dummy: values are constants, not a weather model."],
            license_notes=[],
        )

    def _fill(
        self,
        builder: OutputBuilder,
        request: PredictRequest,
        *,
        state: State,
        overrides: dict[str, Any] | None = None,
    ) -> OutputBuilder:
        overrides = overrides or {}
        as_of = self.resolve_as_of(request)
        kpis = self.resolve_kpis(request)
        zones = self.resolve_entities(request, list(self.ZONES))
        temp = self.require_param("base_temperature_c") + float(
            overrides.get("temperature_offset_c", 0.0)
        )
        rain = float(overrides.get("rain_mm_hr", self.require_param("base_rain_mm_hr")))
        horizon = 0 if state is State.CURRENT else self.resolve_horizon(request)

        if "temperature" in kpis:
            builder.add(
                EntityType.EVENT,
                "EVENT",
                "temperature",
                as_of,
                temp,
                horizon_min=horizon,
                state=state,
            )
        if "heat_index" in kpis:
            # Deliberately simple: the real NOAA algorithm arrives with M21 in Phase 3.
            builder.add(
                EntityType.EVENT,
                "EVENT",
                "heat_index",
                as_of,
                temp + 2.0,
                horizon_min=horizon,
                state=state,
            )
        if "rainfall_intensity" in kpis:
            builder.add(
                EntityType.EVENT,
                "EVENT",
                "rainfall_intensity",
                as_of,
                rain,
                horizon_min=horizon,
                state=state,
            )
        if "waterlogging_probability" in kpis:
            base = self.require_param("waterlogging_base_pct")
            for zone in zones:
                value = min(100.0, base + (rain if zone in self.LOW_LYING else rain * 0.5))
                builder.add(
                    EntityType.ZONE,
                    zone,
                    "waterlogging_probability",
                    as_of,
                    value,
                    horizon_min=horizon,
                    state=state,
                    zone_id=zone,
                    reason_codes=["WATERLOGGING"] if value >= 20 else [],
                )
        return builder

    def predict(self, request: PredictRequest) -> ModelOutput:
        builder = self.new_builder(request)
        self._fill(builder, request, state=State.FORECAST)
        return builder.build()

    def scenario(self, request: ScenarioRequest) -> ModelOutput:
        overrides = self.scenario_overrides(request)
        builder = self.new_builder(request, scenario_id=request.scenario_id, overrides=overrides)
        self._fill(builder, request, state=State.SCENARIO, overrides=overrides)
        if request.compare_to_baseline:
            builder.apply_baseline(self.predict(PredictRequest(**_predict_fields(request))))
        return builder.build()


def _predict_fields(request: ScenarioRequest) -> dict[str, Any]:
    return request.model_dump(
        include={"as_of", "horizon_min", "entity_ids", "kpis", "include_current", "upstream"}
    )


@pytest.fixture
def demo_now() -> datetime:
    return DEMO_NOW


@pytest.fixture
def dummy_folder(tmp_path: Path) -> Path:
    """A minimal model folder with a valid config.yaml."""
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(DUMMY_CONFIG, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "data" / "sample_upstream").mkdir(parents=True)
    (tmp_path / "outputs").mkdir()
    return tmp_path


@pytest.fixture
def dummy_model(dummy_folder: Path) -> DummyModel:
    return DummyModel.from_folder(dummy_folder)


@pytest.fixture
def dummy_app(dummy_folder: Path):
    from twin_common.api import create_app

    return create_app(DummyModel, root=dummy_folder)
