"""World sanity report (docs/04 section 5.8).

Writes a self-contained ``00_common/reports/world_<scenario>.html`` showing arrivals, zone
populations, density against the risk bands, medical cases and weather - the plots a human
looks at to decide whether a generated world is believable.

**No plotting dependency.** matplotlib and plotly are not in the docs/06 section 1 approved
list, and CLAUDE.md says to ask before adding anything that is not. docs/04 section 5.8
allows a report that is free of them, so the charts here are hand-built inline SVG. That
also keeps the report a single file with no assets, which is easier to email to a reviewer.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import assumptions
from ..logging import get_logger
from ..paths import ensure_dir, reports_dir

log = get_logger(__name__)

#: Chart geometry, in SVG user units.
WIDTH = 920
HEIGHT = 260
PAD_LEFT = 64
PAD_RIGHT = 18
PAD_TOP = 18
PAD_BOTTOM = 40

#: Risk band colours, matching the operational vocabulary.
BAND_COLOURS = {
    "green": "#2f9e44",
    "amber": "#f08c00",
    "red": "#e03131",
    "critical": "#862e9c",
}
#: A colour-blind-safe qualitative ramp for the per-zone series.
SERIES_COLOURS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#7f7f7f",
)


@dataclass(frozen=True, slots=True)
class Series:
    """One labelled line on a chart."""

    label: str
    x: Sequence[float]
    y: Sequence[float]
    colour: str


def _scale(
    values: Sequence[float], lo: float, hi: float, out_lo: float, out_hi: float
) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [out_lo for _ in values]
    return [out_lo + (value - lo) / span * (out_hi - out_lo) for value in values]


def _line_chart(
    title: str,
    series: list[Series],
    *,
    y_label: str = "",
    bands: Mapping[str, tuple[float, float]] | None = None,
    x_ticks: Sequence[tuple[float, str]] | None = None,
    y_max: float | None = None,
) -> str:
    """One inline SVG line chart, optionally with horizontal risk bands behind it."""
    if not series or not any(len(s.y) for s in series):
        return f"<p class='empty'>{html.escape(title)}: no data</p>"

    all_x = [value for s in series for value in s.x]
    all_y = [value for s in series for value in s.y]
    x_lo, x_hi = min(all_x), max(all_x)
    y_lo = 0.0
    y_hi = float(y_max) if y_max is not None else max(all_y) * 1.08
    if bands:
        y_hi = max(y_hi, max(hi for _, hi in bands.values() if np.isfinite(hi)) * 1.02)
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0

    plot_w = WIDTH - PAD_LEFT - PAD_RIGHT
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM
    parts: list[str] = [
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" width="100%" role="img" '
        f'aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="var(--panel)"/>',
    ]

    # Risk bands as translucent horizontal stripes.
    if bands:
        for name, (lo, hi) in bands.items():
            hi_clipped = min(hi, y_hi)
            if hi_clipped <= lo:
                continue
            top = PAD_TOP + plot_h - (hi_clipped - y_lo) / (y_hi - y_lo) * plot_h
            bottom = PAD_TOP + plot_h - (lo - y_lo) / (y_hi - y_lo) * plot_h
            parts.append(
                f'<rect x="{PAD_LEFT}" y="{top:.1f}" width="{plot_w}" '
                f'height="{max(0.0, bottom - top):.1f}" fill="{BAND_COLOURS.get(name, "#ccc")}" '
                f'opacity="0.12"/>'
            )

    # Axes and gridlines.
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = PAD_TOP + plot_h - fraction * plot_h
        value = y_lo + fraction * (y_hi - y_lo)
        parts.append(
            f'<line x1="{PAD_LEFT}" y1="{y:.1f}" x2="{WIDTH - PAD_RIGHT}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_LEFT - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{_format_number(value)}</text>'
        )
    if x_ticks:
        for position, label in x_ticks:
            x = _scale([position], x_lo, x_hi, PAD_LEFT, WIDTH - PAD_RIGHT)[0]
            parts.append(
                f'<line x1="{x:.1f}" y1="{PAD_TOP + plot_h}" x2="{x:.1f}" '
                f'y2="{PAD_TOP + plot_h + 4}" stroke="var(--grid)"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{HEIGHT - PAD_BOTTOM + 20}" text-anchor="middle" '
                f'class="tick">{html.escape(label)}</text>'
            )

    # The data.
    for line in series:
        if not len(line.y):
            continue
        xs = _scale(line.x, x_lo, x_hi, PAD_LEFT, WIDTH - PAD_RIGHT)
        ys = _scale(line.y, y_lo, y_hi, PAD_TOP + plot_h, PAD_TOP)
        points = " ".join(
            f"{x:.1f},{min(max(y, PAD_TOP), PAD_TOP + plot_h):.1f}"
            for x, y in zip(xs, ys, strict=True)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{line.colour}" '
            f'stroke-width="1.6" stroke-linejoin="round"/>'
        )

    if y_label:
        parts.append(
            f'<text x="14" y="{PAD_TOP + plot_h / 2:.0f}" class="axis-label" '
            f'transform="rotate(-90 14 {PAD_TOP + plot_h / 2:.0f})" '
            f'text-anchor="middle">{html.escape(y_label)}</text>'
        )
    parts.append("</svg>")

    legend = " ".join(
        f'<span class="key"><i style="background:{s.colour}"></i>{html.escape(s.label)}</span>'
        for s in series
    )
    return (
        f"<figure><figcaption>{html.escape(title)}</figcaption>"
        f'{"".join(parts)}<div class="legend">{legend}</div></figure>'
    )


def _format_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _band_ranges(kind: str) -> dict[str, tuple[float, float]]:
    """Density bands as (lo, hi) pairs for the chart background."""
    bands = assumptions()["crowd"]["density_bands_p_m2"]
    if kind != "density":
        return {}
    return {
        "green": (0.0, float(bands["amber"])),
        "amber": (float(bands["amber"]), float(bands["red"])),
        "red": (float(bands["red"]), float(bands["critical"])),
        "critical": (float(bands["critical"]), float(bands["critical"]) * 1.6),
    }


def _day_ticks(stamps: pd.DatetimeIndex, every: int = 4) -> list[tuple[float, str]]:
    """Tick positions at midnight, thinned so the axis stays readable."""
    ticks: list[tuple[float, str]] = []
    seen: set[str] = set()
    for position, stamp in enumerate(stamps):
        day = stamp.strftime("%m-%d")
        if stamp.hour == 0 and stamp.minute == 0 and day not in seen:
            seen.add(day)
            ticks.append((float(position), day))
    return ticks[:: max(1, every)]


def build_report(
    tables: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    *,
    scenario_id: str,
    out_dir: Path | None = None,
) -> Path:
    """Write the sanity report and return its path."""
    footfall = tables["footfall_15min"]
    weather = tables["weather_hourly"]
    medical = tables.get("medical_incidents", pd.DataFrame())
    gates = tables.get("gate_entries", pd.DataFrame())

    population = footfall.pivot_table(
        index="timestamp", columns="zone_id", values="population", aggfunc="sum"
    ).fillna(0.0)
    density = footfall.pivot_table(
        index="timestamp", columns="zone_id", values="density_p_m2", aggfunc="sum"
    ).fillna(0.0)
    stamps = pd.DatetimeIndex(population.index)
    ticks = _day_ticks(stamps)
    positions = list(range(len(stamps)))

    charts: list[str] = []

    # Arrivals admitted per gate.
    if len(gates):
        gate_pivot = gates.pivot_table(
            index="timestamp", columns="gate_id", values="entries", aggfunc="sum"
        ).fillna(0.0)
        charts.append(
            _line_chart(
                "Arrivals admitted per gate (persons per 15 min)",
                [
                    Series(
                        str(gate),
                        positions,
                        gate_pivot[gate].tolist(),
                        SERIES_COLOURS[index % len(SERIES_COLOURS)],
                    )
                    for index, gate in enumerate(gate_pivot.columns)
                ],
                y_label="persons / 15 min",
                x_ticks=ticks,
            )
        )
        if "queue_persons" in gates.columns:
            queue_pivot = gates.pivot_table(
                index="timestamp", columns="gate_id", values="queue_persons", aggfunc="sum"
            ).fillna(0.0)
            if float(queue_pivot.to_numpy().max()) > 0:
                charts.append(
                    _line_chart(
                        "Queue waiting at each gate (persons)",
                        [
                            Series(
                                str(gate),
                                positions,
                                queue_pivot[gate].tolist(),
                                SERIES_COLOURS[index % len(SERIES_COLOURS)],
                            )
                            for index, gate in enumerate(queue_pivot.columns)
                        ],
                        y_label="persons waiting",
                        x_ticks=ticks,
                    )
                )

    charts.append(
        _line_chart(
            "Zone population (persons)",
            [
                Series(
                    str(zone),
                    positions,
                    population[zone].tolist(),
                    SERIES_COLOURS[index % len(SERIES_COLOURS)],
                )
                for index, zone in enumerate(population.columns)
            ],
            y_label="persons",
            x_ticks=ticks,
        )
    )
    charts.append(
        _line_chart(
            "Crowd density against the risk bands (persons/m2)",
            [
                Series(
                    str(zone),
                    positions,
                    density[zone].tolist(),
                    SERIES_COLOURS[index % len(SERIES_COLOURS)],
                )
                for index, zone in enumerate(density.columns)
            ],
            y_label="persons / m2",
            bands=_band_ranges("density"),
            x_ticks=ticks,
        )
    )

    if len(medical):
        per_step = (
            medical.set_index("timestamp")
            .resample(f"{int((stamps[1] - stamps[0]).total_seconds() // 60)}min")
            .size()
            .reindex(stamps, fill_value=0)
        )
        charts.append(
            _line_chart(
                "Medical presentations (cases per 15 min)",
                [Series("cases", positions, per_step.tolist(), "#d62728")],
                y_label="cases / 15 min",
                x_ticks=ticks,
            )
        )

    weather_positions = list(range(len(weather)))
    weather_ticks = _day_ticks(pd.DatetimeIndex(weather["timestamp"]), every=4)
    charts.append(
        _line_chart(
            "Weather: temperature and heat index (degC)",
            [
                Series(
                    "temperature", weather_positions, weather["temperature_c"].tolist(), "#ff7f0e"
                ),
                Series(
                    "heat index", weather_positions, weather["heat_index_c"].tolist(), "#d62728"
                ),
            ],
            y_label="degC",
            x_ticks=weather_ticks,
        )
    )
    charts.append(
        _line_chart(
            "Weather: rainfall (mm/hr)",
            [Series("rain", weather_positions, weather["rain_mm"].tolist(), "#1f77b4")],
            y_label="mm / hr",
            x_ticks=weather_ticks,
        )
    )

    # ------------------------------------------------------------------ summary
    peak_density = density.max().sort_values(ascending=False)
    bands = assumptions()["crowd"]["density_bands_p_m2"]

    def band_of(value: float) -> str:
        if value >= float(bands["critical"]):
            return "critical"
        if value >= float(bands["red"]):
            return "red"
        if value >= float(bands["amber"]):
            return "amber"
        return "green"

    zone_rows = "".join(
        f"<tr><td>{html.escape(str(zone))}</td>"
        f"<td class='num'>{population[zone].max():,.0f}</td>"
        f"<td class='num'>{value:.2f}</td>"
        f"<td><span class='pill {band_of(value)}'>{band_of(value)}</span></td></tr>"
        for zone, value in peak_density.items()
    )

    validation = manifest.get("validation", {})
    sources = manifest.get("data_sources", {})
    arrivals = manifest.get("arrivals", {})
    table_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class='num'>{info['rows']:,}</td>"
        f"<td class='hash'>{info['sha256'][:12]}</td></tr>"
        for name, info in sorted(manifest.get("tables", {}).items())
    )
    warnings_html = (
        "".join(f"<li>{html.escape(str(w))}</li>" for w in validation.get("warnings", []))
        or "<li>none</li>"
    )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>World sanity report - {html.escape(scenario_id)}</title>
<style>
  :root {{
    --bg: #ffffff; --panel: #fbfbfd; --ink: #1a1a1a; --muted: #5b6270;
    --grid: #dfe3ea; --border: #e4e7ee;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #14161a; --panel: #1b1e24; --ink: #eceef2; --muted: #9aa3b2;
      --grid: #2c313a; --border: #2c313a;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #14161a; --panel: #1b1e24; --ink: #eceef2; --muted: #9aa3b2;
    --grid: #2c313a; --border: #2c313a;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 0 16px 64px; background: var(--bg); color: var(--ink);
         font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin: 28px 0 4px; }}
  h2 {{ font-size: 1.05rem; margin: 32px 0 10px; padding-bottom: 6px;
        border-bottom: 1px solid var(--border); }}
  .sub {{ color: var(--muted); margin: 0 0 8px; }}
  .synthetic {{ display: inline-block; background: #862e9c; color: #fff; font-weight: 600;
                border-radius: 5px; padding: 3px 9px; font-size: .78rem; letter-spacing: .03em; }}
  figure {{ margin: 0 0 26px; border: 1px solid var(--border); border-radius: 10px;
            overflow: hidden; background: var(--panel); }}
  figcaption {{ padding: 9px 13px; font-weight: 600; font-size: .9rem;
                border-bottom: 1px solid var(--border); }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 9px 13px;
             border-top: 1px solid var(--border); font-size: .8rem; color: var(--muted); }}
  .key {{ display: inline-flex; align-items: center; gap: 5px; }}
  .key i {{ width: 11px; height: 3px; border-radius: 2px; display: inline-block; }}
  .tick {{ font-size: 10px; fill: var(--muted); }}
  .axis-label {{ font-size: 11px; fill: var(--muted); }}
  table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
  th, td {{ text-align: left; padding: 6px 9px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.hash {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
             font-size: .78rem; color: var(--muted); }}
  .pill {{ border-radius: 999px; padding: 1px 9px; font-size: .76rem; color: #fff; }}
  .pill.green {{ background: {BAND_COLOURS["green"]}; }}
  .pill.amber {{ background: {BAND_COLOURS["amber"]}; }}
  .pill.red {{ background: {BAND_COLOURS["red"]}; }}
  .pill.critical {{ background: {BAND_COLOURS["critical"]}; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }}
  @media (max-width: 700px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  .empty {{ color: var(--muted); font-style: italic; }}
  ul {{ margin: 6px 0; padding-left: 20px; color: var(--muted); font-size: .88rem; }}
  footer {{ margin-top: 36px; padding-top: 14px; border-top: 1px solid var(--border);
            color: var(--muted); font-size: .8rem; }}
</style>
</head>
<body><div class="wrap">
  <h1>World sanity report - {html.escape(scenario_id)}</h1>
  <p class="sub">{html.escape(str(manifest.get("scenario_name", "")))} &middot;
     seed {manifest.get("seed")} &middot;
     {manifest.get("time_range", {}).get("days", "?")} days &middot;
     generated {html.escape(str(manifest.get("generated_at", "")))}</p>
  <p><span class="synthetic">SYNTHETIC DATA</span></p>

  <h2>Validation</h2>
  <p class="sub">{validation.get("passed", 0)} checks passed,
     {validation.get("failed", 0)} failed (docs/04 section 8).</p>
  <ul>{warnings_html}</ul>

  <h2>Charts</h2>
  {"".join(charts)}

  <div class="grid2">
    <div>
      <h2>Peak per zone</h2>
      <table><thead><tr><th>Zone</th><th class="num">Peak population</th>
        <th class="num">Peak density</th><th>Band</th></tr></thead>
        <tbody>{zone_rows}</tbody></table>
    </div>
    <div>
      <h2>Run</h2>
      <table><tbody>
        <tr><td>Weather source</td><td>{html.escape(str(sources.get("weather", "?")))}</td></tr>
        <tr><td>Weather is real</td><td>{sources.get("weather_is_real")}</td></tr>
        <tr><td>Air quality source</td>
            <td>{html.escape(str(sources.get("air_quality", "?")))}</td></tr>
        <tr><td>Scenario changed weather</td>
            <td>{sources.get("weather_scenario_modified")}</td></tr>
        <tr><td>Arrivals admitted</td>
            <td class="num">{arrivals.get("admitted_total", 0):,.0f}</td></tr>
        <tr><td>Arrivals demanded</td>
            <td class="num">{arrivals.get("demanded_total", 0):,.0f}</td></tr>
        <tr><td>Offline run</td><td>{manifest.get("offline")}</td></tr>
        <tr><td>Elapsed</td><td class="num">{manifest.get("elapsed_s", 0)} s</td></tr>
      </tbody></table>
    </div>
  </div>

  <h2>Tables</h2>
  <table><thead><tr><th>Table</th><th class="num">Rows</th><th>sha256</th></tr></thead>
    <tbody>{table_rows}</tbody></table>

  <footer>
    Every value in this report is synthetic and generated from placeholder assumptions.
    {html.escape(" ".join(manifest.get("attribution", [])))}
  </footer>
</div></body></html>
"""

    target = ensure_dir(Path(out_dir) if out_dir is not None else reports_dir())
    path = target / f"world_{scenario_id}.html"
    path.write_text(document, encoding="utf-8")
    log.info("sanity report written to %s (%.0f KB)", path, path.stat().st_size / 1024)
    return path
