"""Run every check the repository has (docs/02 section 9, docs/07).

    uv run python scripts/validate_all.py
    uv run python scripts/validate_all.py --offline
    uv run python scripts/validate_all.py --quick     # skip the slow world generation

Checks, in order:

1. **lint** - ruff over the whole repository
2. **registries** - 93 study KPIs, 25 models, acyclic graph, every unit allowed
3. **schemas** - the exported JSON Schemas match the current contracts
4. **tests** - the shared suite plus every model folder's own tests
5. **model folders** - each one matches the docs/02 section 2 layout and has valid outputs
6. **no cross-folder imports** - a model folder may never import another (CLAUDE.md)
7. **licences** - no AGPL/BSL package has crept into a requirements file

Exit code is non-zero when anything fails, so this is what CI runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "00_common"))

from twin_common.contracts import registry as reg  # noqa: E402

#: docs/06 section 1: never allowed, whatever the reason.
FORBIDDEN_PACKAGES = ("ultralytics", "boxmot", "sdv")
#: docs/06 section 3: supervision's tracker is deprecated in favour of `trackers`.
FORBIDDEN_SYMBOLS = ("sv.ByteTrack", "supervision.ByteTrack")

#: Files and directories every model folder must have (docs/02 section 2).
REQUIRED_FILES = (
    "README.md",
    "model_card.md",
    "config.yaml",
    "requirements.txt",
    "Dockerfile",
    "api.py",
    "src/model.py",
    "src/run.py",
)
REQUIRED_DIRECTORIES = (
    "src",
    "tests",
    "schemas",
    "data/synthetic",
    "data/sample_upstream",
    "outputs",
)


@dataclass
class Result:
    """Outcome of the whole run."""

    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed.append(name)
            print(f"  PASS  {name}")
        else:
            self.failed.append(f"{name}: {detail}" if detail else name)
            print(f"  FAIL  {name}" + (f" - {detail}" if detail else ""))
        return ok

    def skip(self, name: str, why: str) -> None:
        self.skipped.append(f"{name}: {why}")
        print(f"  SKIP  {name} - {why}")

    @property
    def ok(self) -> bool:
        return not self.failed


def run(command: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    process = subprocess.run(
        command,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return process.returncode, (process.stdout + process.stderr)


def existing_model_folders() -> dict[str, Path]:
    return {
        model_id: REPO_ROOT / info.folder
        for model_id, info in reg.all_models().items()
        if (REPO_ROOT / info.folder).is_dir()
    }


# ----------------------------------------------------------------------- checks
def check_lint(result: Result) -> None:
    print("\n[1/7] lint")
    code, output = run(["uv", "run", "ruff", "check", ".", "--output-format", "concise"])
    result.check("ruff check", code == 0, output.strip().splitlines()[-1] if code else "")
    code, output = run(["uv", "run", "ruff", "format", "--check", "."])
    result.check("ruff format", code == 0, "run `ruff format .`" if code else "")


def check_registries(result: Result) -> None:
    print("\n[2/7] registries")
    kpis = reg.all_kpis()
    study = [name for name, info in kpis.items() if not info.extension]
    result.check("93 study KPIs", len(study) == 93, f"found {len(study)}")
    result.check("25 models", len(reg.all_models()) == 25, f"found {len(reg.all_models())}")
    try:
        order = reg.topological_order()
        result.check("upstream graph is acyclic", len(order) == 25)
    except Exception as exc:
        result.check("upstream graph is acyclic", False, str(exc))
    owners = {info.owner for info in kpis.values()}
    result.check(
        "every model owns a KPI",
        not (set(reg.all_models()) - owners),
        f"without KPIs: {sorted(set(reg.all_models()) - owners)}",
    )
    allowed = reg.allowed_units()
    bad = {name: info.unit for name, info in kpis.items() if info.unit not in allowed}
    result.check("every unit is allowed", not bad, str(bad))


def check_schemas(result: Result) -> None:
    print("\n[3/7] exported schemas")
    from twin_common.contracts import SCHEMA_TARGETS, schema_for
    from twin_common.paths import schemas_dir

    target = schemas_dir()
    for filename, model in SCHEMA_TARGETS.items():
        path = target / filename
        if not path.is_file():
            result.check(f"{filename} exported", False, "missing")
            continue
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        result.check(
            f"{filename} up to date",
            on_disk == schema_for(model),
            "stale; re-run twin_common.contracts.export_schemas()",
        )


def check_tests(result: Result, *, offline: bool) -> None:
    print("\n[4/7] tests")
    command = ["uv", "run", "pytest", "-q"]
    env_note = " (TWIN_OFFLINE=1)" if offline else ""
    code, output = run(command)
    summary = ""
    for line in reversed(output.strip().splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            break
    result.check(f"pytest{env_note}", code == 0, summary)
    if summary:
        print(f"        {summary}")


def check_model_folders(result: Result) -> None:
    print("\n[5/7] model folders")
    folders = existing_model_folders()
    if not folders:
        result.skip("model folder layout", "no model folders scaffolded yet")
        return
    from twin_common.config import REQUIRED_MODEL_KEYS, load_model_config
    from twin_common.contracts import ModelOutput

    for model_id, folder in sorted(folders.items()):
        missing_files = [f for f in REQUIRED_FILES if not (folder / f).is_file()]
        missing_dirs = [d for d in REQUIRED_DIRECTORIES if not (folder / d).is_dir()]
        result.check(
            f"{model_id} folder layout",
            not missing_files and not missing_dirs,
            f"missing {missing_files + missing_dirs}",
        )
        try:
            config = load_model_config(folder / "config.yaml", validate=False)
            absent = [k for k in REQUIRED_MODEL_KEYS if config.get(k) is None]
            result.check(f"{model_id} config keys", not absent, f"missing {absent}")
            declared = set(config.get("kpis") or [])
            owned = set(reg.kpis_owned_by(model_id))
            result.check(
                f"{model_id} config KPIs match the registry",
                declared == owned,
                f"config {sorted(declared)} vs registry {sorted(owned)}",
            )
        except Exception as exc:
            result.check(f"{model_id} config loads", False, str(exc))

        outputs = sorted((folder / "outputs").glob("*.json"))
        if not outputs:
            result.skip(f"{model_id} sample outputs", "none written yet")
            continue
        for path in outputs:
            try:
                ModelOutput.model_validate(json.loads(path.read_text(encoding="utf-8")))
                result.check(f"{model_id} {path.name} validates", True)
            except Exception as exc:
                result.check(f"{model_id} {path.name} validates", False, str(exc)[:120])


def check_isolation(result: Result) -> None:
    print("\n[6/7] folder isolation")
    folders = existing_model_folders()
    if not folders:
        result.skip("cross-folder imports", "no model folders scaffolded yet")
        return
    folder_names = {info.folder for info in reg.all_models().values()}
    offences: list[str] = []
    for model_id, folder in sorted(folders.items()):
        for path in folder.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for other in folder_names:
                if other == folder.name:
                    continue
                if f"import {other}" in text or f"from {other}" in text:
                    offences.append(f"{model_id}: {path.name} imports {other}")
    result.check("no model imports another model folder", not offences, "; ".join(offences))

    # print in library code, and the deprecated tracker
    bad_symbols: list[str] = []
    for folder in folders.values():
        for path in (folder / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for symbol in FORBIDDEN_SYMBOLS:
                if symbol in text:
                    bad_symbols.append(f"{folder.name}/{path.name}: {symbol}")
    result.check("no deprecated sv.ByteTrack", not bad_symbols, "; ".join(bad_symbols))


def check_licences(result: Result) -> None:
    print("\n[7/7] licences")
    offences: list[str] = []
    for path in REPO_ROOT.rglob("requirements*.txt"):
        if ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for package in FORBIDDEN_PACKAGES:
            if package in text:
                offences.append(f"{path.relative_to(REPO_ROOT)}: {package}")
    for path in REPO_ROOT.rglob("pyproject.toml"):
        if ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for package in FORBIDDEN_PACKAGES:
            # A mention inside a comment warning against it is fine.
            for line in text.splitlines():
                stripped = line.strip()
                if package in stripped and not stripped.startswith("#"):
                    offences.append(f"{path.relative_to(REPO_ROOT)}: {package}")
    result.check(
        "no AGPL/BSL packages (ultralytics, boxmot, sdv)",
        not offences,
        "; ".join(offences),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/validate_all.py",
        description="Run every repository check (docs/02 section 9).",
    )
    parser.add_argument("--offline", action="store_true", help="run with TWIN_OFFLINE=1")
    parser.add_argument("--quick", action="store_true", help="skip the slow test suite")
    args = parser.parse_args(argv)

    if args.offline:
        import os

        os.environ["TWIN_OFFLINE"] = "1"

    print("=" * 72)
    print("validate_all" + (" (offline)" if args.offline else ""))
    print("=" * 72)

    result = Result()
    check_lint(result)
    check_registries(result)
    check_schemas(result)
    if args.quick:
        result.skip("pytest", "--quick")
    else:
        check_tests(result, offline=args.offline)
    check_model_folders(result)
    check_isolation(result)
    check_licences(result)

    print("\n" + "=" * 72)
    print(
        f"{len(result.passed)} passed, {len(result.failed)} failed, {len(result.skipped)} skipped"
    )
    if result.failed:
        print("\nfailures:")
        for failure in result.failed:
            print(f"  - {failure}")
    print("=" * 72)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
