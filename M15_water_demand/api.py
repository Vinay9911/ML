"""FastAPI entry point for M15.

Kept to one statement on purpose (docs/02 section 2): endpoints, CORS, error mapping and
upstream resolution all live in ``twin_common.api.create_app``. Model logic belongs in
``src/model.py``.

Run it with::

    uv run uvicorn api:app --app-dir M15_water_demand --port 8015
"""

from src.model import Model

from twin_common.api import create_app

app = create_app(Model)
