"""FastAPI entry point for M18.

Kept to one statement on purpose (docs/02 section 2): endpoints, CORS, error mapping and
upstream resolution all live in ``twin_common.api.create_app``. Model logic belongs in
``src/model.py``.

Run it with::

    uv run uvicorn api:app --app-dir M18_waste_forecast --port 8018
"""

from src.model import Model

from twin_common.api import create_app

app = create_app(Model)
