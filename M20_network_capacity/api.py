"""FastAPI entry point for M20.

Kept to one statement on purpose (docs/02 section 2): endpoints, CORS, error mapping and
upstream resolution all live in ``twin_common.api.create_app``. Model logic belongs in
``src/model.py``.

Run it with::

    uv run uvicorn api:app --app-dir M20_network_capacity --port 8020
"""

from src.model import Model

from twin_common.api import create_app

app = create_app(Model)
