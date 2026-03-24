"""WSGI entrypoint for Azure App Service / Gunicorn.

Gunicorn startup command should reference this as: `wsgi:app`.
"""

# Root deployment: app.py at repo root
try:
    from app import app  # type: ignore
except ModuleNotFoundError:
    # Fallback (only if someone deployed just the frontend folder)
    from frontend.app import app  # type: ignore
