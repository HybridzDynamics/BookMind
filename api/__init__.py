"""
BookMind API Package.

This package exposes the FastAPI application entrypoint and route handlers
for the BookMind book recommendation engine.
"""

from api.main import app

__all__ = ["app"]
