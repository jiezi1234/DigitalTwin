"""Shared test isolation settings."""

import os


# Unit tests must not start real exporters from a developer's local .env file.
os.environ["OTEL_ENABLED"] = "false"
