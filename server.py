from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from rag_runes.api.app import create_app
from rag_runes.api.config import AppConfig, load_dotenv, load_settings


def load_server_runtime() -> tuple[str, int]:
    env_file = Path(os.environ.get("RAG_ENV_FILE", ".env"))
    load_dotenv(env_file)
    settings_path = Path(os.environ.get("RAG_SETTINGS_FILE", "config/settings.toml"))
    settings = load_settings(settings_path)

    server_section = settings.get("server", {})
    if not isinstance(server_section, dict):
        server_section = {}

    default_host = str(server_section.get("host", "0.0.0.0"))
    default_port = str(server_section.get("port", 8000))
    host_env = os.environ.get("RAG_HOST")
    port_env = os.environ.get("RAG_PORT")
    host = host_env.strip() if host_env and host_env.strip() else default_host
    port_raw = port_env.strip() if port_env and port_env.strip() else default_port
    try:
        port = int(port_raw)
    except (TypeError, ValueError):
        port = 8000
    return host, port


CONFIG = AppConfig.from_env()
app = create_app(CONFIG)

if __name__ == "__main__":
    host, port = load_server_runtime()
    uvicorn.run(app, host=host, port=port)
