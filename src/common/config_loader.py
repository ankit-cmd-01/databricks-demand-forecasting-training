"""Loads config.yml and builds fully qualified table names.

Only the entrypoint files import this. The class files never see the config,
they only receive plain values (table names, paths, parameters).
"""
from pathlib import Path

import yaml

# src/common/config_loader.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"


def load_config(config_path=None):
    """Read config.yml and return it as a dictionary."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def table_name(config, schema_key, table_key):
    """Build catalog.schema.table from the config keys."""
    return (
        f"{config['catalog_name']}."
        f"{config['schemas'][schema_key]}."
        f"{config['tables'][table_key]}"
    )


def model_name(config):
    """Unity Catalog model name: catalog.ml_schema.model."""
    return f"{config['catalog_name']}.{config['schemas']['ml']}.{config['model']['name']}"


def resolve_project_path(relative_path):
    """Turn a path written relative to the project root into an absolute path."""
    return str(PROJECT_ROOT / relative_path)