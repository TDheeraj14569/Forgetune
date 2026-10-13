"""
Manifest and cache metadata management for ForgeTune artifacts.
Provides schema validation and atomic file writes.
"""

import json
import os
import tempfile
from typing import Any, Dict

MANIFEST_SCHEMA_VERSION = "1.0"

def write_atomic_json(filepath: str, data: Dict[str, Any]) -> None:
    """
    Atomically writes a dictionary to a JSON file using a temporary file
    and atomic replacement, preventing partially written or corrupted files.
    """
    target_dir = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=target_dir, delete=False, suffix=".tmp", encoding="utf-8") as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name

    # Atomic replace
    os.replace(temp_name, filepath)

def read_manifest(filepath: str) -> Dict[str, Any]:
    """
    Reads and validates a manifest file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Manifest not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema version: {data.get('schema_version')}, expected {MANIFEST_SCHEMA_VERSION}"
        )
    return data

def create_manifest(
    model_name: str,
    input_shapes: list,
    output_shape: list,
    kernel_config: dict,
    compiler_info: dict,
    artifacts: dict,
    optimization_summary: dict = None
) -> dict:
    """
    Constructs a standard ForgeTune manifest dictionary.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_name": model_name,
        "input_shapes": input_shapes,
        "output_shape": output_shape,
        "kernel_config": kernel_config,
        "compiler_info": compiler_info,
        "artifacts": artifacts,
        "optimization_summary": optimization_summary or {},
    }
