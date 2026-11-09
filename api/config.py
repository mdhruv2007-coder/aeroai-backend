"""
AERO.AI MVP1 — config.py
Loads all settings from .env via pydantic-settings.
Never hardcode paths or keys outside this file.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # --- Model paths (relative to project root, overridable via .env) ---
    model_dir: Path = Path("/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1/models_release")

    onnx_model:               str = "aero_ai_v2.2_multifidelity.onnx"

    # Scalers — x/y for each output + adapter extras
    cl_x_scaler:              str = "cl_model_x_scaler.pkl"
    cl_y_scaler:              str = "cl_model_y_scaler.pkl"
    cm_y_scaler:              str = "cm_model_y_scaler.pkl"
    cd_y_scaler:              str = "rans_adapter_cd_y_scaler.pkl"
    stall_y_scaler:           str = "stall_margin_model_y_scaler.pkl"
    extra_feats_scaler:       str = "rans_adapter_extra_feats_scaler.pkl"

    # --- API settings ---
    api_version:              str = "v1"
    model_version:            str = "aero_ai_v2.2_multifidelity"
    host:                     str = "0.0.0.0"
    port:                     int = 8000

    # --- Deprecated: kept only so the N_MC_PASSES line in .env does not fail validation ---
    n_mc_passes:              int = 30

    # --- Reynolds limits ---
    re_hard_min:              float = 500_000
    re_hard_max:              float = 2_000_000
    re_soft_min:              float = 800_000   # warn below this

    # --- API Keys (loaded from .env — NEVER hardcode real values here) ---
    api_key_admin:            str = "changeme-admin"
    api_key_friend1:          str = "changeme-friend1"
    api_key_friend2:          str = "changeme-friend2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single shared instance — import this everywhere
settings = Settings()
