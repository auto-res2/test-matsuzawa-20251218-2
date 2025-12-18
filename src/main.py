
"""
Main orchestrator for SNARL experiment runs.
Receives run_id and mode via Hydra CLI, loads run config, and executes training subprocess.

CLI Usage:
  uv run python -u -m src.main run={run_id} results_dir={path} mode=full
  uv run python -u -m src.main run={run_id} results_dir={path} mode=trial
"""

import sys
import logging
from pathlib import Path
import subprocess
import os

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for experiment orchestration.
    
    Executed from repository root via:
    uv run python -u -m src.main run={run_id} results_dir={path} mode=full
    
    Args:
        cfg: Hydra configuration object
    """
    
    # Validate required parameters
    if cfg.run.run_id is None:
        log.error("ERROR: run.run_id must be specified via CLI: run={run_id}")
        sys.exit(1)
    
    if cfg.mode not in ["full", "trial"]:
        log.error(f"ERROR: mode must be 'full' or 'trial', got {cfg.mode}")
        sys.exit(1)
    
    log.info(f"Starting experiment orchestrator")
    log.info(f"  run_id: {cfg.run.run_id}")
    log.info(f"  mode: {cfg.mode}")
    log.info(f"  results_dir: {cfg.results_dir}")
    
    # Verify run config exists
    run_config_path = Path("config") / "runs" / f"{cfg.run.run_id}.yaml"
    if not run_config_path.exists():
        log.error(f"Run config not found: {run_config_path}")
        log.error(f"Available run configs should be in config/runs/ directory")
        sys.exit(1)
    
    # Create results directory
    results_path = Path(cfg.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Results directory created: {results_path}")
    
    # Configure mode-based settings
    if cfg.mode == "trial":
        log.info("TRIAL MODE: Lightweight validation run")
        log.info("  - epochs=1")
        log.info("  - wandb.mode=disabled")
        log.info("  - optuna.n_trials=0")
        log.info("  - batch limit=2")
    elif cfg.mode == "full":
        log.info("FULL MODE: Complete experiment execution")
        log.info(f"  - wandb.mode=online")
    
    # Save configuration
    config_output_path = results_path / f"{cfg.run.run_id}_config.yaml"
    with open(config_output_path, "w") as f:
        OmegaConf.save(cfg, f)
    log.info(f"Configuration saved: {config_output_path}")
    
    # Execute training subprocess
    try:
        log.info("Launching training subprocess (src.train)...")
        
        # Prepare environment with mode setting
        env = os.environ.copy()
        env['SNARL_MODE'] = cfg.mode
        env['SNARL_BATCH_LIMIT'] = str(2 if cfg.mode == "trial" else 0)
        
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.train",
            f"run={cfg.run.run_id}",
            f"results_dir={cfg.results_dir}",
            f"mode={cfg.mode}",
            f"wandb.mode={'disabled' if cfg.mode == 'trial' else 'online'}",
        ]
        
        log.info(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, env=env)
        
        if result.returncode != 0:
            log.error(f"Training subprocess failed with return code {result.returncode}")
            sys.exit(result.returncode)
        
        log.info(f"✓ Training completed successfully for run_id={cfg.run.run_id}")
    
    except subprocess.CalledProcessError as e:
        log.error(f"Training subprocess failed: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
