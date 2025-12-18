
"""
Independent evaluation and visualization script.
Retrieves experimental results from WandB and generates comprehensive analysis.

Executed separately via:
uv run python -m src.evaluate results_dir={path} run_ids='["run-1", "run-2"]'
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    log.warning("WandB not available")


# ============================================================================
# WandB Data Retrieval
# ============================================================================

def load_wandb_config(config_path: Path) -> Dict:
    """Load WandB configuration."""
    if not HAS_WANDB:
        return {'entity': 'gengaru617-personal', 'project': '2025-11-19'}
    
    if not config_path.exists():
        log.warning(f"Config not found: {config_path}, using defaults")
        return {'entity': 'gengaru617-personal', 'project': '2025-11-19'}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        wandb_cfg = config.get('wandb', {})
        return {
            'entity': wandb_cfg.get('entity', 'gengaru617-personal'),
            'project': wandb_cfg.get('project', '2025-11-19')
        }
    except Exception as e:
        log.warning(f"Error loading config: {e}, using defaults")
        return {'entity': 'gengaru617-personal', 'project': '2025-11-19'}


def retrieve_run_data_from_wandb(run_id: str, entity: str, project: str) -> Optional[Dict]:
    """Retrieve data from WandB API."""
    
    if not HAS_WANDB:
        return None
    
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        history = run.history()
        summary = run.summary._json_dict if hasattr(run.summary, '_json_dict') else dict(run.summary)
        config = dict(run.config)
        
        log.info(f"Retrieved WandB data for {run_id}")
        
        return {
            'run_id': run_id,
            'history': history,
            'summary': summary,
            'config': config,
            'state': run.state
        }
    
    except Exception as e:
        log.warning(f"Error retrieving WandB data: {e}")
        return None


def retrieve_run_data_from_local(results_dir: Path, run_id: str) -> Optional[Dict]:
    """Retrieve metrics from local files."""
    
    metrics_file = results_dir / run_id / "metrics.json"
    
    if not metrics_file.exists():
        log.warning(f"Local metrics file not found: {metrics_file}")
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return {
            'run_id': run_id,
            'history': {
                'train_loss': metrics.get('train_losses', []),
                'val_accuracy': metrics.get('val_accuracies', [])
            },
            'summary': {
                'final_test_accuracy': metrics.get('test_accuracy'),
                'convergence_speed': metrics.get('convergence_speed'),
                'test_loss': metrics.get('test_loss')
            },
            'config': {},
            'state': 'completed'
        }
    
    except Exception as e:
        log.warning(f"Error reading local metrics: {e}")
        return None


def retrieve_run_data(run_id: str, results_dir: Path, entity: str, project: str) -> Optional[Dict]:
    """Retrieve run data from WandB or local fallback."""
    
    if HAS_WANDB:
        data = retrieve_run_data_from_wandb(run_id, entity, project)
        if data is not None:
            return data
    
    return retrieve_run_data_from_local(results_dir, run_id)


# ============================================================================
# Per-Run Processing
# ============================================================================

def process_single_run(run_data: Dict, output_dir: Path) -> Optional[Dict]:
    """Process single run and export metrics."""
    
    if run_data is None:
        return None
    
    run_id = run_data['run_id']
    history = run_data['history']
    summary = run_data['summary']
    
    run_output_dir = output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert history to dict
    metrics_dict = {}
    
    if isinstance(history, pd.DataFrame):
        for col in history.columns:
            if col not in ['_step', '_runtime', '_timestamp']:
                values = history[col].dropna().tolist()
                metrics_dict[col] = values
    else:
        metrics_dict = dict(history)
    
    metrics_dict['summary'] = summary
    
    # Export to JSON
    metrics_file = run_output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    
    log.info(f"Exported metrics for {run_id}: {metrics_file}")
    print(f"Exported: {metrics_file}")
    
    # Generate figures
    generate_run_figures(run_id, metrics_dict, run_output_dir)
    
    return metrics_dict


def generate_run_figures(run_id: str, metrics_dict: Dict, output_dir: Path) -> None:
    """Generate per-run figures (learning curve, etc.)."""
    
    val_acc = metrics_dict.get('val_accuracy', [])
    
    if not val_acc or len(val_acc) == 0:
        log.warning(f"No validation accuracy found for {run_id}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = list(range(len(val_acc)))
    ax.plot(epochs, val_acc, marker='o', markersize=4, label='Validation Accuracy')
    
    # Mark convergence (95% of final)
    if len(val_acc) > 0 and val_acc[-1] > 0:
        target = 0.95 * val_acc[-1]
        for epoch, acc in enumerate(val_acc):
            if acc >= target:
                ax.axvline(x=epoch, color='red', linestyle='--', alpha=0.5,
                          label=f'95% final at epoch {epoch}')
                break
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax.set_title(f'{run_id} - Learning Curve', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Generated: {fig_path}")
    print(f"Generated figure: {fig_path}")


# ============================================================================
# Aggregated Analysis
# ============================================================================

def compute_convergence_speed(metrics_history: List[float], target_percentage: float = 0.95) -> int:
    """Compute convergence speed: epoch where metric reaches 95% of final."""
    if not metrics_history or len(metrics_history) == 0:
        return -1
    
    final_value = metrics_history[-1]
    if final_value <= 0:
        return -1
    
    target_threshold = target_percentage * final_value
    
    for epoch, metric in enumerate(metrics_history):
        if metric >= target_threshold:
            return epoch
    
    return len(metrics_history)


def aggregate_metrics(all_runs_data: List[Dict]) -> Dict:
    """Aggregate metrics across all runs."""
    
    if not all_runs_data:
        return {}
    
    aggregated = {
        'primary_metric': 'early_convergence_speed',
        'metrics': {},
        'best_proposed': None,
        'best_baseline': None,
        'gap': None
    }
    
    for run_data in all_runs_data:
        if run_data is None:
            continue
        
        run_id = run_data['run_id']
        summary = run_data.get('summary', {})
        history = run_data.get('history', {})
        
        # Try to compute convergence speed from history if not in summary
        conv_speed = summary.get('convergence_speed')
        if conv_speed is None and 'val_accuracy' in history:
            val_acc = history.get('val_accuracy', [])
            if isinstance(val_acc, list) and len(val_acc) > 0:
                conv_speed = compute_convergence_speed(val_acc)
        
        if conv_speed is not None:
            aggregated['metrics'].setdefault('convergence_speed', {})[run_id] = conv_speed
        
        final_acc = summary.get('final_test_accuracy')
        if final_acc is not None:
            aggregated['metrics'].setdefault('final_test_accuracy', {})[run_id] = final_acc
    
    # Compute best runs
    if 'convergence_speed' in aggregated['metrics']:
        conv_speeds = aggregated['metrics']['convergence_speed']
        
        proposed = {k: v for k, v in conv_speeds.items() if 'proposed' in k.lower()}
        baseline = {k: v for k, v in conv_speeds.items() if 'baseline' in k.lower() or 'comparative' in k.lower()}
        
        if proposed:
            best_proposed_id = min(proposed.keys(), key=lambda k: proposed[k])
            aggregated['best_proposed'] = {
                'run_id': best_proposed_id,
                'convergence_speed': proposed[best_proposed_id]
            }
        
        if baseline:
            best_baseline_id = min(baseline.keys(), key=lambda k: baseline[k])
            aggregated['best_baseline'] = {
                'run_id': best_baseline_id,
                'convergence_speed': baseline[best_baseline_id]
            }
        
        # Compute gap (speedup percentage)
        if aggregated['best_proposed'] and aggregated['best_baseline']:
            prop_speed = aggregated['best_proposed']['convergence_speed']
            base_speed = aggregated['best_baseline']['convergence_speed']
            if base_speed > 0:
                speedup = (base_speed - prop_speed) / base_speed * 100
                aggregated['gap'] = speedup
                log.info(f"Speedup: {speedup:.1f}%")
    
    return aggregated


def export_aggregated_metrics(aggregated: Dict, output_dir: Path) -> None:
    """Export aggregated metrics to JSON."""
    
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = comparison_dir / "aggregated_metrics.json"
    
    with open(metrics_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    log.info(f"Exported aggregated metrics: {metrics_file}")
    print(f"Exported: {metrics_file}")


def generate_comparison_figures(aggregated: Dict, output_dir: Path) -> None:
    """Generate comparison figures across runs."""
    
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    if 'convergence_speed' not in aggregated.get('metrics', {}):
        log.warning("No convergence speed data to visualize")
        return
    
    conv_speeds = aggregated['metrics']['convergence_speed']
    
    if len(conv_speeds) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    run_ids = list(conv_speeds.keys())
    speeds = list(conv_speeds.values())
    colors = ['green' if 'proposed' in rid.lower() else 'blue' for rid in run_ids]
    
    bars = ax.bar(range(len(run_ids)), speeds, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, speed in zip(bars, speeds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(speed)}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.set_ylabel('Convergence Speed (epochs)', fontsize=11)
    ax.set_title('Convergence Speed Comparison (Lower is Better)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='SNARL (Proposed)'),
        Patch(facecolor='blue', alpha=0.7, label='Baseline')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    fig_path = comparison_dir / "comparison_convergence_speed_bar.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Generated: {fig_path}")
    print(f"Generated figure: {fig_path}")


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    """Main evaluation entry point."""
    
    parser = argparse.ArgumentParser(description="Evaluate SNARL experiment runs")
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    parser.add_argument('--run_ids', type=str, required=True, help='JSON list of run IDs')
    
    args = parser.parse_args()
    
    # Parse run IDs
    try:
        run_ids = json.loads(args.run_ids)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse run_ids: {e}")
        sys.exit(1)
    
    log.info(f"Evaluating {len(run_ids)} runs")
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load WandB config
    config_path = Path("config") / "config.yaml"
    wandb_config = load_wandb_config(config_path)
    entity = wandb_config['entity']
    project = wandb_config['project']
    
    # Retrieve run data
    log.info("Retrieving run data...")
    all_runs_data = []
    for run_id in run_ids:
        run_data = retrieve_run_data(run_id, results_dir, entity, project)
        all_runs_data.append(run_data)
    
    # Process runs
    log.info("Processing runs...")
    for run_data in all_runs_data:
        if run_data:
            process_single_run(run_data, results_dir)
    
    # Aggregate metrics
    log.info("Aggregating metrics...")
    aggregated = aggregate_metrics(all_runs_data)
    
    # Export and visualize
    export_aggregated_metrics(aggregated, results_dir)
    generate_comparison_figures(aggregated, results_dir)
    
    log.info("✓ Evaluation complete")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    if aggregated.get('best_proposed'):
        print(f"Best Proposed (SNARL): {aggregated['best_proposed']['run_id']}")
        print(f"  Convergence Speed: {aggregated['best_proposed']['convergence_speed']} epochs")
    
    if aggregated.get('best_baseline'):
        print(f"Best Baseline: {aggregated['best_baseline']['run_id']}")
        print(f"  Convergence Speed: {aggregated['best_baseline']['convergence_speed']} epochs")
    
    if aggregated.get('gap') is not None:
        print(f"\nSpeedup: {aggregated['gap']:.1f}%")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
