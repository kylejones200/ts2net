"""YAML-based pipeline CLI commands."""

import click
from pathlib import Path
from typing import Optional

def register_pipeline_commands(cli: click.Group) -> None:
    """Register pipeline-related commands."""
    @cli.command("run", help="Run analysis pipeline from YAML configuration")
    @click.argument("config", type=click.Path(exists=True, path_type=Path))
    @click.option(
        "--validate-only",
        is_flag=True,
        help="Only validate configuration, don't run pipeline"
    )
    def run(config: Path, validate_only: bool):
        """Run analysis pipeline from YAML configuration file.
        
        Examples:
        
        \b
            ts2net run configs/spain_smart_meters.yaml
            ts2net run configs/morocco_zones.yaml
            ts2net run configs/north_dakota_wells.yaml --validate-only
        """
        import sys
        import os
        
        # Add scripts directory to path to import run_from_config
        scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        
        try:
            from run_from_config import load_config, main
            
            if validate_only:
                # Just validate the config
                try:
                    cfg = load_config(str(config))
                    click.echo(f"Configuration valid: {config}")
                    click.echo(f"  Dataset: {cfg['dataset'].get('name', 'unknown')}")
                    enabled_graphs = [k for k, v in cfg['graphs'].items() if v.get('enabled', False)]
                    click.echo(f"  Graphs enabled: {', '.join(enabled_graphs) if enabled_graphs else 'none'}")
                    if cfg.get('windows', {}).get('enabled', False):
                        click.echo(f"  Windows: size={cfg['windows'].get('size')}, step={cfg['windows'].get('step')}")
                    click.echo(f"  Output: {cfg['output'].get('path')}")
                except Exception as e:
                    click.echo(f"Configuration error: {e}", err=True)
                    sys.exit(1)
            else:
                # Run the pipeline
                main(str(config))
                
        except ImportError as e:
            click.echo(f"Error importing pipeline runner: {e}", err=True)
            click.echo("Make sure you're running from the ts2net repository root.", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Pipeline failed: {e}", err=True)
            sys.exit(1)
