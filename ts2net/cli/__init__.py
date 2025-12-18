"""
Command-line interface for ts2net.

This module provides the main entry point for the ts2net CLI.
"""

import click
import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Default values
DEFAULT_SEED = 3363

# Create the main Click group
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="ts2net")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Time series to network conversion and analysis."""
    ctx.ensure_object(dict)
    ctx.obj["seed"] = DEFAULT_SEED

# Dynamically load all command modules
def register_commands() -> None:
    """Dynamically discover and register all command modules."""
    commands_pkg = Path(__file__).parent / "commands"
    
    # Import all modules in the commands package
    for _, module_name, _ in pkgutil.iter_modules([str(commands_pkg)]):
        module = importlib.import_module(f"ts2net.cli.commands.{module_name}")
        
        # Look for register_*_commands functions and call them
        for name, func in module.__dict__.items():
            if name.startswith("register_") and name.endswith("_commands"):
                func(cli)

# Register all commands
register_commands()

# Main entry point
def main() -> None:
    """Main entry point for the CLI."""
    cli(obj={})

# For backward compatibility
class CLI:
    """Legacy CLI class for backward compatibility."""
    
    def __init__(self):
        self.name = "ts2net"  # Add name attribute for tests
    
    def __call__(self, *args, **kwargs):
        return main()
