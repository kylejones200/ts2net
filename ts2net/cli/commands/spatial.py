"""Spatial analysis CLI commands."""

import click
import numpy as np
from pathlib import Path
from typing import Optional

# Import spatial functions (these will need to be implemented in the core module)
from ts2net.core.spatial import radius_weights, knn_weights

def register_spatial_commands(cli: click.Group) -> None:
    """Register all spatial analysis commands."""
    @cli.group("spatial-weights", help="Spatial weights matrix generation")
    def spatial_weights():
        pass
    
    # Register spatial weights subcommands
    register_radius_command(spatial_weights)
    register_knn_command(spatial_weights)

def register_radius_command(spatial_group: click.Group) -> None:
    """Register the radius-based spatial weights command."""
    @spatial_group.command("radius", help="Generate spatial weights using radius-based neighbors")
    @click.argument("coords_file", type=click.Path(exists=True))
    @click.option(
        "--radius", 
        type=float, 
        required=True,
        help="Radius for neighbor inclusion"
    )
    @click.option(
        "--output", 
        type=click.Path(),
        default="weights.txt",
        help="Output file path (default: weights.txt)"
    )
    @click.option(
        "--binary/--weighted",
        default=True,
        help="Create binary weights (True) or distance-weighted (False)"
    )
    def radius(coords_file: str, radius: float, output: str, binary: bool):
        """Generate spatial weights using radius-based neighbors."""
        # Load coordinates
        coords = np.loadtxt(coords_file)
        
        # Generate weights
        weights = radius_weights(coords, radius=radius, binary=binary)
        
        # Save results
        np.savetxt(output, weights, fmt='%g')
        click.echo(f"Spatial weights saved to {output}")

def register_knn_command(spatial_group: click.Group) -> None:
    """Register the KNN-based spatial weights command."""
    @spatial_group.command("knn", help="Generate spatial weights using K-nearest neighbors")
    @click.argument("coords_file", type=click.Path(exists=True))
    @click.option(
        "--k", 
        type=int, 
        required=True,
        help="Number of nearest neighbors"
    )
    @click.option(
        "--output", 
        type=click.Path(),
        default="weights_knn.txt",
        help="Output file path (default: weights_knn.txt)"
    )
    @click.option(
        "--binary/--weighted",
        default=True,
        help="Create binary weights (True) or distance-weighted (False)"
    )
    def knn(coords_file: str, k: int, output: str, binary: bool):
        """Generate spatial weights using K-nearest neighbors."""
        # Load coordinates
        coords = np.loadtxt(coords_file)
        
        # Generate weights
        weights = knn_weights(coords, k=k, binary=binary)
        
        # Save results
        np.savetxt(output, weights, fmt='%g')
        click.echo(f"Spatial weights saved to {output}")
