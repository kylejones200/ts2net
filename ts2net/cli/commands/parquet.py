"""Parquet file format conversion commands."""

import os
import json
import click
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Optional, Dict, Any

def register_parquet_commands(cli: click.Group) -> None:
    """Register all Parquet-related commands."""
    # to-parquet command
    @cli.command("to-parquet", help="Convert edge list to Parquet format")
    @click.argument("input_file", type=click.Path(exists=True))
    @click.option(
        "--name", 
        required=True,
        help="Name of the graph"
    )
    @click.option(
        "--output", 
        type=click.Path(),
        default="output",
        help="Output directory (default: 'output')"
    )
    @click.option(
        "--directed/--undirected",
        default=False,
        help="Create a directed graph (default: False)"
    )
    def to_parquet(input_file: str, name: str, output: str, directed: bool):
        """Convert an edge list file to Parquet format."""
        # Create output directory if it doesn't exist
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read edge list
        edges = pd.read_csv(input_file, header=None, names=["source", "target", "weight"], 
                           dtype={"source": str, "target": str, "weight": float})
        
        # Create nodes DataFrame
        nodes = pd.DataFrame({
            "id": pd.concat([edges["source"], edges["target"]]).unique(),
            "label": None  # Add any additional node attributes here
        })
        
        # Save nodes and edges to Parquet
        nodes_path = output_path / "nodes.parquet"
        edges_path = output_path / "edges.parquet"
        meta_path = output_path / "graph_meta.json"
        
        nodes.to_parquet(nodes_path, index=False)
        edges.to_parquet(edges_path, index=False)
        
        # Save metadata
        meta = {
            "name": name,
            "directed": directed,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes_file": "nodes.parquet",
            "edges_file": "edges.parquet"
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        click.echo(f"Graph saved to {output_path}")
    
    # from-parquet command
    @cli.command("from-parquet", help="Convert from Parquet to other formats")
    @click.argument("meta_file", type=click.Path(exists=True))
    @click.option(
        "--graphml",
        type=click.Path(),
        help="Export to GraphML format"
    )
    def from_parquet(meta_file: str, graphml: Optional[str] = None):
        """Convert from Parquet to other graph formats."""
        meta_path = Path(meta_file)
        base_dir = meta_path.parent
        
        # Load metadata
        with open(meta_file) as f:
            meta = json.load(f)
        
        # Load nodes and edges
        nodes_path = base_dir / meta["nodes_file"]
        edges_path = base_dir / meta["edges_file"]
        
        nodes = pd.read_parquet(nodes_path)
        edges = pd.read_parquet(edges_path)
        
        # Create NetworkX graph
        G = nx.DiGraph() if meta.get("directed", False) else nx.Graph()
        
        # Add nodes with attributes
        for _, row in nodes.iterrows():
            attrs = {k: v for k, v in row.items() if k != 'id' and pd.notna(v)}
            G.add_node(row['id'], **attrs)
        
        # Add edges with weights
        for _, row in edges.iterrows():
            G.add_edge(row['source'], row['target'], weight=row.get('weight', 1.0))
        
        # Export to requested formats
        if graphml:
            try:
                nx.write_graphml(G, graphml)
                click.echo(f"Graph saved to {graphml} in GraphML format")
            except Exception as e:
                click.echo(f"Error exporting to GraphML: {e}", err=True)
                return 1
        
        if not any([graphml]):
            click.echo("No output format specified. Use --graphml to specify an output format.")
            return 1
        
        return 0
