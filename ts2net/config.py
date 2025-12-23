"""
Configuration schemas for ts2net YAML-based pipeline.

Provides type-safe, validated configuration classes using dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    path: str
    id_col: Optional[str] = None
    time_col: str = "timestamp"
    value_col: str = "value"
    start: Optional[str] = None
    end: Optional[str] = None
    tz: Optional[str] = None
    
    def __post_init__(self):
        """Validate dataset configuration."""
        if not self.name:
            raise ValueError("Dataset name is required")
        if not self.path:
            raise ValueError("Dataset path is required")


@dataclass
class SamplingConfig:
    """Sampling/resampling configuration."""
    frequency: Optional[str] = None
    agg: str = "mean"
    resample: bool = False
    
    def __post_init__(self):
        """Validate sampling configuration."""
        valid_agg = {"mean", "sum", "median", "min", "max"}
        if self.agg not in valid_agg:
            raise ValueError(f"agg must be one of {valid_agg}, got {self.agg}")


@dataclass
class HVGConfig:
    """Horizontal Visibility Graph configuration."""
    enabled: bool = False
    output: str = "stats"
    weighted: bool = False
    limit: Optional[int] = None
    directed: bool = False
    
    def __post_init__(self):
        """Validate HVG configuration."""
        valid_output = {"edges", "degrees", "stats"}
        if self.output not in valid_output:
            raise ValueError(f"output must be one of {valid_output}, got {self.output}")


@dataclass
class NVGConfig:
    """Natural Visibility Graph configuration."""
    enabled: bool = False
    output: str = "stats"
    weighted: bool = False
    limit: Optional[int] = None
    max_edges: Optional[int] = None
    max_edges_per_node: Optional[int] = None
    max_memory_mb: Optional[int] = None
    
    def __post_init__(self):
        """Validate NVG configuration."""
        valid_output = {"edges", "degrees", "stats"}
        if self.output not in valid_output:
            raise ValueError(f"output must be one of {valid_output}, got {self.output}")


@dataclass
class RecurrenceConfig:
    """Recurrence Network configuration."""
    enabled: bool = False
    output: str = "stats"
    rule: str = "knn"
    k: int = 10
    m: Optional[int] = None
    tau: int = 1
    epsilon: float = 0.1
    metric: str = "euclidean"
    
    def __post_init__(self):
        """Validate recurrence configuration."""
        valid_output = {"edges", "degrees", "stats"}
        if self.output not in valid_output:
            raise ValueError(f"output must be one of {valid_output}, got {self.output}")
        
        valid_rule = {"knn", "epsilon", "radius"}
        if self.rule not in valid_rule:
            raise ValueError(f"rule must be one of {valid_rule}, got {self.rule}")
        
        valid_metric = {"euclidean", "manhattan", "chebyshev", "minkowski"}
        if self.metric not in valid_metric:
            raise ValueError(f"metric must be one of {valid_metric}, got {self.metric}")
        
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.tau < 1:
            raise ValueError(f"tau must be >= 1, got {self.tau}")


@dataclass
class TransitionConfig:
    """Transition Network configuration."""
    enabled: bool = False
    output: str = "stats"
    symbolizer: str = "ordinal"
    order: int = 3
    n_states: Optional[int] = None
    
    def __post_init__(self):
        """Validate transition configuration."""
        valid_output = {"edges", "degrees", "stats"}
        if self.output not in valid_output:
            raise ValueError(f"output must be one of {valid_output}, got {self.output}")
        
        valid_symbolizer = {"ordinal", "equal_width", "equal_freq", "kmeans"}
        if self.symbolizer not in valid_symbolizer:
            raise ValueError(f"symbolizer must be one of {valid_symbolizer}, got {self.symbolizer}")
        
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")


@dataclass
class GraphsConfig:
    """Graph methods configuration."""
    hvg: HVGConfig = field(default_factory=HVGConfig)
    nvg: NVGConfig = field(default_factory=NVGConfig)
    recurrence: RecurrenceConfig = field(default_factory=RecurrenceConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphsConfig:
        """Create GraphsConfig from dictionary."""
        return cls(
            hvg=HVGConfig(**data.get('hvg', {})),
            nvg=NVGConfig(**data.get('nvg', {})),
            recurrence=RecurrenceConfig(**data.get('recurrence', {})),
            transition=TransitionConfig(**data.get('transition', {}))
        )


@dataclass
class WindowsConfig:
    """Windowing configuration."""
    enabled: bool = False
    size: Optional[int] = None
    step: Optional[int] = None
    
    def __post_init__(self):
        """Validate windows configuration."""
        if self.enabled and self.size is None:
            raise ValueError("window size is required when windows are enabled")
        if self.step is not None and self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")
        if self.size is not None and self.size < 1:
            raise ValueError(f"size must be >= 1, got {self.size}")


@dataclass
class BSTSConfig:
    """BSTS decomposition configuration."""
    enabled: bool = False
    level: bool = True
    trend: bool = False
    seasonal_periods: Optional[List[int]] = None
    robust: bool = False
    standardize_residual: bool = True
    max_points: int = 10000
    window: Optional[int] = None


@dataclass
class OutputConfig:
    """Output configuration."""
    format: str = "parquet"
    path: str = "results/output.parquet"
    overwrite: bool = True
    mode: Optional[str] = None  # Deprecated, use graph-specific output
    
    def __post_init__(self):
        """Validate output configuration."""
        valid_format = {"parquet", "csv", "json"}
        if self.format not in valid_format:
            raise ValueError(f"format must be one of {valid_format}, got {self.format}")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_errors: bool = True
    error_path: Optional[str] = None


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    dataset: DatasetConfig
    graphs: GraphsConfig
    output: OutputConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    bsts: BSTSConfig = field(default_factory=BSTSConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        """Create PipelineConfig from dictionary (e.g., from YAML)."""
        return cls(
            dataset=DatasetConfig(**data['dataset']),
            graphs=GraphsConfig.from_dict(data.get('graphs', {})),
            output=OutputConfig(**data.get('output', {})),
            sampling=SamplingConfig(**data.get('sampling', {})),
            windows=WindowsConfig(**data.get('windows', {})),
            bsts=BSTSConfig(**data.get('bsts', {})),
            logging=LoggingConfig(**data.get('logging', {}))
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> PipelineConfig:
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate required sections
        required = ['dataset', 'graphs', 'output']
        for section in required:
            if section not in data:
                raise ValueError(f"Missing required section: {section}")
        
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)

