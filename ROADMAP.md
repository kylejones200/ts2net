# ts2net Feature Roadmap

## Current State Assessment

### Already Implemented ✅
- **Bounded NVG**: First-class method with `limit`, `max_edges`, `max_edges_per_node`, `max_memory_mb`
- **Weighted Visibility Graphs**: Basic support with `weighted=True` (uses absolute difference)
- **Windowed APIs**: `api_windows.py` and `multivariate/windows.py` for rolling features
- **Ordinal Transition Networks**: `TransitionNetwork` with ordinal patterns
- **Scalable Recurrence**: k-NN recurrence already available
- **YAML Config Pipeline**: Full pipeline with config validation

### Needs Enhancement 🔧
- **Weighted Visibility**: Only supports `absdiff` currently. `_vis_weights()` function exists but needs integration with more modes (minimum clearance, time gap, slope)
- **Directed Visibility Graphs**: ✅ **COMPLETED** - Implemented with `directed` parameter, in/out degree sequences, and irreversibility_score
- **Null Models & Significance**: No surrogate testing or z-score reporting

## Recommended Priority: Smart Meters

**Rationale:**
- Config exists: `spain_smart_meters.yaml`
- Benchmarks successful: 50 meters analyzed, 2.3M points
- Clear operational value: Fault detection, consumption pattern analysis, anomaly detection
- Strong use cases for directed and weighted graphs

## Top 3 Immediate Additions

### 1. Directed HVG/NVG (Priority: High) ✅ **COMPLETED**

**Why:** Asymmetry in consumption patterns (ramp up vs ramp down) provides irreversibility metrics useful for fault onset detection.

**Status:** ✅ Fully implemented
- `directed: bool` parameter added to HVG/NVG
- Forward edges: `i → j` where `i < j` (time-forward only)
- In-degree and out-degree computation implemented
- `irreversibility_score` = `abs(in_degree - out_degree) / total_degree` added to stats
- Config schemas updated: `HVGConfig.directed: bool = False`
- Factory updated to handle directed mode
- `in_degree_sequence()` and `out_degree_sequence()` methods available
- Comprehensive tests added (15 tests in `test_directed_hvg.py`)

---

### 2. Enhanced Weighted Visibility Graphs (Priority: High)

**Why:** Current implementation only supports `absdiff`. Adding minimum clearance and time gap weights enables stronger anomaly detection.

**Implementation:**
- Expand `weighted` parameter: `weighted: bool | str = False`
- String modes: `"absdiff"`, `"time_gap"`, `"min_clearance"`, `"slope"`
- `"absdiff"`: Current behavior (absolute height difference)
- `"time_gap"`: Weight by temporal distance `|j - i|`
- `"min_clearance"`: Weight by minimum clearance above intermediate points
- `"slope"`: Weight by slope `(x[j] - x[i]) / (j - i)`
- Update both HVG and NVG to support all modes
- Integrate existing `_vis_weights()` function properly

**Outputs:**
- Weight statistics in `stats()`: `min_weight`, `max_weight`, `mean_weight`, `std_weight`
- Weight distributions useful for anomaly detection

**Tests:**
- Different weight modes produce expected weight ranges
- Anomalies show weight distribution shifts
- Memory footprint same as unweighted (only edge storage increases)

---

### 3. Ordinal Partition Networks (Priority: Medium-High)

**Why:** TransitionNetwork already has ordinal patterns, but can be enhanced for partition-based analysis with entropy and motif counts.

**Implementation:**
- Enhance `TransitionNetwork` with partition mode
- Add `partition_mode: bool = False` parameter
- When enabled: Build Markov graph over ordinal patterns, compute entropy rate
- Add `entropy_rate()` method
- Add `pattern_motifs()` for motif counting
- Tiny graphs, stable memory (already O(n) complexity)

**Outputs:**
- `entropy_rate`: Measure of pattern complexity
- `pattern_distribution`: Frequency of each ordinal pattern
- `motif_counts`: Counts of 3-node and 4-node motifs

**Tests:**
- White noise → high entropy rate
- Periodic signal → low entropy rate
- Chaos → medium-high entropy rate

---

## Use Case Prioritization

### Smart Meters (Recommended Starting Point)
**Best additions:** Directed HVG + Enhanced Weighted HVG
- **Directed HVG**: Detects consumption asymmetry (faults, unusual patterns)
- **Weighted HVG**: Anomaly detection via weight distribution shifts
- **Why meters first**: Operational value clear, benchmarks exist, config ready

### Wells (Next Priority)
**Best additions:** Ordinal Partition Networks + Multiscale Graphs
- **Ordinal Partition**: Captures nonlinear decline patterns
- **Multiscale**: Different time scales reveal different production phases
- **Why wells second**: Windowed analysis already configured, production monitoring value

### Turbines (Future)
**Best additions:** Joint Recurrence + Cross Visibility
- **Joint Recurrence**: Coupling between multiple sensor streams
- **Cross Visibility**: Relationships between wind speed and power output
- **Why turbines later**: More complex multivariate needs, fewer existing configs

---

## Implementation Plan

### Phase 1: Directed Graphs (Week 1)
1. Update `ts2net/core/visibility/hvg.py` with directed mode
2. Update `ts2net/core/visibility/nvg.py` with directed mode  
3. Add in-degree/out-degree computation
4. Add irreversibility_score to stats
5. Update config schemas and factory
6. Tests and benchmarks

### Phase 2: Enhanced Weights (Week 1-2)
1. Refactor weighted parameter to accept string modes
2. Implement minimum clearance calculation (O(n) for HVG)
3. Implement time gap weighting
4. Integrate `_vis_weights()` properly
5. Add weight statistics to stats()
6. Update config schemas
7. Tests for all weight modes

### Phase 3: Ordinal Partition Enhancement (Week 2)
1. Add partition_mode to TransitionNetwork
2. Implement entropy rate calculation
3. Implement pattern motif counting
4. Update stats() with new metrics
5. Tests and validation

---

## Future Enhancements (Lower Priority)

### Scalable Recurrence Networks
- Add spatial indexing (KD-tree, ball tree) for k-NN
- Approximate neighbor search for epsilon recurrence
- Already have k-NN, but can optimize further

### Multiscale Graphs
- Coarse-grain at multiple scales
- Compute same features at each scale
- Scale signature for detection stability

### Joint/Cross Methods
- Joint recurrence for two series
- Cross visibility graphs
- Coupling measures without full embedding

### Null Models & Significance
- Surrogate data generation (shuffle, phase randomization, block bootstrap)
- Z-score reporting for network metrics
- Turns plots into statistical evidence

---

## Success Metrics

For each addition:
- ✅ Maintains O(n) or O(n×limit) complexity (fast)
- ✅ Memory footprint similar to current methods (safe)
- ✅ Handles 100k+ point series (fleet scale)
- ✅ Clear operational value demonstrated
- ✅ Config-driven, YAML-compatible
- ✅ Well-tested with edge cases

---

## Notes

- All additions follow existing patterns: dataclass configs, factory dispatch, type safety
- Backward compatible: New parameters have sensible defaults
- Documented: Each feature gets docstrings, examples, config docs
- Benchmarked: Validate performance and memory on real datasets

