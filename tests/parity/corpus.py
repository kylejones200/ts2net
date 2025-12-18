"""
Test corpus generation for parity testing.

Generates deterministic time series covering edge cases and common patterns.
"""
import numpy as np
from typing import Dict, Callable


def _generate_ar1(n: int, phi: float, rng: np.random.RandomState) -> np.ndarray:
    """Generate AR(1) process: x_t = phi * x_{t-1} + eps_t."""
    x = np.zeros(n)
    x[0] = rng.randn()
    for t in range(1, n):
        x[t] = phi * x[t-1] + rng.randn()
    return x


def _add_spikes(x: np.ndarray, rng: np.random.RandomState, n_spikes: int = 5) -> np.ndarray:
    """Add random spikes to a series."""
    x = x.copy()
    spike_idx = rng.choice(len(x), size=n_spikes, replace=False)
    x[spike_idx] += rng.randn(n_spikes) * 5
    return x


def generate_test_corpus(seed: int = 42) -> Dict[str, Callable]:
    """
    Generate deterministic test series corpus.
    
    Returns dictionary mapping series name to generator function.
    Each generator takes n (length) and returns np.ndarray.
    """
    rng = np.random.RandomState(seed)
    
    def _sine_clean(n):
        return np.sin(np.linspace(0, 4*np.pi, n))
    
    def _sine_noise(n):
        rng_local = np.random.RandomState(seed)
        return np.sin(np.linspace(0, 4*np.pi, n)) + 0.1 * rng_local.randn(n)
    
    def _cosine_clean(n):
        return np.cos(np.linspace(0, 6*np.pi, n))
    
    def _linear_trend(n):
        return np.linspace(0, 10, n)
    
    def _quadratic(n):
        t = np.linspace(-2, 2, n)
        return t**2
    
    def _random_walk(n):
        rng_local = np.random.RandomState(seed)
        return np.cumsum(rng_local.randn(n))
    
    def _ar1_positive(n):
        rng_local = np.random.RandomState(seed)
        return _generate_ar1(n, 0.7, rng_local)
    
    def _ar1_negative(n):
        rng_local = np.random.RandomState(seed)
        return _generate_ar1(n, -0.5, rng_local)
    
    def _white_noise(n):
        rng_local = np.random.RandomState(seed)
        return rng_local.randn(n)
    
    def _uniform_noise(n):
        rng_local = np.random.RandomState(seed)
        return rng_local.uniform(-1, 1, n)
    
    def _constant_segments(n):
        """Piecewise constant with 3 levels."""
        seg_len = n // 3
        return np.concatenate([
            np.ones(seg_len),
            2 * np.ones(seg_len),
            3 * np.ones(n - 2*seg_len)
        ])
    
    def _repeated_values(n):
        """Series with many repeated values (ties)."""
        rng_local = np.random.RandomState(seed)
        return rng_local.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n)
    
    def _spikes(n):
        """Zero baseline with random spikes."""
        rng_local = np.random.RandomState(seed)
        x = np.zeros(n)
        return _add_spikes(x, rng_local, n_spikes=max(3, n//20))
    
    def _sawtooth(n):
        """Sawtooth wave."""
        return np.mod(np.linspace(0, 10, n), 1.0)
    
    def _step_function(n):
        """Step function with multiple levels."""
        steps = np.zeros(n)
        step_size = n // 5
        for i in range(5):
            steps[i*step_size:(i+1)*step_size] = i + 1
        return steps
    
    def _alternating(n):
        """Alternating between two values."""
        return np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    
    def _exponential_growth(n):
        """Exponential growth."""
        return np.exp(np.linspace(0, 2, n))
    
    def _logistic(n):
        """Logistic curve."""
        t = np.linspace(-6, 6, n)
        return 1 / (1 + np.exp(-t))
    
    def _mixed_frequencies(n):
        """Sum of multiple sine waves."""
        t = np.linspace(0, 4*np.pi, n)
        return np.sin(t) + 0.5*np.sin(3*t) + 0.25*np.sin(5*t)
    
    def _lorenz_x(n):
        """X component of Lorenz attractor (simplified)."""
        rng_local = np.random.RandomState(seed)
        # Simplified: use AR process as proxy
        return _generate_ar1(n, 0.9, rng_local) + 0.5 * rng_local.randn(n)
    
    # Size-specific series
    def _short_10():
        return np.sin(np.linspace(0, 2*np.pi, 10))
    
    def _short_20():
        return np.sin(np.linspace(0, 2*np.pi, 20))
    
    def _medium_50():
        return np.sin(np.linspace(0, 4*np.pi, 50))
    
    def _medium_100():
        return np.sin(np.linspace(0, 4*np.pi, 100))
    
    def _large_500():
        return np.sin(np.linspace(0, 8*np.pi, 500))
    
    corpus = {
        # Clean patterns
        "sine_clean": _sine_clean,
        "sine_noise": _sine_noise,
        "cosine_clean": _cosine_clean,
        "linear_trend": _linear_trend,
        "quadratic": _quadratic,
        
        # Stochastic processes
        "random_walk": _random_walk,
        "ar1_positive": _ar1_positive,
        "ar1_negative": _ar1_negative,
        "white_noise": _white_noise,
        "uniform_noise": _uniform_noise,
        
        # Edge cases
        "constant_segments": _constant_segments,
        "repeated_values": _repeated_values,
        "spikes": _spikes,
        "sawtooth": _sawtooth,
        "step_function": _step_function,
        "alternating": _alternating,
        
        # Complex patterns
        "exponential_growth": _exponential_growth,
        "logistic": _logistic,
        "mixed_frequencies": _mixed_frequencies,
        "lorenz_x": _lorenz_x,
        
        # Fixed sizes (for exact edge comparison)
        "short_10": lambda n=None: _short_10(),
        "short_20": lambda n=None: _short_20(),
        "medium_50": lambda n=None: _medium_50(),
        "medium_100": lambda n=None: _medium_100(),
        "large_500": lambda n=None: _large_500(),
    }
    
    return corpus


def save_corpus_to_csv(output_dir: str = "tests/parity/data", seed: int = 42):
    """Generate and save all test series to CSV files."""
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    corpus = generate_test_corpus(seed)
    
    # Default length for parameterized series
    default_n = 100
    
    for name, generator in corpus.items():
        try:
            # Try calling with n parameter
            x = generator(default_n)
        except TypeError:
            # Fixed-size series
            x = generator()
        
        # Save to CSV
        csv_path = output_path / f"{name}.csv"
        np.savetxt(csv_path, x, delimiter=",", fmt="%.16e")
        print(f"Saved {name}: n={len(x)}")
    
    print(f"\nGenerated {len(corpus)} test series in {output_path}")


if __name__ == "__main__":
    save_corpus_to_csv()

