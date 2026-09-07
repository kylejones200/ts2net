//! Surrogate series for hypothesis testing.

use ndarray::Array1;
use num_complex::Complex;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rustfft::FftPlanner;

/// Phase-randomised surrogate: keeps the amplitude spectrum, randomises phases.
///
/// The result has the same power spectrum (and so the same linear
/// autocorrelation) as `v` but no nonlinear structure, which is the standard
/// null for tests of nonlinearity.
pub fn surrogate_phase(v: &Array1<f64>, seed: u64) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let mut spec: Vec<Complex<f64>> = v.iter().map(|&r| Complex { re: r, im: 0.0 }).collect();
    fft.process(&mut spec);
    let mag: Vec<f64> = spec
        .iter()
        .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
        .collect();
    let mut rng = StdRng::seed_from_u64(seed);
    for k in 1..(n / 2) {
        let theta: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
        spec[k] = Complex {
            re: mag[k] * theta.cos(),
            im: mag[k] * theta.sin(),
        };
        spec[n - k] = Complex {
            re: spec[k].re,
            im: -spec[k].im,
        };
    }
    if n % 2 == 0 {
        spec[n / 2] = Complex {
            re: mag[n / 2],
            im: 0.0,
        };
    }
    ifft.process(&mut spec);
    spec.iter().map(|c| c.re / (n as f64)).collect()
}

/// Iterative Fourier-transform surrogate with rank matching.
///
/// Each iteration imposes the amplitude spectrum of `v` on the working series,
/// then reorders the result to carry the rank ordering of `v`.
///
/// Note the rank-matching step assigns the working series' **own** sorted
/// values, not `v`'s, so the surrogate reproduces the rank ordering and the
/// power spectrum of `v` but not its value distribution -- textbook IAAFT
/// restores the original amplitudes. This matches the NumPy fallback in
/// `ts2net.stats.stats.iaaft` (`np.sort(y)[np.argsort(np.argsort(x))]`); the
/// two implementations must agree, so neither should be changed alone.
pub fn iaaft(v: &Array1<f64>, iters: usize, seed: u64) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let mut spec_t: Vec<Complex<f64>> = v.iter().map(|&r| Complex { re: r, im: 0.0 }).collect();
    fft.process(&mut spec_t);
    let mag_t: Vec<f64> = spec_t
        .iter()
        .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
        .collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut y: Vec<f64> = v.iter().copied().collect();
    y.shuffle(&mut rng);
    for _ in 0..iters {
        let mut spec: Vec<Complex<f64>> = y.iter().map(|&r| Complex { re: r, im: 0.0 }).collect();
        fft.process(&mut spec);
        for k in 0..n {
            let c = &spec[k];
            let phase = c.im.atan2(c.re);
            spec[k] = Complex {
                re: mag_t[k] * phase.cos(),
                im: mag_t[k] * phase.sin(),
            };
        }
        ifft.process(&mut spec);
        let y2: Vec<f64> = spec.iter().map(|c| c.re / (n as f64)).collect();
        // rank-order match original
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut ridx: Vec<usize> = (0..n).collect();
        ridx.sort_by(|&i, &j| y2[i].partial_cmp(&y2[j]).unwrap());
        let mut out = vec![0.0; n];
        for k in 0..n {
            out[idx[k]] = y2[ridx[k]];
        }
        y = out;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn signal(n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|i| (i as f64 * 0.37).sin() + 0.25 * (i as f64 * 1.1).cos()))
    }

    /// Rank of every element, ties broken by position.
    fn ranks(x: &[f64]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..x.len()).collect();
        order.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
        let mut r = vec![0usize; x.len()];
        for (rank, &i) in order.iter().enumerate() {
            r[i] = rank;
        }
        r
    }

    fn spectrum(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n);
        let mut buf: Vec<Complex<f64>> = x.iter().map(|&r| Complex { re: r, im: 0.0 }).collect();
        fft.process(&mut buf);
        buf.iter()
            .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
            .collect()
    }

    #[test]
    fn empty_input_does_not_panic() {
        let empty = Array1::<f64>::zeros(0);
        assert!(surrogate_phase(&empty, 1).is_empty());
        assert!(iaaft(&empty, 10, 1).is_empty());
    }

    #[test]
    fn surrogates_are_deterministic_for_a_given_seed() {
        let v = signal(64);
        assert_eq!(surrogate_phase(&v, 7), surrogate_phase(&v, 7));
        assert_eq!(iaaft(&v, 20, 7), iaaft(&v, 20, 7));
        assert_ne!(surrogate_phase(&v, 7), surrogate_phase(&v, 8));
    }

    #[test]
    fn phase_surrogate_preserves_the_amplitude_spectrum() {
        let v = signal(64);
        let s = surrogate_phase(&v, 3);
        assert_eq!(s.len(), 64);
        let (a, b) = (spectrum(v.as_slice().unwrap()), spectrum(&s));
        for k in 0..a.len() {
            assert_relative_eq!(a[k], b[k], epsilon = 1e-8);
        }
    }

    #[test]
    fn iaaft_reproduces_the_rank_ordering_of_the_original() {
        let v = signal(64);
        let s = iaaft(&v, 50, 3);
        assert_eq!(s.len(), v.len());
        assert_eq!(ranks(v.as_slice().unwrap()), ranks(&s));
    }

    #[test]
    fn iaaft_does_not_restore_the_original_amplitudes() {
        // Guards the documented deviation from textbook IAAFT: the surrogate
        // carries its own values, not a permutation of the original's.
        let v = signal(64);
        let mut original: Vec<f64> = v.to_vec();
        let mut surrogate = iaaft(&v, 50, 3);
        original.sort_by(|a, b| a.partial_cmp(b).unwrap());
        surrogate.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            original
                .iter()
                .zip(surrogate.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "value distribution unexpectedly preserved; see the note on iaaft"
        );
    }
}
