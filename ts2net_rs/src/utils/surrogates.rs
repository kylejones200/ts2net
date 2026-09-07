//! Surrogate series for hypothesis testing.

use std::sync::Arc;

use ndarray::Array1;
use num_complex::Complex;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rustfft::{Fft, FftPlanner};

/// Amplitude spectrum of `x` under an already-planned forward transform.
fn amplitude_spectrum(x: &[f64], fft: &Arc<dyn Fft<f64>>) -> Vec<f64> {
    let mut spec: Vec<Complex<f64>> = x.iter().map(|&r| Complex { re: r, im: 0.0 }).collect();
    fft.process(&mut spec);
    spec.iter()
        .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
        .collect()
}

/// Replace the amplitude spectrum of `y` with `mag_target`, keeping its phases.
fn impose_spectrum(
    y: &[f64],
    mag_target: &[f64],
    fft: &Arc<dyn Fft<f64>>,
    ifft: &Arc<dyn Fft<f64>>,
) -> Vec<f64> {
    let n = y.len();
    let mut spec: Vec<Complex<f64>> = y.iter().map(|&r| Complex { re: r, im: 0.0 }).collect();
    fft.process(&mut spec);
    for k in 0..n {
        let c = &spec[k];
        let phase = c.im.atan2(c.re);
        spec[k] = Complex {
            re: mag_target[k] * phase.cos(),
            im: mag_target[k] * phase.sin(),
        };
    }
    ifft.process(&mut spec);
    spec.iter().map(|c| c.re / (n as f64)).collect()
}

/// Indices sorting `x` ascending: element `k` is the index of the k-th smallest.
fn argsort(x: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
    idx
}

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

/// Iterative amplitude-adjusted Fourier transform (IAAFT) surrogate.
///
/// Starting from a random permutation of `v`, each iteration imposes the
/// amplitude spectrum of `v` on the working series and then restores `v`'s
/// exact value distribution by rank. The returned surrogate is therefore
/// always a permutation of `v`, and its power spectrum approaches `v`'s as
/// `iters` grows -- the two defining properties of IAAFT.
///
/// `seed` makes the run deterministic.
pub fn iaaft(v: &Array1<f64>, iters: usize, seed: u64) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let original: Vec<f64> = v.iter().copied().collect();
    let mag_t = amplitude_spectrum(&original, &fft);
    let mut sorted_v = original.clone();
    sorted_v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut rng = StdRng::seed_from_u64(seed);
    let mut y = original;
    y.shuffle(&mut rng);
    for _ in 0..iters {
        let y2 = impose_spectrum(&y, &mag_t, &fft, &ifft);
        // Amplitude adjustment: the position holding the k-th smallest value
        // of y2 takes the k-th smallest value of the original series.
        let ridx = argsort(&y2);
        let mut out = vec![0.0; n];
        for k in 0..n {
            out[ridx[k]] = sorted_v[k];
        }
        y = out;
    }
    y
}

/// The pre-0.10 `iaaft`, kept only to reproduce previously published results.
///
/// This is **not** IAAFT. Its rank-matching step assigns the working series'
/// own sorted values rather than the original's, so the surrogate carries the
/// rank ordering of `v` and the amplitudes of the spectrum-corrected series --
/// it never restores `v`'s value distribution. Prefer [`iaaft`].
pub fn iaaft_legacy(v: &Array1<f64>, iters: usize, seed: u64) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let original: Vec<f64> = v.iter().copied().collect();
    let mag_t = amplitude_spectrum(&original, &fft);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut y = original.clone();
    y.shuffle(&mut rng);
    for _ in 0..iters {
        let y2 = impose_spectrum(&y, &mag_t, &fft, &ifft);
        let idx = argsort(&original);
        let ridx = argsort(&y2);
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
        let mut r = vec![0usize; x.len()];
        for (rank, &i) in argsort(x).iter().enumerate() {
            r[i] = rank;
        }
        r
    }

    fn spectrum(x: &[f64]) -> Vec<f64> {
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(x.len());
        amplitude_spectrum(x, &fft)
    }

    /// Relative L2 distance between two amplitude spectra.
    fn spectral_error(a: &[f64], b: &[f64]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum();
        let den: f64 = a.iter().map(|p| p * p).sum();
        (num / den).sqrt()
    }

    #[test]
    fn empty_input_does_not_panic() {
        let empty = Array1::<f64>::zeros(0);
        assert!(surrogate_phase(&empty, 1).is_empty());
        assert!(iaaft(&empty, 10, 1).is_empty());
        assert!(iaaft_legacy(&empty, 10, 1).is_empty());
    }

    #[test]
    fn surrogates_are_deterministic_for_a_given_seed() {
        let v = signal(64);
        assert_eq!(surrogate_phase(&v, 7), surrogate_phase(&v, 7));
        assert_eq!(iaaft(&v, 20, 7), iaaft(&v, 20, 7));
        assert_eq!(iaaft_legacy(&v, 20, 7), iaaft_legacy(&v, 20, 7));
        assert_ne!(surrogate_phase(&v, 7), surrogate_phase(&v, 8));
        assert_ne!(iaaft(&v, 20, 7), iaaft(&v, 20, 8));
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
    fn iaaft_preserves_the_exact_value_distribution() {
        let v = signal(64);
        let s = iaaft(&v, 50, 3);
        assert_eq!(s.len(), v.len());
        let mut original: Vec<f64> = v.to_vec();
        let mut surrogate = s;
        original.sort_by(|a, b| a.partial_cmp(b).unwrap());
        surrogate.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 0..original.len() {
            assert_relative_eq!(original[k], surrogate[k], epsilon = 1e-12);
        }
    }

    #[test]
    fn iaaft_is_a_permutation_of_the_original() {
        let v = signal(48);
        let s = iaaft(&v, 30, 11);
        let mut seen = vec![false; v.len()];
        for value in &s {
            let pos = v
                .iter()
                .enumerate()
                .find(|(i, o)| !seen[*i] && (*o - value).abs() < 1e-12)
                .map(|(i, _)| i);
            assert!(pos.is_some(), "surrogate value {value} is not in the original");
            seen[pos.unwrap()] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn iaaft_targets_the_original_power_spectrum() {
        // IAAFT converges to a fixed point that trades exact amplitudes
        // against an exact spectrum, so the residual spectral error is small
        // but never zero. Judge it against a shuffle -- same value
        // distribution, spectrum destroyed -- rather than a magic constant.
        let v = signal(128);
        let target = spectrum(v.as_slice().unwrap());

        let mut shuffled: Vec<f64> = v.to_vec();
        shuffled.shuffle(&mut StdRng::seed_from_u64(5));
        let baseline = spectral_error(&target, &spectrum(&shuffled));

        let one = spectral_error(&target, &spectrum(&iaaft(&v, 1, 5)));
        let many = spectral_error(&target, &spectrum(&iaaft(&v, 200, 5)));

        assert!(
            many < baseline / 5.0,
            "iaaft should recover the spectrum far better than a shuffle: \
             {many} vs baseline {baseline}"
        );
        assert!(
            many <= one,
            "more iterations should not worsen the spectrum: {one} -> {many}"
        );
    }

    #[test]
    fn iaaft_is_not_the_identity() {
        let v = signal(64);
        let s = iaaft(&v, 50, 3);
        assert!(
            v.iter().zip(&s).any(|(a, b)| (a - b).abs() > 1e-9),
            "surrogate is identical to the input"
        );
    }

    #[test]
    fn legacy_reproduces_the_pre_fix_behaviour() {
        // The legacy path keeps the original's rank ordering and does not
        // restore its amplitudes -- the defect the canonical fn no longer has.
        let v = signal(64);
        let s = iaaft_legacy(&v, 50, 3);
        assert_eq!(ranks(v.as_slice().unwrap()), ranks(&s));
        let mut original: Vec<f64> = v.to_vec();
        let mut surrogate = s;
        original.sort_by(|a, b| a.partial_cmp(b).unwrap());
        surrogate.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            original
                .iter()
                .zip(&surrogate)
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "legacy unexpectedly preserved the value distribution"
        );
    }
}
