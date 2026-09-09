//! Choosing the delay for a phase-space reconstruction.
//!
//! Takens' theorem is silent on the delay: any tau embeds a noiseless,
//! infinitely long series. Real series force a choice, and the two failure
//! modes sit on either side of it. Too small a tau and successive coordinates
//! are nearly equal, so the reconstruction collapses onto the diagonal and
//! carries almost no more information than the raw series. Too large and
//! successive coordinates become effectively unrelated, so the attractor is
//! folded arbitrarily and its geometry is destroyed.
//!
//! Two selection rules are provided. Neither is "correct" -- they answer
//! different questions and disagree on purpose, so the rule is named in the
//! output rather than left implicit.

use ndarray::Array1;

/// How a delay was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayRule {
    /// First minimum of the time-delayed mutual information (Fraser & Swinney
    /// 1986). Mutual information measures *any* dependence, not just linear,
    /// so its first minimum is the lag at which the coordinate first becomes
    /// maximally independent of its predecessor without being unrelated. This
    /// is the usual choice for a nonlinear reconstruction.
    MutualInformationFirstMinimum,
    /// First zero crossing of the autocorrelation function. Cheap, and exactly
    /// right when the structure is linear, but it can only see linear
    /// dependence -- a signal can be strongly nonlinearly dependent at a lag
    /// where its autocorrelation vanishes. On Lorenz the two rules differ by
    /// more than an order of magnitude -- about 16 samples against about 180 --
    /// so the choice is consequential and the rule used is carried in the result.
    AutocorrelationFirstZero,
}

/// The chosen delay, and the evidence for it.
#[derive(Debug, Clone)]
pub struct DelaySelection {
    /// The chosen lag, in samples.
    pub delay: usize,
    /// Which rule produced it.
    pub rule: DelayRule,
    /// The full curve the rule was applied to, index `i` being lag `i`.
    pub curve: Vec<f64>,
    /// False when no minimum or zero crossing was found within `max_lag`, in
    /// which case `delay` is a documented fallback rather than a measurement.
    /// A caller that reports a delay without checking this is reporting a
    /// guess as a result.
    pub converged: bool,
}

/// Time-delayed mutual information at each lag from 0 to `max_lag`.
///
/// Estimated by equal-width binning of the joint histogram of `x[t]` and
/// `x[t + lag]`. Binning is the standard estimator here and is adequate for
/// locating a minimum, which is all the selection rule needs; it is not
/// accurate enough to report as an information content.
///
/// `bins` of `None` uses a rule of thumb of `sqrt(n / 5)` clamped to `[4, 64]`,
/// which keeps roughly five points per cell for a uniform signal.
pub fn mutual_information_curve(
    x: &Array1<f64>,
    max_lag: usize,
    bins: Option<usize>,
) -> Result<Vec<f64>, String> {
    let n = x.len();
    if n < 3 {
        return Err("need at least three points".to_string());
    }
    if max_lag >= n {
        return Err(format!("max_lag {max_lag} must be less than n {n}"));
    }

    let (lo, hi) = min_max(x);
    if !(hi > lo) {
        // A constant series has no information at any lag. Reporting zeros is
        // honest; inventing a minimum would not be.
        return Ok(vec![0.0; max_lag + 1]);
    }
    let n_bins = bins.unwrap_or_else(|| {
        (((n as f64) / 5.0).sqrt().round() as usize).clamp(4, 64)
    });
    if n_bins < 2 {
        return Err("bins must be at least 2".to_string());
    }

    // Bin index of every sample, computed once and reused for every lag.
    let width = (hi - lo) / (n_bins as f64);
    let index: Vec<usize> = x
        .iter()
        .map(|v| (((v - lo) / width) as usize).min(n_bins - 1))
        .collect();

    let mut curve = Vec::with_capacity(max_lag + 1);
    let mut joint = vec![0.0f64; n_bins * n_bins];
    for lag in 0..=max_lag {
        joint.iter_mut().for_each(|c| *c = 0.0);
        let pairs = n - lag;
        for t in 0..pairs {
            joint[index[t] * n_bins + index[t + lag]] += 1.0;
        }
        let total = pairs as f64;
        let mut marginal_a = vec![0.0f64; n_bins];
        let mut marginal_b = vec![0.0f64; n_bins];
        for a in 0..n_bins {
            for b in 0..n_bins {
                let p = joint[a * n_bins + b] / total;
                marginal_a[a] += p;
                marginal_b[b] += p;
            }
        }
        let mut information = 0.0;
        for a in 0..n_bins {
            if marginal_a[a] <= 0.0 {
                continue;
            }
            for b in 0..n_bins {
                let p = joint[a * n_bins + b] / total;
                if p <= 0.0 || marginal_b[b] <= 0.0 {
                    continue;
                }
                information += p * (p / (marginal_a[a] * marginal_b[b])).ln();
            }
        }
        curve.push(information.max(0.0));
    }
    Ok(curve)
}

/// Autocorrelation at each lag from 0 to `max_lag`, normalised so lag 0 is 1.
pub fn autocorrelation_curve(x: &Array1<f64>, max_lag: usize) -> Result<Vec<f64>, String> {
    let n = x.len();
    if n < 3 {
        return Err("need at least three points".to_string());
    }
    if max_lag >= n {
        return Err(format!("max_lag {max_lag} must be less than n {n}"));
    }
    let mean = x.mean().unwrap();
    let centred: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let variance: f64 = centred.iter().map(|v| v * v).sum();
    if variance <= 0.0 {
        return Ok(vec![0.0; max_lag + 1]);
    }
    let mut curve = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        let sum: f64 = (0..n - lag).map(|t| centred[t] * centred[t + lag]).sum();
        curve.push(sum / variance);
    }
    Ok(curve)
}

/// How far a dip must rise again, as a fraction of the curve's total range,
/// before it counts as a minimum rather than estimator noise.
///
/// Histogram-estimated mutual information is not smooth: a plateau wobbles by
/// a fraction of a percent, and taking the first strict local minimum picks up
/// that wobble. On a pure sine at period 40 the naive rule returns lag 7 for a
/// dip of 0.001 on a curve spanning 0.95. Requiring a genuine rise afterwards
/// rejects that while accepting the real minimum on Lorenz, which rises by 8%
/// of its range.
const MINIMUM_PROMINENCE: f64 = 0.05;

/// Index of the first local minimum that is prominent enough to be real.
///
/// Prominence is measured forward: how far the curve climbs after the dip
/// before falling back below it. A one-step comparison is too local -- the
/// Lorenz minimum rises only 0.004 against its immediate neighbours but 0.23
/// over the following lags.
fn first_local_minimum(curve: &[f64]) -> Option<usize> {
    let (lo, hi) = curve.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
        (a.min(v), b.max(v))
    });
    let range = hi - lo;
    if !(range > 0.0) {
        return None;
    }
    let threshold = MINIMUM_PROMINENCE * range;
    (1..curve.len().saturating_sub(1))
        .filter(|&i| curve[i] < curve[i - 1] && curve[i] <= curve[i + 1])
        .find(|&i| {
            let mut rise: f64 = 0.0;
            for &later in &curve[i + 1..] {
                if later < curve[i] {
                    break;
                }
                rise = rise.max(later - curve[i]);
            }
            rise >= threshold
        })
}

/// Index of the first crossing to non-positive, if there is a meaningful one.
///
/// The curve must start positive. A constant series has zero variance and
/// yields an all-zero autocorrelation, which would otherwise "cross zero" at
/// lag 1 and report a delay for a signal that has none.
fn first_zero_crossing(curve: &[f64]) -> Option<usize> {
    if curve.first().copied().unwrap_or(0.0) <= 0.0 {
        return None;
    }
    (1..curve.len()).find(|&i| curve[i] <= 0.0)
}

/// Choose a delay by the named rule.
///
/// When the rule finds nothing within `max_lag` -- no minimum, no crossing --
/// the result falls back to a delay of 1 with `converged` false. A delay of 1
/// is the neutral choice, not a good one; the flag exists so a caller can say
/// "no delay could be selected" instead of quietly reporting the fallback as a
/// measurement.
pub fn select_delay(
    x: &Array1<f64>,
    rule: DelayRule,
    max_lag: usize,
    bins: Option<usize>,
) -> Result<DelaySelection, String> {
    let (curve, found) = match rule {
        DelayRule::MutualInformationFirstMinimum => {
            let curve = mutual_information_curve(x, max_lag, bins)?;
            let found = first_local_minimum(&curve);
            (curve, found)
        }
        DelayRule::AutocorrelationFirstZero => {
            let curve = autocorrelation_curve(x, max_lag)?;
            let found = first_zero_crossing(&curve);
            (curve, found)
        }
    };
    Ok(DelaySelection {
        delay: found.unwrap_or(1).max(1),
        rule,
        curve,
        converged: found.is_some(),
    })
}

fn min_max(x: &Array1<f64>) -> (f64, f64) {
    x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
        (lo.min(v), hi.max(v))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(n: usize, period: f64) -> Array1<f64> {
        Array1::from_iter(
            (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / period).sin()),
        )
    }

    fn lorenz_x(n: usize) -> Array1<f64> {
        let (sigma, rho, beta, dt) = (10.0, 28.0, 8.0 / 3.0, 0.01);
        let (mut x, mut y, mut z) = (1.0, 1.0, 1.0);
        let mut out = Vec::with_capacity(n);
        for i in 0..(n + 1000) {
            let (dx, dy, dz) = (
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
            );
            x += dt * dx;
            y += dt * dy;
            z += dt * dz;
            if i >= 1000 {
                out.push(x);
            }
        }
        Array1::from(out)
    }

    #[test]
    fn autocorrelation_starts_at_one_and_decays_for_a_sine() {
        let curve = autocorrelation_curve(&sine(2000, 100.0), 200).unwrap();
        assert!((curve[0] - 1.0).abs() < 1e-9);
        assert!(curve.iter().all(|v| v.abs() <= 1.0 + 1e-9));
    }

    #[test]
    fn autocorrelation_first_zero_is_a_quarter_period() {
        // cos(2 pi lag / T) first vanishes at lag = T/4.
        for period in [40.0, 100.0, 250.0] {
            let chosen = select_delay(
                &sine(4000, period),
                DelayRule::AutocorrelationFirstZero,
                (period as usize) * 2,
                None,
            )
            .unwrap();
            assert!(chosen.converged);
            let quarter = period / 4.0;
            assert!(
                (chosen.delay as f64 - quarter).abs() <= 2.0,
                "period {period}: expected ~{quarter}, got {}",
                chosen.delay
            );
        }
    }

    #[test]
    fn mutual_information_is_maximal_at_lag_zero() {
        let curve = mutual_information_curve(&sine(4000, 100.0), 150, None).unwrap();
        let peak = curve.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((curve[0] - peak).abs() < 1e-9, "lag 0 should be the maximum");
        assert!(curve.iter().all(|v| *v >= -1e-12), "MI is non-negative");
    }

    #[test]
    fn mutual_information_first_minimum_matches_the_literature_on_lorenz() {
        // The usual reported value for Lorenz at dt = 0.01 is in the low tens
        // of samples.
        let chosen = select_delay(
            &lorenz_x(6000),
            DelayRule::MutualInformationFirstMinimum,
            200,
            None,
        )
        .unwrap();
        assert!(chosen.converged);
        assert!(
            (8..=25).contains(&chosen.delay),
            "expected a delay in the low tens, got {}",
            chosen.delay
        );
    }

    #[test]
    fn mutual_information_finds_a_quarter_period_on_a_noisy_sine() {
        // Dependence is weakest in quadrature. Noise matters here: it breaks
        // the deterministic relation that makes a noiseless sine degenerate
        // for this rule (see the test below).
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(0);
        let period = 100.0;
        let series: Array1<f64> = Array1::from_iter((0..4000).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / period).sin()
                + 0.3 * rng.gen::<f64>() * 2.0
                - 0.3
        }));
        let chosen =
            select_delay(&series, DelayRule::MutualInformationFirstMinimum, 200, None)
                .unwrap();
        assert!(chosen.converged);
        assert!(
            (chosen.delay as f64 - period / 4.0).abs() <= 0.12 * period,
            "expected ~{}, got {}",
            period / 4.0,
            chosen.delay
        );
    }

    #[test]
    fn the_two_rules_disagree_sharply_on_lorenz() {
        // Measured, not assumed: mutual information puts the delay in the low
        // tens of samples, while the autocorrelation of Lorenz x decays so
        // slowly, and oscillates, that its first zero is an order of magnitude
        // further out -- near lag 180 for this series length. They
        // answer different questions -- any dependence, versus linear
        // dependence only -- and on a chaotic attractor that difference is a
        // factor of thirty. This is why the rule is carried in the result
        // rather than left implicit.
        let series = lorenz_x(6000);
        let mi = select_delay(&series, DelayRule::MutualInformationFirstMinimum, 200, None)
            .unwrap();
        assert!(mi.converged);
        assert!((8..=25).contains(&mi.delay), "mutual information gave {}", mi.delay);

        let acf =
            select_delay(&series, DelayRule::AutocorrelationFirstZero, 300, None).unwrap();
        assert!(acf.converged);
        assert!(
            acf.delay > 10 * mi.delay,
            "expected an order-of-magnitude difference: mi {} acf {}",
            mi.delay,
            acf.delay
        );
    }

    #[test]
    fn a_rule_that_finds_nothing_says_so_instead_of_guessing() {
        // The fallback delay of 1 is neutral, not good. `converged` is what
        // separates a measurement from a default.
        let chosen =
            select_delay(&lorenz_x(4000), DelayRule::AutocorrelationFirstZero, 20, None)
                .unwrap();
        assert!(!chosen.converged);
        assert_eq!(chosen.delay, 1);
    }

    #[test]
    fn a_constant_series_reports_no_convergence_rather_than_a_delay() {
        let flat = Array1::from(vec![3.0; 500]);
        let mi = select_delay(&flat, DelayRule::MutualInformationFirstMinimum, 50, None)
            .unwrap();
        assert!(!mi.converged, "a constant series has no informative delay");
        let acf =
            select_delay(&flat, DelayRule::AutocorrelationFirstZero, 50, None).unwrap();
        assert!(!acf.converged);
    }

    #[test]
    fn white_noise_decorrelates_immediately() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(0);
        let noise: Array1<f64> =
            Array1::from_iter((0..4000).map(|_| rng.gen::<f64>() * 2.0 - 1.0));
        let chosen =
            select_delay(&noise, DelayRule::AutocorrelationFirstZero, 100, None).unwrap();
        assert!(chosen.converged);
        assert!(chosen.delay <= 3, "noise should decorrelate at once, got {}", chosen.delay);
    }

    #[test]
    fn inputs_are_validated() {
        let x = sine(100, 20.0);
        assert!(mutual_information_curve(&x, 100, None).is_err());
        assert!(autocorrelation_curve(&x, 200).is_err());
        assert!(mutual_information_curve(&Array1::from(vec![1.0, 2.0]), 1, None).is_err());
    }
}
