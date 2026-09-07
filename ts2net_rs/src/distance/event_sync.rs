//! Event synchronization between two point processes.

/// Outcome of an [`event_sync`] comparison of two event trains.
#[derive(Debug, Clone, PartialEq)]
pub struct EventSync {
    /// Count of events in the second train that follow a synchronised event in the first.
    pub c12: f64,
    /// Count of events in the first train that follow a synchronised event in the second.
    pub c21: f64,
    /// Count of exactly simultaneous events, which are attributed to neither direction.
    pub ties: f64,
    /// `c12` normalised by the number of events in the first train.
    pub q12: f64,
    /// `c21` normalised by the number of events in the second train.
    pub q21: f64,
    /// Symmetric synchronization strength, the mean of `q12` and `q21`.
    pub q: f64,
    /// Absolute lag of every synchronised, non-simultaneous pair.
    pub delays: Vec<f64>,
}

/// Event synchronization between two sorted event-time trains.
///
/// With `adaptive` the coincidence window for each pair is half the smaller of
/// the two local inter-event intervals; otherwise it is a fixed one sample.
/// `tau_max` caps the window regardless of which rule produced it.
pub fn event_sync(e1: &[usize], e2: &[usize], adaptive: bool, tau_max: Option<f64>) -> EventSync {
    let a = e1;
    let b = e2;
    let n1 = a.len();
    let n2 = b.len();
    let mut c12 = 0.0f64;
    let mut c21 = 0.0f64;
    let mut ties = 0.0f64;
    let mut delays: Vec<f64> = Vec::with_capacity(n1 + n2);
    let tmax = tau_max.unwrap_or(f64::INFINITY);
    for i in 0..n1 {
        let ti = a[i] as f64;
        for j in 0..n2 {
            let tj = b[j] as f64;
            let tau = if adaptive {
                let dl = if i == 0 && n1 > 1 {
                    a[i + 1] - a[i]
                } else if i > 0 {
                    a[i] - a[i - 1]
                } else {
                    1
                };
                let dr = if j == 0 && n2 > 1 {
                    b[j + 1] - b[j]
                } else if j > 0 {
                    b[j] - b[j - 1]
                } else {
                    1
                };
                0.5f64 * ((dl.min(dr)) as f64)
            } else {
                1.0
            };
            if (ti - tj).abs() <= tau && (ti - tj).abs() <= tmax {
                if ti < tj {
                    c12 += 1.0;
                    delays.push(tj - ti);
                } else if ti > tj {
                    c21 += 1.0;
                    delays.push(ti - tj);
                } else {
                    ties += 1.0;
                }
            }
        }
    }
    let q12 = if n1 > 0 { c12 / (n1 as f64) } else { 0.0 };
    let q21 = if n2 > 0 { c21 / (n2 as f64) } else { 0.0 };
    let q = if n1 + n2 > 0 { (q12 + q21) / 2.0 } else { 0.0 };
    EventSync {
        c12,
        c21,
        ties,
        q12,
        q21,
        q,
        delays,
    }
}

impl EventSync {
    /// Flatten to the packed layout the Python binding returns.
    ///
    /// The first seven entries are `c12, c21, ties, q12, q21, q, n_delays`,
    /// followed by `n_delays` lag values.
    pub fn into_packed(self) -> Vec<f64> {
        let mut out = vec![
            self.c12,
            self.c21,
            self.ties,
            self.q12,
            self.q21,
            self.q,
            self.delays.len() as f64,
        ];
        out.extend(self.delays);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_trains_are_all_ties() {
        let e = [0usize, 5, 10];
        let r = event_sync(&e, &e, false, None);
        assert_eq!(r.ties, 3.0);
        assert_eq!(r.c12, 0.0);
        assert_eq!(r.c21, 0.0);
        assert!(r.delays.is_empty());
    }

    #[test]
    fn a_uniform_lag_of_one_is_counted_in_one_direction() {
        let a = [0usize, 5, 10];
        let b = [1usize, 6, 11];
        let r = event_sync(&a, &b, false, None);
        assert_eq!(r.c12, 3.0);
        assert_eq!(r.c21, 0.0);
        assert_eq!(r.delays, vec![1.0, 1.0, 1.0]);
        assert_eq!(r.q12, 1.0);
    }

    #[test]
    fn tau_max_suppresses_coincidences() {
        let a = [0usize, 5];
        let b = [1usize, 6];
        let r = event_sync(&a, &b, false, Some(0.5));
        assert_eq!(r.c12, 0.0);
    }

    #[test]
    fn packed_layout_matches_the_python_contract() {
        let a = [0usize, 5, 10];
        let b = [1usize, 6, 11];
        let packed = event_sync(&a, &b, false, None).into_packed();
        assert_eq!(packed.len(), 7 + 3);
        assert_eq!(packed[6], 3.0);
        assert_eq!(&packed[7..], &[1.0, 1.0, 1.0]);
    }
}
