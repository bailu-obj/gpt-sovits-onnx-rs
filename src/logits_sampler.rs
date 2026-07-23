use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

const T2S_DECODER_EOS: i64 = 1024;

/// Finds the token with the highest logit value (argmax).
pub fn argmax(logits: &[f32]) -> i64 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i64)
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    /// Fixed seed for reproducible sampling; `None` uses OS RNG.
    pub seed: Option<u64>,
}

pub struct SamplingParamsBuilder {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: f32,
    seed: Option<u64>,
}

impl SamplingParamsBuilder {
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            seed: None,
        }
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = if temperature >= 0.0 { temperature } else { 1.0 };
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = if repetition_penalty > 0.0 {
            repetition_penalty
        } else {
            1.0
        };
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn build(self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
        }
    }
}

fn apply_repetition_penalty(logits: &mut [f32], previous_tokens: &[i64], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    // Match PyTorch gather/scatter over every prior token (including duplicates).
    for &token_id in previous_tokens {
        let idx = token_id as usize;
        if idx >= logits.len() {
            continue;
        }
        let logit = &mut logits[idx];
        if *logit >= 0.0 {
            *logit /= penalty;
        } else {
            *logit *= penalty;
        }
    }
}

fn apply_top_p(logits: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_l = logits[order[0]];
    let mut exp_sum = 0.0f32;
    for &token_idx in &order {
        exp_sum += (logits[token_idx] - max_l).exp();
    }
    let mut cum = 0.0f32;
    let mut cutoff = order.len();
    for (i, &token_idx) in order.iter().enumerate() {
        cum += (logits[token_idx] - max_l).exp() / exp_sum;
        if cum > top_p && i > 0 {
            cutoff = i;
            break;
        }
    }
    // Keep at least one option (matches PyTorch sorted_indices_to_remove[:, 0] = False).
    cutoff = cutoff.max(1);
    for &rm in &order[cutoff..] {
        logits[rm] = f32::NEG_INFINITY;
    }
}

fn apply_top_k(logits: &mut [f32], top_k: usize) {
    let vocab = logits.len();
    let k = top_k.min(vocab);
    if k == 0 || k >= vocab {
        return;
    }
    // Partial select: O(n) average vs full O(n log n) sort — same threshold semantics.
    let mut pivot: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pivot.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = pivot[k - 1].1;
    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

/// Port of GPT-SoVITS `logits_to_probs` (AR/models/utils.py).
pub fn logits_to_probs(
    logits: &mut [f32],
    previous_tokens: &[i64],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: f32,
) -> Vec<f32> {
    apply_repetition_penalty(logits, previous_tokens, repetition_penalty);

    if let Some(p) = top_p {
        apply_top_p(logits, p);
    }

    let inv_temp = 1.0 / temperature.max(1e-5);
    for logit in logits.iter_mut() {
        *logit *= inv_temp;
    }

    if let Some(k) = top_k {
        apply_top_k(logits, k);
    }

    softmax(logits)
}

/// Gumbel-max draw matching PyTorch `multinomial_sample_one_no_sync`.
fn gumbel_max_sample(probs: &[f32], rng: &mut StdRng) -> i64 {
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (idx, &prob) in probs.iter().enumerate() {
        let u: f32 = rng.random::<f32>().max(1e-10);
        let score = prob / (-u.ln());
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx as i64
}

pub struct Sampler {
    rng: StdRng,
}

impl Sampler {
    pub fn new(_vocab_size: usize) -> Self {
        Self {
            rng: StdRng::from_os_rng(),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn sample(
        &mut self,
        logits: &mut [f32],
        prev_tokens: &[i64],
        params: &SamplingParams,
    ) -> i64 {
        if params.temperature == 0.0 {
            // Match PyTorch: still apply repetition/top-k/top-p, then take the mode.
            logits_to_probs(
                logits,
                prev_tokens,
                1e-5,
                params.top_k,
                params.top_p,
                params.repetition_penalty,
            );
            return argmax(logits);
        }

        let probs = logits_to_probs(
            logits,
            prev_tokens,
            params.temperature,
            params.top_k,
            params.top_p,
            params.repetition_penalty,
        );

        if probs.is_empty() {
            return 0;
        }

        gumbel_max_sample(&probs, &mut self.rng)
    }
}

/// Extract semantic tokens for VITS from the full T2S token sequence.
/// Mirrors Python `pred_semantic[-idx:]` after EOS is removed from `y`.
pub fn extract_semantic_tokens(y_vec: &[i64], prefix_len: usize, stop_idx: usize) -> Vec<i64> {
    let mut y = y_vec.to_vec();
    if y.last().copied() == Some(T2S_DECODER_EOS) {
        y.pop();
    }
    if stop_idx == 0 || y.len() <= prefix_len {
        return Vec::new();
    }
    let start = y.len().saturating_sub(stop_idx);
    y[start..]
        .iter()
        .map(|&token| if token == T2S_DECODER_EOS { 0 } else { token })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_semantic_tokens_matches_python_negative_idx_slice() {
        let prefix_len = 5;
        let mut y = vec![1, 2, 3, 4, 5];
        y.extend((10..10 + 53).map(|v| v as i64));
        y.push(1024);
        let out = extract_semantic_tokens(&y, prefix_len, 52);
        assert_eq!(out.len(), 52);
        assert_eq!(out[0], 11);
        assert_eq!(out[51], 62);
    }

    #[test]
    fn extract_semantic_tokens_after_eos_pop() {
        let prefix_len = 5;
        let y = vec![1, 2, 3, 4, 5, 10, 11, 12, 1024];
        let out = extract_semantic_tokens(&y, prefix_len, 3);
        assert_eq!(out, vec![10, 11, 12]);
    }

    #[test]
    fn gumbel_max_is_deterministic_with_seed() {
        let mut logits = vec![0.1f32; 16];
        logits[3] = 3.0;
        let params = SamplingParamsBuilder::new()
            .temperature(1.0)
            .seed(42)
            .build();
        let mut sampler = Sampler::with_seed(42);
        let mut l1 = logits.clone();
        let mut l2 = logits.clone();
        let a = sampler.sample(&mut l1, &[], &params);
        let mut sampler2 = Sampler::with_seed(42);
        let b = sampler2.sample(&mut l2, &[], &params);
        assert_eq!(a, b);
    }

    #[test]
    fn top_k_select_matches_full_sort_threshold() {
        let mut a = vec![0.1, 5.0, 0.2, 4.0, 0.3, 3.0, 0.4, 2.0];
        let mut b = a.clone();
        apply_top_k(&mut a, 3);
        // Reference full-sort path
        let mut pivot: Vec<(usize, f32)> = b.iter().copied().enumerate().collect();
        pivot.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        let threshold = pivot[2].1;
        for logit in &mut b {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
        assert_eq!(a, b);
    }

    #[test]
    fn default_sampling_path_is_seed_stable() {
        let mut base = vec![0.05f32; 1025];
        base[10] = 2.5;
        base[20] = 2.4;
        base[30] = 2.3;
        base[40] = 2.2;
        base[50] = 1.0;
        let params = SamplingParamsBuilder::new()
            .temperature(1.0)
            .top_k(4)
            .top_p(0.9)
            .repetition_penalty(1.35)
            .seed(7)
            .build();
        let prev = vec![10i64, 20];
        let mut s1 = Sampler::with_seed(7);
        let mut s2 = Sampler::with_seed(7);
        let mut l1 = base.clone();
        let mut l2 = base.clone();
        assert_eq!(s1.sample(&mut l1, &prev, &params), s2.sample(&mut l2, &prev, &params));
    }
}
