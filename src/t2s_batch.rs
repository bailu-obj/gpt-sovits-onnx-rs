//! T2S dynamic-batching control-flow prototype (not wired into inference).
//!
//! # Background
//!
//! Serial inference runs one fragment at a time through FS decoder + stage-decoder AR
//! loop. Python CPUFast batches multiple fragments by padding to a shared `x_length`
//! bucket, running batched FS + AR, then **compacting** finished rows and **shrinking
//! KV views** to each row's valid prefix only (not the padded tail).
//!
//! This module ports that *control-flow* to Rust (`ndarray` workspace rows, not PyTorch
//! tensors). Full ORT batched graphs are **not** wired up yet — see
//! `patch/GPT_SoVITS/export_onnx_v2.py` (`GSV_EXPORT_T2S_BATCH=1`) and
//! `doc/cpu_improvements.md`.
//!
//! # Ship gate
//!
//! Keep serial streaming as the default. Wire ORT batching only when multi-fragment
//! benchmarks show **≥10% E2E** improvement after padding waste and while VITS remains
//! serial (historical upper bound ~1.4× on T2S alone).
//!
//! # Compaction algorithm (stage decoder)
//!
//! 1. Maintain `active_slots: Vec<usize>` — batch row → original fragment index.
//! 2. Each AR step: pass `ik_cache_*` / `iv_cache_*` views `[0..valid_len]` per row
//!    (valid-prefix shrink — same as serial [`KvWorkspace::k_view`]).
//! 3. When row `r` emits EOS: mark fragment finished; **compact** by swapping the last
//!    active row into slot `r` and popping the batch (CPUFast remap).
//! 4. On remap: copy KV valid prefix `[0..valid_len]` from last active row into slot `r`
//!    for every layer (see [`ActiveBatch::compact_finished`]).
//!
//! # Length buckets (FS decoder)
//!
//! Fragments with similar phoneme+ref length (`x_len`) are grouped so batched `x` /
//! `bert` padding waste stays bounded. [`assign_length_buckets`] uses upper-bound
//! bucket edges (default powers of two from 64).
//!
//! # Decision gate
//!
//! Batching stays **off** in production until multi-fragment benchmarks show a win.
//! `cargo test t2s_batch` covers the compaction helpers.

/// Per-fragment T2S stage wall times (milliseconds).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FragmentT2STiming {
    pub fs_decoder_ms: f64,
    pub s_decoder_ms: f64,
    pub semantic_len: usize,
}

impl FragmentT2STiming {
    pub fn t2s_total_ms(&self) -> f64 {
        self.fs_decoder_ms + self.s_decoder_ms
    }
}

/// Metadata needed to bucket fragments before a batched FS pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentSlot {
    pub fragment_index: usize,
    pub x_len: usize,
}

/// Assign each fragment to a length bucket (upper-bound edges).
///
/// Returns `(bucket_upper_bound, fragment_indices_in_bucket)` sorted by bucket edge.
pub fn assign_length_buckets(
    fragments: &[FragmentSlot],
    bucket_edges: &[usize],
) -> Vec<(usize, Vec<usize>)> {
    let mut buckets: Vec<(usize, Vec<usize>)> = bucket_edges
        .iter()
        .map(|&edge| (edge, Vec::new()))
        .collect();

    for frag in fragments {
        let edge = bucket_edges
            .iter()
            .find(|&&e| frag.x_len <= e)
            .copied()
            .unwrap_or(*bucket_edges.last().unwrap_or(&frag.x_len));
        if let Some(b) = buckets.iter_mut().find(|(e, _)| *e == edge) {
            b.1.push(frag.fragment_index);
        } else {
            buckets.push((edge, vec![frag.fragment_index]));
        }
    }

    buckets.retain(|(_, ids)| !ids.is_empty());
    buckets
}

/// Default FS-decoder bucket edges (phoneme+ref token count upper bounds).
pub fn default_bucket_edges() -> Vec<usize> {
    vec![64, 128, 256, 512, 1024, 2048]
}

/// Tracks active batch rows during batched AR and remaps KV on early EOS.
#[derive(Debug, Clone)]
pub struct ActiveBatch {
    /// Batch slot → original fragment index (only active slots).
    pub active_slots: Vec<usize>,
    /// Per batch slot valid KV length (grows each AR step).
    pub valid_lens: Vec<usize>,
    finished: Vec<bool>,
    num_fragments: usize,
}

impl ActiveBatch {
    pub fn new(num_fragments: usize, initial_valid_len: usize) -> Self {
        Self {
            active_slots: (0..num_fragments).collect(),
            valid_lens: vec![initial_valid_len; num_fragments],
            finished: vec![false; num_fragments],
            num_fragments,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.active_slots.len()
    }

    pub fn is_active(&self, fragment_index: usize) -> bool {
        !self.finished[fragment_index]
    }

    pub fn all_finished(&self) -> bool {
        self.active_slots.is_empty()
    }

    /// Mark fragment done and compact batch by swapping last active row into `slot`.
    pub fn compact_finished(&mut self, slot: usize) -> Option<CompactRemap> {
        if slot >= self.active_slots.len() {
            return None;
        }
        let fragment_idx = self.active_slots[slot];
        self.finished[fragment_idx] = true;

        let last = self.active_slots.len() - 1;
        if slot == last {
            self.active_slots.pop();
            self.valid_lens.pop();
            return None;
        }

        let src_slot = last;
        let dst_slot = slot;
        let moved_fragment = self.active_slots[src_slot];
        self.active_slots[dst_slot] = moved_fragment;
        self.valid_lens[dst_slot] = self.valid_lens[src_slot];
        self.active_slots.pop();
        self.valid_lens.pop();

        Some(CompactRemap {
            dst_slot,
            src_slot: last,
            valid_len: self.valid_lens[dst_slot],
        })
    }

    pub fn bump_valid_len(&mut self, slot: usize) {
        if slot < self.valid_lens.len() {
            self.valid_lens[slot] += 1;
        }
    }
}

/// KV row remap triggered when an active row finishes before others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactRemap {
    pub dst_slot: usize,
    pub src_slot: usize,
    pub valid_len: usize,
}

/// Copy valid-prefix KV rows after compaction (one layer, rank-3 `[1, seq, dim]`).
pub fn remap_kv_layer(
    k: &mut ndarray::Array3<f32>,
    v: &mut ndarray::Array3<f32>,
    remap: CompactRemap,
) {
    use ndarray::s;
    let len = remap.valid_len.min(k.shape()[1]);
    if remap.dst_slot >= k.shape()[0] || remap.src_slot >= k.shape()[0] {
        return;
    }
    let src_k = k.slice(s![remap.src_slot, 0..len, ..]).to_owned();
    let src_v = v.slice(s![remap.src_slot, 0..len, ..]).to_owned();
    k.slice_mut(s![remap.dst_slot, 0..len, ..])
        .assign(&src_k);
    v.slice_mut(s![remap.dst_slot, 0..len, ..])
        .assign(&src_v);
}

/// Ideal batched stage-decoder time if all fragments share one AR loop for
/// `max(semantic_len)` steps at the observed per-step cost. Returns `(serial_ms, ideal_batched_ms)`.
pub fn estimate_batched_ar_upper_bound(per_fragment: &[FragmentT2STiming]) -> (f64, f64) {
    let serial_ms: f64 = per_fragment.iter().map(|t| t.s_decoder_ms).sum();
    let total_steps: usize = per_fragment.iter().map(|t| t.semantic_len).sum();
    if total_steps == 0 || per_fragment.is_empty() {
        return (serial_ms, serial_ms);
    }
    let per_step_ms = serial_ms / total_steps as f64;
    let max_steps = per_fragment
        .iter()
        .map(|t| t.semantic_len)
        .max()
        .unwrap_or(0);
    let ideal_batched_ms = max_steps as f64 * per_step_ms;
    (serial_ms, ideal_batched_ms.min(serial_ms))
}

/// Ideal batched FS-decoder time: one bucketed call instead of N serial calls.
/// Uses the slowest observed FS latency as the batched cost (same `x` bucket).
pub fn estimate_batched_fs_upper_bound(per_fragment: &[FragmentT2STiming]) -> (f64, f64) {
    let serial_ms: f64 = per_fragment.iter().map(|t| t.fs_decoder_ms).sum();
    let batched_ms = per_fragment
        .iter()
        .map(|t| t.fs_decoder_ms)
        .fold(0.0_f64, f64::max);
    (serial_ms, batched_ms.min(serial_ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn length_buckets_group_by_x_len() {
        let frags = vec![
            FragmentSlot {
                fragment_index: 0,
                x_len: 40,
            },
            FragmentSlot {
                fragment_index: 1,
                x_len: 100,
            },
            FragmentSlot {
                fragment_index: 2,
                x_len: 90,
            },
        ];
        let buckets = assign_length_buckets(&frags, &default_bucket_edges());
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0], (64, vec![0]));
        assert_eq!(buckets[1], (128, vec![1, 2]));
    }

    #[test]
    fn compact_finished_swaps_last_active_row() {
        let mut batch = ActiveBatch::new(3, 10);
        batch.valid_lens = vec![10, 11, 12];
        let remap = batch.compact_finished(0).expect("swap expected");
        assert_eq!(remap.dst_slot, 0);
        assert_eq!(remap.src_slot, 2);
        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.active_slots, vec![2, 1]);
        assert!(batch.finished[0]);
        assert!(!batch.finished[1]);
        assert!(!batch.finished[2]);
    }

    #[test]
    fn compact_last_slot_pops_without_remap() {
        let mut batch = ActiveBatch::new(2, 5);
        assert!(batch.compact_finished(1).is_none());
        assert_eq!(batch.batch_size(), 1);
        assert_eq!(batch.active_slots, vec![0]);
    }

    #[test]
    fn remap_kv_layer_copies_valid_prefix() {
        let mut k = Array3::<f32>::zeros((3, 8, 2));
        let mut v = Array3::<f32>::zeros((3, 8, 2));
        k.slice_mut(ndarray::s![2, 0..4, ..]).fill(7.0);
        v.slice_mut(ndarray::s![2, 0..4, ..]).fill(3.0);

        remap_kv_layer(
            &mut k,
            &mut v,
            CompactRemap {
                dst_slot: 0,
                src_slot: 2,
                valid_len: 4,
            },
        );

        assert!(k.slice(ndarray::s![0, 0..4, 0]).iter().all(|&x| x == 7.0));
        assert!(v.slice(ndarray::s![0, 0..4, 0]).iter().all(|&x| x == 3.0));
        assert_eq!(k[[0, 4, 0]], 0.0);
    }

    #[test]
    fn ar_upper_bound_uses_max_steps_not_wall_sum() {
        let frags = vec![
            FragmentT2STiming {
                fs_decoder_ms: 20.0,
                s_decoder_ms: 90.0,
                semantic_len: 30,
            },
            FragmentT2STiming {
                fs_decoder_ms: 22.0,
                s_decoder_ms: 120.0,
                semantic_len: 40,
            },
        ];
        let (serial, ideal) = estimate_batched_ar_upper_bound(&frags);
        assert!((serial - 210.0).abs() < 1e-6);
        // per_step = 210/70 = 3; ideal = 40*3 = 120
        assert!((ideal - 120.0).abs() < 1e-6);
        assert!(ideal < serial);
    }
}
