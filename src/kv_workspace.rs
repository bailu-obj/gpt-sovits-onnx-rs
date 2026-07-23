//! Adaptive, reusable T2S KV-cache workspace.
//!
//! Avoids allocating a fixed 2048-token buffer for every fragment. Capacity
//! starts near the first-stage prefix length plus a modest headroom, then grows
//! geometrically when needed.
//!
//! Supports both cache layouts:
//! - Legacy: `[batch, seq, hidden]` (rank 3) — time-prefix views are contiguous when B=1
//! - Head-major: `[batch, heads, seq, head_dim]` (rank 4) — time-prefix views are
//!   non-contiguous, so [`Self::compact_for_ort`] materializes contiguous ORT feeds

use crate::error::GSVError;
use log::info;
use ndarray::{Array, ArrayBase, ArrayView, Dimension, IxDyn, OwnedRepr, s};

pub type KvDType = f32;

/// Minimum headroom beyond the first-stage prefix for short generations.
const MIN_HEADROOM: usize = 128;
/// Soft upper bound used when growing; still grows further if needed.
const SOFT_CAP: usize = 2048;

fn seq_axis(ndim: usize) -> usize {
    if ndim >= 4 { 2 } else { 1 }
}

/// Model-owned reusable KV buffers for the autoregressive T2S decoder.
pub struct KvWorkspace {
    k_caches: Vec<ArrayBase<OwnedRepr<KvDType>, IxDyn>>,
    v_caches: Vec<ArrayBase<OwnedRepr<KvDType>, IxDyn>>,
    /// Contiguous `[B,H,valid,D]` feeds for ORT when caches are head-major.
    k_ort: Vec<ArrayBase<OwnedRepr<KvDType>, IxDyn>>,
    v_ort: Vec<ArrayBase<OwnedRepr<KvDType>, IxDyn>>,
    capacity: usize,
    num_layers: usize,
}

impl KvWorkspace {
    pub fn new(num_layers: usize) -> Self {
        Self {
            k_caches: Vec::new(),
            v_caches: Vec::new(),
            k_ort: Vec::new(),
            v_ort: Vec::new(),
            capacity: 0,
            num_layers,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_head_major(&self) -> bool {
        self.k_caches.first().map(|a| a.ndim() >= 4).unwrap_or(false)
    }

    /// Suggested initial capacity: prefix + headroom, rounded up to a power of two.
    pub fn suggested_capacity(initial_seq_len: usize) -> usize {
        let needed = initial_seq_len.saturating_add(MIN_HEADROOM).max(256);
        let capped = needed.min(SOFT_CAP).max(initial_seq_len + 1);
        next_pow2(capped)
    }

    /// Ensure workspace can hold `needed` sequence positions matching `template` rank/layout.
    pub fn ensure_capacity(
        &mut self,
        needed: usize,
        template: &ArrayView<'_, KvDType, IxDyn>,
    ) -> Result<(), GSVError> {
        let target = next_pow2(needed.max(1));
        if self.capacity >= target && self.k_caches.len() == self.num_layers {
            return Ok(());
        }

        let mut dims = template.raw_dim().clone();
        let axis = seq_axis(dims.ndim());
        if dims.ndim() < 2 {
            return Err(GSVError::from("KV cache template must have seq dim"));
        }
        dims[axis] = target;

        info!(
            "Allocating KV workspace capacity {} -> {} (layers={}, rank={}, seq_axis={})",
            self.capacity,
            target,
            self.num_layers,
            dims.ndim(),
            axis
        );

        self.k_caches = (0..self.num_layers)
            .map(|_| Array::zeros(dims.clone()))
            .collect();
        self.v_caches = (0..self.num_layers)
            .map(|_| Array::zeros(dims.clone()))
            .collect();
        self.k_ort.clear();
        self.v_ort.clear();
        self.capacity = target;
        Ok(())
    }

    /// Grow capacity geometrically while preserving valid prefix rows.
    pub fn grow_to(&mut self, needed: usize, valid_len: usize) -> Result<(), GSVError> {
        if needed <= self.capacity {
            return Ok(());
        }
        let mut target = (self.capacity.max(256) * 2).max(needed);
        while target < needed {
            target = target.saturating_mul(2);
        }

        info!(
            "Growing KV workspace from {} to {} (valid_len={})",
            self.capacity, target, valid_len
        );

        for i in 0..self.num_layers {
            let old_k = &self.k_caches[i];
            let old_v = &self.v_caches[i];
            let axis = seq_axis(old_k.ndim());
            let mut new_k_dims = old_k.raw_dim().clone();
            new_k_dims[axis] = target;
            let mut new_v_dims = old_v.raw_dim().clone();
            new_v_dims[axis] = target;

            let mut new_k = Array::zeros(new_k_dims);
            let mut new_v = Array::zeros(new_v_dims);
            let copy_len = valid_len.min(old_k.shape()[axis]);
            if axis == 2 {
                new_k
                    .slice_mut(s![.., .., 0..copy_len, ..])
                    .assign(&old_k.slice(s![.., .., 0..copy_len, ..]));
                new_v
                    .slice_mut(s![.., .., 0..copy_len, ..])
                    .assign(&old_v.slice(s![.., .., 0..copy_len, ..]));
            } else {
                new_k
                    .slice_mut(s![.., 0..copy_len, ..])
                    .assign(&old_k.slice(s![.., 0..copy_len, ..]));
                new_v
                    .slice_mut(s![.., 0..copy_len, ..])
                    .assign(&old_v.slice(s![.., 0..copy_len, ..]));
            }
            self.k_caches[i] = new_k;
            self.v_caches[i] = new_v;
        }
        self.capacity = target;
        Ok(())
    }

    /// Copy first-stage decoder KV into the workspace (prefix only).
    #[allow(dead_code)]
    pub fn load_initial(
        &mut self,
        k_inits: &[ArrayView<'_, KvDType, IxDyn>],
        v_inits: &[ArrayView<'_, KvDType, IxDyn>],
    ) -> Result<(), GSVError> {
        if k_inits.len() != self.num_layers || v_inits.len() != self.num_layers {
            return Err(GSVError::from("KV init layer count mismatch"));
        }
        let axis = seq_axis(k_inits[0].ndim());
        let initial_seq_len = k_inits[0].shape()[axis];
        self.ensure_capacity(Self::suggested_capacity(initial_seq_len), &k_inits[0])?;

        for i in 0..self.num_layers {
            self.write_prefix_layer(i, &k_inits[i], &v_inits[i])?;
        }
        Ok(())
    }

    /// Write one layer's prefix from an ORT output view (single copy into workspace).
    pub fn write_prefix_layer(
        &mut self,
        layer: usize,
        k: &ArrayView<'_, KvDType, IxDyn>,
        v: &ArrayView<'_, KvDType, IxDyn>,
    ) -> Result<(), GSVError> {
        if layer >= self.num_layers {
            return Err(GSVError::from("KV layer out of range"));
        }
        let axis = seq_axis(k.ndim());
        let seq = k.shape()[axis];
        if self.capacity < seq {
            self.ensure_capacity(Self::suggested_capacity(seq), k)?;
        }
        if axis == 2 {
            self.k_caches[layer]
                .slice_mut(s![.., .., 0..seq, ..])
                .assign(k);
            self.v_caches[layer]
                .slice_mut(s![.., .., 0..seq, ..])
                .assign(v);
        } else {
            self.k_caches[layer]
                .slice_mut(s![.., 0..seq, ..])
                .assign(k);
            self.v_caches[layer]
                .slice_mut(s![.., 0..seq, ..])
                .assign(v);
        }
        Ok(())
    }

    /// Materialize contiguous `[B,H,valid,D]` ORT feeds for head-major caches.
    ///
    /// No-op for legacy rank-3 caches (time-prefix views are already contiguous when B=1).
    pub fn compact_for_ort(&mut self, valid_len: usize) {
        if !self.is_head_major() {
            return;
        }
        if self.k_ort.len() != self.num_layers {
            self.k_ort = (0..self.num_layers)
                .map(|_| Array::zeros(IxDyn(&[0])))
                .collect();
            self.v_ort = (0..self.num_layers)
                .map(|_| Array::zeros(IxDyn(&[0])))
                .collect();
        }
        for i in 0..self.num_layers {
            let k_src = self.k_caches[i].slice(s![.., .., 0..valid_len, ..]);
            let v_src = self.v_caches[i].slice(s![.., .., 0..valid_len, ..]);
            self.k_ort[i] = k_src.to_owned().into_dyn();
            self.v_ort[i] = v_src.to_owned().into_dyn();
            debug_assert!(
                self.k_ort[i].is_standard_layout(),
                "head-major ORT feed must be contiguous"
            );
        }
    }

    pub fn k_view(
        &self,
        layer: usize,
        valid_len: usize,
    ) -> ArrayView<'_, KvDType, IxDyn> {
        if self.is_head_major() {
            debug_assert_eq!(
                self.k_ort.get(layer).map(|a| a.shape().get(2).copied()),
                Some(Some(valid_len)),
                "compact_for_ort({}) must be called before k_view on head-major caches",
                valid_len
            );
            self.k_ort[layer].view()
        } else {
            self.k_caches[layer]
                .slice(s![.., 0..valid_len, ..])
                .into_dyn()
        }
    }

    pub fn v_view(
        &self,
        layer: usize,
        valid_len: usize,
    ) -> ArrayView<'_, KvDType, IxDyn> {
        if self.is_head_major() {
            debug_assert_eq!(
                self.v_ort.get(layer).map(|a| a.shape().get(2).copied()),
                Some(Some(valid_len)),
                "compact_for_ort({}) must be called before v_view on head-major caches",
                valid_len
            );
            self.v_ort[layer].view()
        } else {
            self.v_caches[layer]
                .slice(s![.., 0..valid_len, ..])
                .into_dyn()
        }
    }

    pub fn write_slice_from_inc(
        &mut self,
        layer: usize,
        valid_len: usize,
        inc_k: &ArrayView<'_, KvDType, IxDyn>,
        inc_v: &ArrayView<'_, KvDType, IxDyn>,
    ) {
        let axis = seq_axis(self.k_caches[layer].ndim());
        let inc_axis = seq_axis(inc_k.ndim());
        let delta = inc_k.shape().get(inc_axis).copied() == Some(1);

        if axis == 2 {
            let k_new = if delta {
                inc_k.slice(s![.., .., 0, ..])
            } else {
                inc_k.slice(s![.., .., valid_len, ..])
            };
            let v_new = if delta {
                inc_v.slice(s![.., .., 0, ..])
            } else {
                inc_v.slice(s![.., .., valid_len, ..])
            };
            self.k_caches[layer]
                .slice_mut(s![.., .., valid_len, ..])
                .assign(&k_new);
            self.v_caches[layer]
                .slice_mut(s![.., .., valid_len, ..])
                .assign(&v_new);
        } else {
            let k_new = if delta {
                inc_k.slice(s![.., 0, ..])
            } else {
                inc_k.slice(s![.., valid_len, ..])
            };
            let v_new = if delta {
                inc_v.slice(s![.., 0, ..])
            } else {
                inc_v.slice(s![.., valid_len, ..])
            };
            self.k_caches[layer]
                .slice_mut(s![.., valid_len, ..])
                .assign(&k_new);
            self.v_caches[layer]
                .slice_mut(s![.., valid_len, ..])
                .assign(&v_new);
        }
    }
}

fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Array4};

    #[test]
    fn suggested_capacity_is_pow2_and_covers_prefix() {
        let c = KvWorkspace::suggested_capacity(129);
        assert!(c >= 129 + MIN_HEADROOM || c >= 256);
        assert_eq!(c.count_ones(), 1);
    }

    #[test]
    fn grow_preserves_prefix() {
        let mut ws = KvWorkspace::new(1);
        let init = Array3::<f32>::ones((1, 10, 4)).into_dyn();
        let view = init.view();
        ws.ensure_capacity(16, &view).unwrap();
        ws.k_caches[0]
            .slice_mut(s![.., 0..10, ..])
            .fill(3.0);
        ws.grow_to(40, 10).unwrap();
        assert!(ws.capacity() >= 40);
        assert!(ws.k_caches[0].slice(s![0, 0..10, 0]).iter().all(|&x| x == 3.0));
    }

    #[test]
    fn head_major_prefix_and_delta() {
        let mut ws = KvWorkspace::new(1);
        let init = Array4::<f32>::ones((1, 16, 8, 32)).into_dyn();
        ws.write_prefix_layer(0, &init.view(), &init.view()).unwrap();
        assert_eq!(ws.capacity() >= 8, true);
        let delta = Array4::<f32>::from_elem((1, 16, 1, 32), 7.0).into_dyn();
        ws.write_slice_from_inc(0, 8, &delta.view(), &delta.view());
        assert!(ws.k_caches[0].slice(s![0, 0, 8, 0]).iter().all(|&x| x == 7.0));
    }

    #[test]
    fn head_major_ort_feed_is_contiguous() {
        let mut ws = KvWorkspace::new(1);
        let init = Array4::<f32>::ones((1, 16, 8, 32)).into_dyn();
        ws.write_prefix_layer(0, &init.view(), &init.view()).unwrap();
        ws.compact_for_ort(8);
        let view = ws.k_view(0, 8);
        assert!(view.is_standard_layout());
        assert_eq!(view.shape(), &[1, 16, 8, 32]);
        assert!(view.as_slice().is_some());
    }
}
