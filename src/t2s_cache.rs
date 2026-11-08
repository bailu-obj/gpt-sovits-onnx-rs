use ort::{
    session::{IoBinding, Session, SessionOutputs},
    value::{DynValue, Tensor},
};

use crate::GSVError;

pub(crate) const CACHE_RESERVE: usize = 128;
pub(crate) type LayerCache = (DynValue, DynValue);

#[derive(Clone, Copy)]
pub(crate) struct CacheLayout {
    heads: usize,
    head_dim: usize,
}

impl CacheLayout {
    pub(crate) fn from_decoder(session: &Session) -> Result<Option<Self>, GSVError> {
        if !session
            .inputs()
            .iter()
            .any(|input| input.name() == "past_length")
        {
            return Ok(None);
        }
        let shape = session
            .inputs()
            .iter()
            .find(|input| input.name() == "ik_cache_0")
            .and_then(|input| input.dtype().tensor_shape())
            .ok_or_else(|| GSVError::from("missing shared T2S cache shape"))?;
        match &shape[..] {
            [1, heads, -1, head_dim] if *heads > 0 && *head_dim > 0 => Ok(Some(Self {
                heads: *heads as usize,
                head_dim: *head_dim as usize,
            })),
            _ => Err(GSVError::from("invalid shared T2S cache layout")),
        }
    }

    pub(crate) fn prefill_length(self, tensor: &DynValue) -> Result<usize, GSVError> {
        let (shape, _) = tensor.try_extract_tensor::<f32>()?;
        match &shape[..] {
            [1, length, width] if *length > 0 && *width == (self.heads * self.head_dim) as i64 => {
                Ok(*length as usize)
            }
            _ => Err(GSVError::from("invalid T2S prefill cache layout")),
        }
    }

    fn copy_to_capacity(
        self,
        tensor: &DynValue,
        filled: usize,
        capacity: usize,
    ) -> Result<Tensor<f32>, GSVError> {
        let (shape, source) = tensor.try_extract_tensor::<f32>()?;
        let width = self.heads * self.head_dim;
        let mut destination = vec![0.; capacity * width];
        match &shape[..] {
            [1, rows, channels] if *rows == filled as i64 && *channels == width as i64 => {
                for (token, row) in source.chunks_exact(width).enumerate() {
                    for (head, values) in row.chunks_exact(self.head_dim).enumerate() {
                        let offset = (head * capacity + token) * self.head_dim;
                        destination[offset..offset + self.head_dim].copy_from_slice(values);
                    }
                }
            }
            [1, heads, rows, dim]
                if *heads == self.heads as i64
                    && *rows == filled as i64
                    && *dim == self.head_dim as i64 =>
            {
                let valid = filled * self.head_dim;
                for (head, values) in source.chunks_exact(valid).enumerate() {
                    let offset = head * capacity * self.head_dim;
                    destination[offset..offset + valid].copy_from_slice(values);
                }
            }
            _ => return Err(GSVError::from("invalid T2S cache shape during binding")),
        }
        Ok(Tensor::from_array((
            [1, self.heads, capacity, self.head_dim],
            destination,
        ))?)
    }
}

pub(crate) struct SharedKvCache {
    binding: IoBinding,
    pub(crate) capacity: usize,
}

impl SharedKvCache {
    pub(crate) fn new(
        session: &Session,
        layout: CacheLayout,
        caches: &[LayerCache],
        filled: usize,
        capacity: usize,
        vocab_size: usize,
    ) -> Result<Self, GSVError> {
        let mut binding = session.create_binding()?;
        binding.bind_output(
            "logits",
            Tensor::from_array(([vocab_size], vec![0_f32; vocab_size]))?,
        )?;
        for (layer, (key, value)) in caches.iter().enumerate() {
            for (kind, tensor) in [("k", key), ("v", value)] {
                let cache = layout.copy_to_capacity(tensor, filled, capacity)?;
                let name = format!("{kind}_cache_{layer}");
                binding.bind_input(format!("i{name}"), &cache)?;
                binding.bind_output(name, cache)?;
            }
        }
        Ok(Self { binding, capacity })
    }

    pub(crate) fn run<'a>(
        &'a mut self,
        session: &'a mut Session,
        token: i64,
        position: i64,
        past: usize,
    ) -> Result<SessionOutputs<'a>, GSVError> {
        debug_assert!(past < self.capacity);
        self.binding
            .bind_input("iy", &Tensor::from_array(([1, 1], vec![token]))?)?;
        self.binding
            .bind_input("position", &Tensor::from_array(([1], vec![position]))?)?;
        self.binding.bind_input(
            "past_length",
            &Tensor::from_array((
                [1],
                vec![i32::try_from(past).map_err(|_| GSVError::from("T2S cache length overflow"))?],
            ))?,
        )?;
        Ok(session.run_binding(&self.binding)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_and_growth_preserve_head_order() -> Result<(), GSVError> {
        let layout = CacheLayout {
            heads: 2,
            head_dim: 2,
        };
        let prefill =
            Tensor::from_array(([1, 2, 4], vec![1_f32, 2., 3., 4., 5., 6., 7., 8.]))?.into_dyn();
        assert_eq!(layout.prefill_length(&prefill)?, 2);
        let head_major = layout.copy_to_capacity(&prefill, 2, 2)?.into_dyn();
        let grown = layout.copy_to_capacity(&head_major, 2, 3)?;
        let (shape, data) = grown.try_extract_tensor::<f32>()?;
        assert_eq!(&shape[..], &[1, 2, 3, 2]);
        assert_eq!(data, &[1., 2., 5., 6., 0., 0., 3., 4., 7., 8., 0., 0.]);
        Ok(())
    }
}
