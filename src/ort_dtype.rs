//! Detect ORT tensor element types and convert between f32 / f16 for native-FP16 graphs.

use half::f16;
use ndarray::{Array, ArrayBase, Data, Dimension};
use ort::{
    session::Session,
    value::{DynValue, TensorElementType, ValueType},
};

use crate::error::GSVError;

/// True when `session` input `name` is `tensor(float16)`.
pub fn session_input_is_f16(session: &Session, name: &str) -> bool {
    session.inputs().iter().any(|input| {
        input.name() == name
            && matches!(
                input.dtype(),
                ValueType::Tensor {
                    ty: TensorElementType::Float16,
                    ..
                }
            )
    })
}

/// True when `session` output `name` is `tensor(float16)`.
pub fn session_output_is_f16(session: &Session, name: &str) -> bool {
    session.outputs().iter().any(|output| {
        output.name() == name
            && matches!(
                output.dtype(),
                ValueType::Tensor {
                    ty: TensorElementType::Float16,
                    ..
                }
            )
    })
}

/// Map an f32 ndarray to f16 (owned).
pub fn array_to_f16<S, D>(arr: &ArrayBase<S, D>) -> Array<f16, D>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    arr.mapv(f16::from_f32)
}

/// Extract a floating tensor as `f32`, accepting either FP32 or FP16 storage.
pub fn extract_array_f32(value: &DynValue) -> Result<ndarray::ArrayD<f32>, GSVError> {
    if let Ok(arr) = value.try_extract_array::<f32>() {
        return Ok(arr.into_owned());
    }
    let arr = value
        .try_extract_array::<f16>()
        .map_err(|e| GSVError::from(e.to_string()))?;
    Ok(arr.mapv(|x| x.to_f32()))
}
