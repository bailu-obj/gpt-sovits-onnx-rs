"""Built-in sampling defaults (aligned with Rust `InferParams::default()`)."""

DEFAULT_SAMPLING = {
    "top_k": 4,
    "top_p": 0.9,
    "temperature": 1.0,
    "repetition_penalty": 1.35,
}

# Fixed seed for Rust/Python parity scripts only.
COMPARE_SEED = 42
