use super::normalize::normalize_fragment;

/// Recover fragment order after batched inference (Python `recovery_order`).
pub fn recovery_order(
    batches: Vec<Vec<Vec<f32>>>,
    batch_index_list: &[Vec<usize>],
) -> Vec<Vec<f32>> {
    let length: usize = batch_index_list.iter().map(|list| list.len()).sum();
    let mut ordered = vec![None; length];

    for (i, index_list) in batch_index_list.iter().enumerate() {
        for (j, &index) in index_list.iter().enumerate() {
            ordered[index] = Some(batches[i][j].clone());
        }
    }

    ordered.into_iter().flatten().collect()
}

fn flatten_batches(batches: Vec<Vec<Vec<f32>>>) -> Vec<Vec<f32>> {
    batches.into_iter().flatten().collect()
}

fn silence_len(sample_rate: u32, fragment_interval: f32) -> usize {
    if fragment_interval > 0.0 {
        (sample_rate as f32 * fragment_interval) as usize
    } else {
        0
    }
}

/// Append trailing silence to each fragment after peak normalization.
pub fn process_fragments(
    mut batches: Vec<Vec<Vec<f32>>>,
    sample_rate: u32,
    fragment_interval: f32,
) -> Vec<Vec<Vec<f32>>> {
    let pad_len = silence_len(sample_rate, fragment_interval);

    for batch in &mut batches {
        for fragment in batch.iter_mut() {
            normalize_fragment(fragment);
            if pad_len > 0 {
                fragment.extend(std::iter::repeat_n(0.0f32, pad_len));
            }
        }
    }

    batches
}

/// Concatenate ordered fragments into a single buffer.
pub fn concat_fragments(fragments: Vec<Vec<f32>>) -> Vec<f32> {
    let total_len: usize = fragments.iter().map(|f| f.len()).sum();
    let mut out = Vec::with_capacity(total_len);
    for fragment in fragments {
        out.extend(fragment);
    }
    out
}

pub fn flatten_and_concat(
    batches: Vec<Vec<Vec<f32>>>,
    split_bucket: bool,
    batch_index_list: Option<&[Vec<usize>]>,
) -> Vec<f32> {
    let fragments = if split_bucket {
        if let Some(list) = batch_index_list {
            recovery_order(batches, list)
        } else {
            flatten_batches(batches)
        }
    } else {
        flatten_batches(batches)
    };

    concat_fragments(fragments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragment_interval_appends_silence() {
        let batches = vec![vec![vec![1.0; 1000]]];
        let processed = process_fragments(batches, 32000, 0.3);
        assert_eq!(processed[0][0].len(), 1000 + 9600);
        assert!(processed[0][0][1000..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn recovery_order_restores_sequence() {
        let batches = vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0]]];
        let index_list = vec![vec![0, 2], vec![1]];
        let ordered = recovery_order(batches, &index_list);
        assert_eq!(ordered.len(), 3);
        assert_eq!(ordered[0], vec![1.0]);
        assert_eq!(ordered[1], vec![3.0]);
        assert_eq!(ordered[2], vec![2.0]);
    }

    #[test]
    fn audio_postprocess_concat_no_bucket() {
        let batches = vec![vec![vec![1.0; 100], vec![2.0; 50]]];
        let processed = process_fragments(batches, 32000, 0.3);
        let out = flatten_and_concat(processed, false, None);
        assert_eq!(out.len(), 100 + 9600 + 50 + 9600);
    }
}
