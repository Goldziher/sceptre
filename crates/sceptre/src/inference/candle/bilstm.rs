//! A bidirectional LSTM layer composed from candle's two per-direction cells.

use candle_core::{D, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_nn::rnn::{Direction, LSTM, LSTMConfig, RNN, lstm};

use super::ops::reverse_time;

/// A bidirectional LSTM: a forward pass and a backward pass concatenated per timestep.
///
/// `candle_nn` has no bidirectional layer. Its [`Direction`] only chooses which weight
/// names a cell reads — `RNN::seq` iterates forwards either way — so the reverse direction
/// is produced here by running the reversed sequence and reversing the result back. Losing
/// that step still yields plausible output, which is why [`Self::forward`] is covered by a
/// test that fails if the halves are identical.
pub(super) struct BiLstm {
    forward: LSTM,
    backward: LSTM,
}

impl BiLstm {
    /// Load both directions from `vb`, which supplies the `_reverse` weights for the second.
    pub(super) fn new(in_dim: usize, hidden: usize, vb: VarBuilder) -> CandleResult<Self> {
        let forward = lstm(
            in_dim,
            hidden,
            LSTMConfig {
                direction: Direction::Forward,
                ..LSTMConfig::default()
            },
            vb.clone(),
        )?;
        let backward = lstm(
            in_dim,
            hidden,
            LSTMConfig {
                direction: Direction::Backward,
                ..LSTMConfig::default()
            },
            vb,
        )?;
        Ok(Self { forward, backward })
    }

    /// Run `[batch, steps, in_dim]` and return `[batch, steps, 2 * hidden]`.
    pub(super) fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let forward = self.forward.states_to_tensor(&self.forward.seq(input)?)?;
        let reversed = reverse_time(input)?;
        let backward = self.backward.states_to_tensor(&self.backward.seq(&reversed)?)?;
        let backward = reverse_time(&backward)?;
        Tensor::cat(&[forward, backward], D::Minus1)?.contiguous()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device};

    use super::*;

    /// Build a one-unit BiLSTM whose two directions carry **identical** weights.
    ///
    /// Identical weights are what makes the reversal testable: the only thing that can then
    /// make the two halves differ is the order in which each reads the sequence.
    fn twin_bilstm() -> BiLstm {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        for (suffix, scale) in [("", 0.5_f32), ("_reverse", 0.5_f32)] {
            tensors.insert(
                format!("weight_ih_l0{suffix}"),
                Tensor::from_vec(vec![scale; 4], (4, 1), &device).unwrap(),
            );
            tensors.insert(
                format!("weight_hh_l0{suffix}"),
                Tensor::from_vec(vec![scale; 4], (4, 1), &device).unwrap(),
            );
            tensors.insert(
                format!("bias_ih_l0{suffix}"),
                Tensor::zeros(4, DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("bias_hh_l0{suffix}"),
                Tensor::zeros(4, DType::F32, &device).unwrap(),
            );
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        BiLstm::new(1, 1, vb).expect("build the bilstm")
    }

    fn half(output: &Tensor, index: usize) -> Vec<f32> {
        output
            .narrow(D::Minus1, index, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    fn should_concatenate_both_directions_along_the_feature_axis() {
        let input = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], (1, 3, 1), &Device::Cpu).unwrap();

        let output = twin_bilstm().forward(&input).expect("forward");

        assert_eq!(output.dims(), &[1, 3, 2]);
    }

    /// The regression for the `Direction::Backward` trap.
    ///
    /// Both directions hold the same weights, so if the backward cell were run over the
    /// sequence in its original order — which is exactly what dropping [`reverse_time`]
    /// would do — the two halves would be equal element for element. They must not be.
    #[test]
    fn should_run_the_backward_direction_over_the_reversed_sequence() {
        let ascending = Tensor::from_vec(vec![1.0_f32, 2.0, 9.0], (1, 3, 1), &Device::Cpu).unwrap();

        let output = twin_bilstm().forward(&ascending).expect("forward");

        assert_ne!(
            half(&output, 0),
            half(&output, 1),
            "identical weights leave the reading order as the only difference between halves"
        );
    }

    /// On a palindrome the backward pass reads the same values in the same order as the
    /// forward pass, so its output is the forward output reversed — timestep for timestep.
    ///
    /// This pins the difference between the halves to the reading order specifically: a bug
    /// that perturbed the backward direction in any other way would break this too.
    #[test]
    fn should_mirror_the_forward_pass_on_a_palindromic_sequence() {
        let palindrome = Tensor::from_vec(vec![1.0_f32, 7.0, 1.0], (1, 3, 1), &Device::Cpu).unwrap();

        let output = twin_bilstm().forward(&palindrome).expect("forward");

        let forward: Vec<f32> = half(&output, 0);
        let mut mirrored = forward.clone();
        mirrored.reverse();
        assert_eq!(half(&output, 1), mirrored);
    }
}
