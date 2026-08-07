//! The gen2 CRNN recognizer, written out against `candle_nn`.
//!
//! Structure recovered from the exported graphs, which are identical across all eight
//! language models apart from the width of the final projection: a seven-layer VGG-style
//! feature extractor, a mean over the residual height axis, two bidirectional LSTMs each
//! followed by a projection, and a final projection to the CTC classes.
//!
//! The graph's `Shape`/`Gather`/`Reshape`/`ConstantOfShape` nodes have no counterpart here;
//! they exist only to compute dynamic sizes and zero initial states, which a hand-written
//! forward pass takes from the tensor itself.

use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, Module, VarBuilder, conv2d, linear};

use super::bilstm::BiLstm;
use super::ops::max_pool2d_padded;

/// Channel counts of the seven feature-extractor convolutions, input side then output.
const FEATURE_CHANNELS: [(usize, usize); 7] = [
    (1, 32),
    (32, 64),
    (64, 128),
    (128, 128),
    (128, 256),
    (256, 256),
    (256, 256),
];

/// Hidden units per direction in each bidirectional LSTM.
const HIDDEN_SIZE: usize = 256;

/// Feature width entering the recurrent stack.
const FEATURE_SIZE: usize = 256;

/// The gen2 CRNN recognizer.
pub(super) struct CrnnNet {
    features: Vec<Conv2d>,
    recurrent: [(BiLstm, Linear); 2],
    prediction: Linear,
}

impl CrnnNet {
    /// Build the network, reading its weights from `vb`.
    pub(super) fn new(vb: VarBuilder) -> CandleResult<Self> {
        let padded = Conv2dConfig {
            padding: 1,
            ..Conv2dConfig::default()
        };
        let mut features = Vec::with_capacity(FEATURE_CHANNELS.len());
        for (index, (in_channels, out_channels)) in FEATURE_CHANNELS.iter().enumerate() {
            // The last convolution is 2x2 and unpadded; the rest are padded 3x3. ~keep
            let (kernel, config) = if index == FEATURE_CHANNELS.len() - 1 {
                (2, Conv2dConfig::default())
            } else {
                (3, padded)
            };
            features.push(conv2d(
                *in_channels,
                *out_channels,
                kernel,
                config,
                vb.pp(format!("conv.{index}")),
            )?);
        }

        let recurrent = [
            (
                BiLstm::new(FEATURE_SIZE, HIDDEN_SIZE, vb.pp("rnn.0"))?,
                linear(2 * HIDDEN_SIZE, FEATURE_SIZE, vb.pp("linear.0"))?,
            ),
            (
                BiLstm::new(FEATURE_SIZE, HIDDEN_SIZE, vb.pp("rnn.1"))?,
                linear(2 * HIDDEN_SIZE, FEATURE_SIZE, vb.pp("linear.1"))?,
            ),
        ];
        let prediction = linear_from_shape(&vb, "linear.2")?;

        Ok(Self {
            features,
            recurrent,
            prediction,
        })
    }

    /// Map `[batch, 1, 64, width]` to raw CTC logits `[batch, steps, classes]`.
    pub(super) fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let sequence = self.extract_features(input)?;
        let mut hidden = sequence;
        for (recurrent, projection) in &self.recurrent {
            hidden = projection.forward(&recurrent.forward(&hidden)?)?;
        }
        self.prediction.forward(&hidden)
    }

    /// Run the convolutional stack and collapse it into a `[batch, steps, features]` sequence.
    fn extract_features(&self, input: &Tensor) -> CandleResult<Tensor> {
        let pool = |tensor: &Tensor, window: (usize, usize)| max_pool2d_padded(tensor, window, window, (0, 0));

        let mut hidden = self.features[0].forward(input)?.relu()?;
        hidden = pool(&hidden, (2, 2))?;
        hidden = self.features[1].forward(&hidden)?.relu()?;
        hidden = pool(&hidden, (2, 2))?;
        hidden = self.features[2].forward(&hidden)?.relu()?;
        hidden = self.features[3].forward(&hidden)?.relu()?;
        hidden = pool(&hidden, (2, 1))?;
        hidden = self.features[4].forward(&hidden)?.relu()?;
        hidden = self.features[5].forward(&hidden)?.relu()?;
        hidden = pool(&hidden, (2, 1))?;
        hidden = self.features[6].forward(&hidden)?.relu()?;

        // [batch, channels, height, width] -> [batch, width, channels, height], then average ~keep
        // away the residual height so each output column becomes one sequence step. ~keep
        hidden
            .permute((0, 3, 1, 2))?
            .contiguous()?
            .mean_keepdim(3)?
            .squeeze(3)?
            .contiguous()
    }
}

/// Build the final projection, taking its class count from the stored weight.
///
/// The number of CTC classes is a property of the language model rather than of the
/// architecture, so it is read from the weight instead of being passed in and re-checked.
fn linear_from_shape(vb: &VarBuilder, path: &str) -> CandleResult<Linear> {
    let weight = vb.pp(path).get_unchecked("weight")?;
    let classes = weight.dim(0)?;
    linear(FEATURE_SIZE, classes, vb.pp(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_describe_seven_feature_convolutions() {
        assert_eq!(
            FEATURE_CHANNELS.len(),
            7,
            "the exported recognizers carry seven Conv nodes"
        );
        for window in FEATURE_CHANNELS.windows(2) {
            assert_eq!(
                window[0].1, window[1].0,
                "each convolution must consume the previous one's channels"
            );
        }
        assert_eq!(
            FEATURE_CHANNELS[0].0, 1,
            "the recognizer reads a single grayscale plane"
        );
        assert_eq!(
            FEATURE_CHANNELS[FEATURE_CHANNELS.len() - 1].1,
            FEATURE_SIZE,
            "the feature width entering the recurrent stack must match the last convolution"
        );
    }
}
