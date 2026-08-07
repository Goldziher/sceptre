//! The CRAFT text detector, written out against `candle_nn`.
//!
//! A VGG16 backbone (its batch normalization folded into the convolutions by the export)
//! feeding a U-Net decoder with three skip connections, then a small classification head.
//! The graph's bilinear `Resize` nodes take their target extent from the skip tensor they
//! are about to be concatenated with, which a hand-written pass reads directly from the
//! tensor — so unlike `tract`, this backend needs no fixed square canvas (ADR 0027).

use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d};

use super::ops::max_pool2d_padded;

/// Backbone convolutions: `(in, out, kernel, padding, dilation, rectified)`.
///
/// The last three carry no rectifier, matching CRAFT's fifth VGG stage.
const BACKBONE: [(usize, usize, usize, usize, usize, bool); 14] = [
    (3, 64, 3, 1, 1, true),
    (64, 64, 3, 1, 1, true),
    (64, 128, 3, 1, 1, true),
    (128, 128, 3, 1, 1, true),
    (128, 256, 3, 1, 1, true),
    (256, 256, 3, 1, 1, true),
    (256, 256, 3, 1, 1, true),
    (256, 512, 3, 1, 1, true),
    (512, 512, 3, 1, 1, true),
    (512, 512, 3, 1, 1, true),
    (512, 512, 3, 1, 1, true),
    (512, 512, 3, 1, 1, false),
    (512, 1024, 3, 6, 6, false),
    (1024, 1024, 1, 0, 1, false),
];

/// Decoder stages: `(concatenated channels, intermediate channels, output channels)`.
const DECODER: [(usize, usize, usize); 4] = [(1536, 512, 256), (768, 256, 128), (384, 128, 64), (192, 64, 32)];

/// Head convolutions: `(in, out, kernel, rectified)`.
const HEAD: [(usize, usize, usize, bool); 5] = [
    (32, 32, 3, true),
    (32, 32, 3, true),
    (32, 16, 3, true),
    (16, 16, 1, true),
    (16, 2, 1, false),
];

/// Index of the first decoder convolution among the graph's `Conv` nodes.
const DECODER_CONV_OFFSET: usize = BACKBONE.len();

/// Index of the first head convolution among the graph's `Conv` nodes.
const HEAD_CONV_OFFSET: usize = DECODER_CONV_OFFSET + 2 * DECODER.len();

/// One decoder stage: a 1x1 mixing convolution then a 3x3 refinement, both rectified.
struct UpConv {
    mix: Conv2d,
    refine: Conv2d,
}

impl UpConv {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        self.refine.forward(&self.mix.forward(input)?.relu()?)?.relu()
    }
}

/// The CRAFT detector.
pub(super) struct CraftNet {
    backbone: Vec<Conv2d>,
    decoder: Vec<UpConv>,
    head: Vec<Conv2d>,
}

impl CraftNet {
    /// Build the network, reading its weights from `vb`.
    pub(super) fn new(vb: VarBuilder) -> CandleResult<Self> {
        let mut backbone = Vec::with_capacity(BACKBONE.len());
        for (index, (in_channels, out_channels, kernel, padding, dilation, _)) in BACKBONE.iter().enumerate() {
            backbone.push(conv2d(
                *in_channels,
                *out_channels,
                *kernel,
                Conv2dConfig {
                    padding: *padding,
                    dilation: *dilation,
                    ..Conv2dConfig::default()
                },
                vb.pp(format!("conv.{index}")),
            )?);
        }

        let mut decoder = Vec::with_capacity(DECODER.len());
        for (stage, (concatenated, intermediate, output)) in DECODER.iter().enumerate() {
            let index = DECODER_CONV_OFFSET + 2 * stage;
            decoder.push(UpConv {
                mix: conv2d(
                    *concatenated,
                    *intermediate,
                    1,
                    Conv2dConfig::default(),
                    vb.pp(format!("conv.{index}")),
                )?,
                refine: conv2d(
                    *intermediate,
                    *output,
                    3,
                    Conv2dConfig {
                        padding: 1,
                        ..Conv2dConfig::default()
                    },
                    vb.pp(format!("conv.{}", index + 1)),
                )?,
            });
        }

        let mut head = Vec::with_capacity(HEAD.len());
        for (offset, (in_channels, out_channels, kernel, _)) in HEAD.iter().enumerate() {
            head.push(conv2d(
                *in_channels,
                *out_channels,
                *kernel,
                Conv2dConfig {
                    padding: kernel / 2,
                    ..Conv2dConfig::default()
                },
                vb.pp(format!("conv.{}", HEAD_CONV_OFFSET + offset)),
            )?);
        }

        Ok(Self {
            backbone,
            decoder,
            head,
        })
    }

    /// Map `[1, 3, height, width]` to the region/link heat-maps `[1, height/2, width/2, 2]`.
    pub(super) fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let (features, skips) = self.run_backbone(input)?;
        let decoded = self.run_decoder(features, &skips)?;
        let scores = self.run_head(&decoded)?;
        // The pipeline reads the heat-maps channel-last, matching the exported graph's ~keep
        // trailing transpose. ~keep
        scores.permute((0, 2, 3, 1))?.contiguous()
    }

    /// Run the VGG backbone, returning its output and the three skip tensors.
    fn run_backbone(&self, input: &Tensor) -> CandleResult<(Tensor, Vec<Tensor>)> {
        let apply = |index: usize, tensor: &Tensor| -> CandleResult<Tensor> {
            let output = self.backbone[index].forward(tensor)?;
            if BACKBONE[index].5 { output.relu() } else { Ok(output) }
        };
        let halve = |tensor: &Tensor| max_pool2d_padded(tensor, (2, 2), (2, 2), (0, 0));

        // Each skip is taken from the stage's *second* convolution, before the third one ~keep
        // and before the pool. The third convolution preserves the channel count, so a skip ~keep
        // read one layer too late still type-checks — only the numbers would be wrong. ~keep
        let mut skips = Vec::with_capacity(3);
        let mut hidden = apply(1, &apply(0, input)?)?;
        hidden = halve(&hidden)?;
        hidden = apply(3, &apply(2, &hidden)?)?;
        skips.push(hidden.clone());
        hidden = halve(&hidden)?;
        hidden = apply(5, &apply(4, &hidden)?)?;
        skips.push(hidden.clone());
        hidden = apply(6, &hidden)?;
        hidden = halve(&hidden)?;
        hidden = apply(8, &apply(7, &hidden)?)?;
        skips.push(hidden.clone());
        hidden = apply(9, &hidden)?;
        hidden = halve(&hidden)?;
        hidden = apply(11, &apply(10, &hidden)?)?;

        // The fifth stage pools with a padded 3x3 window over an unrectified convolution, ~keep
        // so the border must not be filled with zeros. ~keep
        let stage_five = max_pool2d_padded(&hidden, (3, 3), (1, 1), (1, 1))?;
        let features = apply(13, &apply(12, &stage_five)?)?;
        Ok((Tensor::cat(&[features, hidden], 1)?, skips))
    }

    /// Run the four decoder stages, upsampling onto each skip tensor in turn.
    fn run_decoder(&self, features: Tensor, skips: &[Tensor]) -> CandleResult<Tensor> {
        let mut hidden = self.decoder[0].forward(&features)?;
        for (stage, skip) in self.decoder[1..].iter().zip(skips.iter().rev()) {
            let (_, _, height, width) = skip.dims4()?;
            // `align_corners = false` is ONNX's `half_pixel` coordinate transform, which ~keep
            // is what the exported Resize nodes request. ~keep
            let upsampled = hidden.upsample_bilinear2d(height, width, false)?;
            hidden = stage.forward(&Tensor::cat(&[&upsampled, skip], 1)?)?;
        }
        Ok(hidden)
    }

    /// Run the classification head down to the two score channels.
    fn run_head(&self, input: &Tensor) -> CandleResult<Tensor> {
        let mut hidden = input.clone();
        for (index, convolution) in self.head.iter().enumerate() {
            hidden = convolution.forward(&hidden)?;
            if HEAD[index].3 {
                hidden = hidden.relu()?;
            }
        }
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_chain_the_backbone_channels() {
        for window in BACKBONE.windows(2) {
            assert_eq!(
                window[0].1, window[1].0,
                "each backbone convolution must consume the previous one's channels"
            );
        }
        assert_eq!(BACKBONE[0].0, 3, "CRAFT reads an RGB image");
    }

    /// The concatenated width of each decoder stage must equal the upsampled tensor plus
    /// the skip it is joined with, or the 1x1 mixing convolution would reject its input.
    #[test]
    fn should_size_each_decoder_stage_to_its_skip_connection() {
        let backbone_tail = BACKBONE[BACKBONE.len() - 1].1;
        let unrectified_stage = BACKBONE[11].1;
        assert_eq!(
            DECODER[0].0,
            backbone_tail + unrectified_stage,
            "the first stage joins the backbone output with the fifth stage input"
        );

        // Skips come from convolutions 8, 5 and 3 — the second of each stage. ~keep
        let skips = [BACKBONE[8].1, BACKBONE[5].1, BACKBONE[3].1];
        let upsampled = [DECODER[0].2, DECODER[1].2, DECODER[2].2];
        for ((stage, skip), incoming) in DECODER[1..].iter().zip(skips).zip(upsampled) {
            assert_eq!(
                stage.0,
                incoming + skip,
                "a decoder stage consumes the previous stage's output joined with its skip"
            );
        }
    }

    #[test]
    fn should_place_the_head_after_every_other_convolution() {
        assert_eq!(HEAD_CONV_OFFSET, 22);
        assert_eq!(
            HEAD_CONV_OFFSET + HEAD.len(),
            27,
            "the exported detector carries 27 Conv nodes in total"
        );
        assert_eq!(HEAD[HEAD.len() - 1].1, 2, "the head emits the region and link maps");
    }
}
