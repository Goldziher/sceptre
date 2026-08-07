//! The [`ModelBackend`] implementation for the hand-written candle networks.

use candle_core::{Device, Tensor as CandleTensor};

use super::craft_net::CraftNet;
use super::crnn_net::CrnnNet;
use super::onnx_proto::OnnxGraph;
use super::{candle_error, weights};
use crate::error::{OcrError, Result};
use crate::inference::{BackendOptions, ModelBackend, NetworkKind, Tensor, buffer};

/// `Conv` nodes in the exported CRAFT detector.
const DETECTOR_CONVOLUTIONS: usize = 27;

/// `Conv` nodes in an exported gen2 recognizer.
const RECOGNIZER_CONVOLUTIONS: usize = 7;

/// Bidirectional `LSTM` nodes in an exported gen2 recognizer.
const RECOGNIZER_RECURRENT_LAYERS: usize = 2;

/// The network a set of model bytes was resolved into.
enum Net {
    // Boxed because CRAFT's 27 convolutions make it far larger than the recognizer. ~keep
    Craft(Box<CraftNet>),
    Crnn(Box<CrnnNet>),
}

/// A hand-written network with its trained weights, bound to a compute device.
///
/// The weights are read once at load time and the network holds them, so [`Self::run`]
/// needs no interior mutability and the backend is `Send + Sync` without a lock.
pub(crate) struct CandleBackend {
    net: Net,
    device: Device,
}

impl CandleBackend {
    /// Decode the model, check it matches the requested role, and build the network.
    pub(crate) fn load(model_bytes: &[u8], options: BackendOptions<'_>) -> Result<Self> {
        // The thread budget is not plumbed through: candle sizes its own rayon pool from ~keep
        // the environment at first use, and every inference already runs inside the reader's ~keep
        // private pool, so the shared budget is honored by containment rather than by a knob. ~keep
        let _ = options.threads;
        let device = Device::Cpu;
        let graph = OnnxGraph::decode(model_bytes)?;
        validate_network(&graph, options.network)?;
        let vb = weights::var_builder(&graph, &device)?;

        let net = match options.network {
            NetworkKind::Detector => Net::Craft(Box::new(
                CraftNet::new(vb).map_err(|error| candle_error("build the CRAFT detector", error))?,
            )),
            NetworkKind::Recognizer => Net::Crnn(Box::new(
                CrnnNet::new(vb).map_err(|error| candle_error("build the CRNN recognizer", error))?,
            )),
        };
        Ok(Self { net, device })
    }
}

impl ModelBackend for CandleBackend {
    fn name(&self) -> &str {
        "candle"
    }

    fn run(&self, input: Tensor) -> Result<Tensor> {
        let (shape, data) = buffer::input_buffer(input);
        let tensor = CandleTensor::from_vec(data, shape.as_slice(), &self.device)
            .map_err(|error| candle_error("build the candle input tensor", error))?;

        let output = match &self.net {
            Net::Craft(net) => net.forward(&tensor),
            Net::Crnn(net) => net.forward(&tensor),
        }
        .map_err(|error| candle_error("run candle inference", error))?;

        let dims = output.dims().to_vec();
        // CRAFT ends on a permuted view and `flatten_all` reshapes, which needs a ~keep
        // contiguous buffer; without this the heat-maps would be read in the wrong order. ~keep
        let values = output
            .contiguous()
            .and_then(|output| output.flatten_all())
            .and_then(|output| output.to_vec1::<f32>())
            .map_err(|error| candle_error("read the candle output tensor as f32", error))?;
        buffer::array_from_parts("candle", &dims, values)
    }
}

/// Reject bytes whose structure does not match the network the caller asked for.
///
/// The requested kind selects which network to build; this checks the graph agrees. Weights are
/// resolved positionally, so without it an unrelated model — which [`ModelProvider`]
/// callers may supply (ADR 0028) — would load whatever tensors happened to sit at those
/// positions and produce confident nonsense instead of an error.
///
/// [`ModelProvider`]: crate::ModelProvider
fn validate_network(graph: &OnnxGraph, network: NetworkKind) -> Result<()> {
    let (expected_convolutions, expected_recurrent) = match network {
        NetworkKind::Detector => (DETECTOR_CONVOLUTIONS, 0),
        NetworkKind::Recognizer => (RECOGNIZER_CONVOLUTIONS, RECOGNIZER_RECURRENT_LAYERS),
    };
    let convolutions = graph.op_count("Conv");
    let recurrent = graph.op_count("LSTM");
    if convolutions == expected_convolutions && recurrent == expected_recurrent {
        return Ok(());
    }
    Err(OcrError::inference(format!(
        "the model does not look like the {network:?} network the candle backend was asked for: \
         expected {expected_convolutions} Conv and {expected_recurrent} LSTM nodes, \
         found {convolutions} Conv and {recurrent} LSTM"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_reject_bytes_that_do_not_decode() {
        let error = CandleBackend::load(&[0xff, 0xff], BackendOptions::default())
            .err()
            .expect("garbage must not load");
        assert!(matches!(error, OcrError::Inference { .. }));
    }

    /// A graph that does not match the requested network must be refused, not loaded positionally.
    #[test]
    fn should_reject_a_graph_that_does_not_match_the_requested_network() {
        let graph = OnnxGraph {
            nodes: Vec::new(),
            initializers: std::collections::HashMap::new(),
        };

        let error = validate_network(&graph, NetworkKind::Recognizer).expect_err("an empty graph is not a recognizer");

        let message = format!("{error}");
        assert!(
            message.contains("Recognizer") && message.contains("LSTM"),
            "the error must name the expected role and what it looked for: {message}"
        );
    }
}

/// Whole-model agreement with the `ort` backend on the real exported models.
///
/// The hand-written forward pass can diverge from the graph anywhere — a mistransposed
/// weight, a misordered LSTM gate, a skip connection taken one convolution too late — and
/// every one of those produces plausible output. Comparing whole-model output on fixed
/// input localizes such a defect to a single model and shape, which reading recognized text
/// cannot: by the time words differ, the cause is many layers back.
///
/// Bit-exactness is unachievable, because candle accumulates its bilinear interpolation
/// weights in `f64` where ONNX Runtime uses `f32`, so the bar is a small absolute tolerance.
/// The shapes are the ones `sceptre_rs_tools.export` already validates torch against
/// onnxruntime, so they are known to cover the dynamic axes and the batched path.
#[cfg(all(test, feature = "ort"))]
mod ort_parity {
    use ndarray::{ArrayD, IxDyn};

    use super::*;
    use crate::config::{Backend, Language, OcrConfig};
    use crate::inference::load_backend;
    use crate::models::provision::model_manifest;

    /// Largest absolute difference tolerated between the two backends' outputs.
    const TOLERANCE: f32 = 1e-4;

    /// Multiplier and increment of a full-period linear congruential generator.
    const LCG: (u32, u32) = (1_664_525, 1_013_904_223);

    fn require_models() -> bool {
        std::env::var("SCEPTRE_REQUIRE_MODELS")
            .map(|value| !matches!(value.trim(), "" | "0" | "false" | "no"))
            .unwrap_or(false)
    }

    /// Deterministic pseudo-random values in `[-1, 1)`, so any failure reproduces exactly.
    fn pseudo_random(shape: &[usize], seed: u32) -> Vec<f32> {
        let count: usize = shape.iter().product();
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(LCG.0).wrapping_add(LCG.1);
                (state >> 8) as f32 / (1_u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn cached_model(language: Language, name: &str) -> Option<std::path::PathBuf> {
        let mut config = OcrConfig::default();
        config.model.languages = vec![language];
        let manifest = model_manifest(&config).expect("build the model manifest");
        let entry = manifest.into_iter().find(|info| info.name == name)?;
        entry.cached.then_some(entry.path?)
    }

    fn run(
        backend: Backend,
        bytes: &[u8],
        network: NetworkKind,
        shape: &[usize],
        values: &[f32],
    ) -> (Vec<usize>, Vec<f32>) {
        let options = BackendOptions {
            network,
            ..BackendOptions::default()
        };
        let loaded = load_backend(backend, bytes, options).expect("load the model");
        let input = ArrayD::from_shape_vec(IxDyn(shape), values.to_vec()).expect("build the input");
        let output = loaded.run(input).expect("run inference");
        (output.shape().to_vec(), output.iter().copied().collect())
    }

    fn assert_agreement(model: &std::path::Path, network: NetworkKind, shape: &[usize], seed: u32) {
        let bytes = std::fs::read(model).expect("read the model file");
        let values = pseudo_random(shape, seed);
        let (ort_shape, ort_values) = run(Backend::Ort, &bytes, network, shape, &values);
        let (candle_shape, candle_values) = run(Backend::Candle, &bytes, network, shape, &values);

        assert_eq!(
            ort_shape, candle_shape,
            "{network:?} output shapes differ for input {shape:?}"
        );
        let differences: Vec<f32> = ort_values
            .iter()
            .zip(candle_values.iter())
            .map(|(left, right)| (left - right).abs())
            .collect();
        let (index, difference) = differences
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("the output is non-empty");
        let exceeding = differences.iter().filter(|value| **value > TOLERANCE).count();
        let coordinates: Vec<String> = differences
            .iter()
            .enumerate()
            .filter(|(_, value)| **value > TOLERANCE)
            .take(8)
            .map(|(flat, _)| {
                let mut remainder = flat;
                let mut position = Vec::new();
                for extent in ort_shape.iter().rev() {
                    position.push(remainder % extent);
                    remainder /= extent;
                }
                position.reverse();
                format!("{position:?}")
            })
            .collect();
        assert!(
            difference <= TOLERANCE,
            "{network:?} at input {shape:?}: candle differs from ort by {difference} at flat index \
             {index} of {} (ort={}, candle={}); {exceeding} values ({:.1}%) exceed the {TOLERANCE} \
             tolerance, output shape {ort_shape:?}; first differing positions {coordinates:?}",
            differences.len(),
            ort_values[index],
            candle_values[index],
            100.0 * exceeding as f64 / differences.len() as f64
        );
    }

    #[test]
    fn should_agree_with_ort_on_the_craft_detector() {
        let Some(model) = cached_model(Language::English, "craft_mlt_25k") else {
            assert!(!require_models(), "craft_mlt_25k is not cached");
            return;
        };
        for (index, shape) in [[1, 3, 256, 256], [1, 3, 320, 480], [1, 3, 512, 288]]
            .iter()
            .enumerate()
        {
            assert_agreement(&model, NetworkKind::Detector, shape, 1 + index as u32);
        }
    }

    #[test]
    fn should_agree_with_ort_on_the_english_recognizer() {
        let Some(model) = cached_model(Language::English, "english_g2") else {
            assert!(!require_models(), "english_g2 is not cached");
            return;
        };
        for (index, shape) in [[1, 1, 64, 128], [2, 1, 64, 256], [3, 1, 64, 96]].iter().enumerate() {
            assert_agreement(&model, NetworkKind::Recognizer, shape, 11 + index as u32);
        }
    }

    #[test]
    fn should_agree_with_ort_on_the_cyrillic_recognizer() {
        let Some(model) = cached_model(Language::Cyrillic, "cyrillic_g2") else {
            assert!(!require_models(), "cyrillic_g2 is not cached");
            return;
        };
        assert_agreement(&model, NetworkKind::Recognizer, &[2, 1, 64, 192], 21);
    }
}
