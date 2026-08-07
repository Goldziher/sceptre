//! Minimal ONNX protobuf decoding, reduced to what a hand-written forward pass needs.
//!
//! The `candle` backend does not interpret the ONNX graph; it runs a hand-written network
//! and reads the trained weights out of the file's initializers. Only a handful of the
//! schema's messages are needed for that, so they are declared here by hand rather than
//! generated.
//!
//! Generating them would mean depending on `candle-onnx` or `prost-build`, both of which
//! invoke `protoc` at build time. That would make `cargo install` require a native
//! toolchain again — the defect ADR 0029 removed for the ONNX Runtime backend — for a
//! backend whose whole purpose is to need no native toolchain. `prost` itself is a pure
//! decoder with no build script, so the wire handling is still a maintained library and
//! only the ~40 lines of field numbers below are ours.
//!
//! Field numbers are fixed by `onnx.proto3` and are covered by tests that decode
//! hand-encoded bytes, so a wrong tag fails here rather than silently reading zeros.

use std::collections::HashMap;

use prost::Message;

use crate::error::{OcrError, Result};

/// `TensorProto.DataType.FLOAT`, the only element type these models carry.
const TENSOR_DATA_TYPE_FLOAT: i32 = 1;

/// Bytes per `f32` in a `TensorProto.raw_data` buffer.
const RAW_DATA_BYTES_PER_FLOAT: usize = 4;

/// `onnx.ModelProto`, reduced to its graph.
#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
}

/// `onnx.GraphProto`, reduced to its nodes and initializers.
#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    node: Vec<NodeProto>,
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
}

/// `onnx.NodeProto`, reduced to its operator and operand names.
#[derive(Clone, PartialEq, Message)]
struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, tag = "4")]
    op_type: String,
}

/// `onnx.TensorProto`, reduced to a named `f32` array.
#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    data_type: i32,
    #[prost(float, repeated, tag = "4")]
    float_data: Vec<f32>,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes = "vec", tag = "9")]
    raw_data: Vec<u8>,
}

/// One graph node, in the file's topological order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OnnxNode {
    /// The operator name, e.g. `Conv` or `LSTM`.
    pub op_type: String,
    /// Operand names, which index the initializer table for trained weights.
    pub inputs: Vec<String>,
    /// Result names, which later nodes reference as their operands.
    pub outputs: Vec<String>,
}

/// One trained tensor: its shape and its values in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct OnnxTensor {
    /// Dimensions, outermost first.
    pub dims: Vec<usize>,
    /// Values in row-major order, `dims.iter().product()` of them.
    pub data: Vec<f32>,
}

/// An ONNX graph reduced to the node sequence and the initializer table.
#[derive(Debug, Clone)]
pub(super) struct OnnxGraph {
    /// Nodes in the file's topological order.
    pub nodes: Vec<OnnxNode>,
    /// Trained tensors keyed by name.
    pub initializers: HashMap<String, OnnxTensor>,
}

impl OnnxGraph {
    /// Decode an ONNX model from its serialized bytes.
    ///
    /// Only the graph's nodes and initializers are read; attributes, value types, and
    /// metadata are skipped, which prost does by ignoring unknown fields.
    pub(super) fn decode(model_bytes: &[u8]) -> Result<Self> {
        let model = ModelProto::decode(model_bytes).map_err(|error| OcrError::Inference {
            message: "candle backend failed to decode the ONNX model".to_string(),
            source: Some(Box::new(error)),
        })?;
        let graph = model
            .graph
            .ok_or_else(|| OcrError::inference("the ONNX model carries no graph"))?;

        let nodes = graph
            .node
            .into_iter()
            .map(|node| OnnxNode {
                op_type: node.op_type,
                inputs: node.input,
                outputs: node.output,
            })
            .collect();

        let mut initializers = HashMap::with_capacity(graph.initializer.len());
        for tensor in graph.initializer {
            let name = tensor.name.clone();
            initializers.insert(name, decode_tensor(tensor)?);
        }

        Ok(Self { nodes, initializers })
    }

    /// How many nodes carry `op_type`.
    pub(super) fn op_count(&self, op_type: &str) -> usize {
        self.nodes.iter().filter(|node| node.op_type == op_type).count()
    }

    /// Every node whose operator is `op_type`, in graph order.
    pub(super) fn ops(&self, op_type: &str) -> impl Iterator<Item = &OnnxNode> {
        self.nodes.iter().filter(move |node| node.op_type == op_type)
    }

    /// The node that consumes `value` as an operand, if any.
    pub(super) fn consumer_of(&self, value: &str) -> Option<&OnnxNode> {
        self.nodes
            .iter()
            .find(|node| node.inputs.iter().any(|input| input == value))
    }

    /// Look up an initializer by name.
    pub(super) fn initializer(&self, name: &str) -> Result<&OnnxTensor> {
        self.initializers
            .get(name)
            .ok_or_else(|| OcrError::inference(format!("the ONNX graph has no initializer named `{name}`")))
    }
}

/// Convert a `TensorProto` into a shape plus a row-major `f32` buffer.
///
/// Values live in `raw_data` as little-endian `f32` (ONNX fixes the byte order), with
/// `float_data` as the alternative encoding smaller tensors sometimes use.
fn decode_tensor(tensor: TensorProto) -> Result<OnnxTensor> {
    if tensor.data_type != TENSOR_DATA_TYPE_FLOAT {
        return Err(OcrError::inference(format!(
            "initializer `{}` has ONNX data type {} but only f32 ({TENSOR_DATA_TYPE_FLOAT}) is supported",
            tensor.name, tensor.data_type
        )));
    }
    if let Some(&negative) = tensor.dims.iter().find(|&&dim| dim < 0) {
        return Err(OcrError::inference(format!(
            "initializer `{}` has a negative dimension {negative}",
            tensor.name
        )));
    }
    let dims: Vec<usize> = tensor.dims.iter().map(|&dim| dim as usize).collect();
    let expected: usize = dims.iter().product();

    let data = if tensor.raw_data.is_empty() {
        tensor.float_data
    } else {
        if tensor.raw_data.len() % RAW_DATA_BYTES_PER_FLOAT != 0 {
            return Err(OcrError::inference(format!(
                "initializer `{}` has a raw_data length {} that is not a multiple of {RAW_DATA_BYTES_PER_FLOAT}",
                tensor.name,
                tensor.raw_data.len()
            )));
        }
        tensor
            .raw_data
            .chunks_exact(RAW_DATA_BYTES_PER_FLOAT)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    };

    if data.len() != expected {
        return Err(OcrError::inference(format!(
            "initializer `{}` declares shape {dims:?} ({expected} elements) but carries {}",
            tensor.name,
            data.len()
        )));
    }
    Ok(OnnxTensor { dims, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Append a protobuf tag byte for `field` with `wire_type`.
    fn push_tag(buffer: &mut Vec<u8>, field: u32, wire_type: u32) {
        push_varint(buffer, u64::from((field << 3) | wire_type));
    }

    fn push_varint(buffer: &mut Vec<u8>, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                buffer.push(byte);
                return;
            }
            buffer.push(byte | 0x80);
        }
    }

    fn push_bytes(buffer: &mut Vec<u8>, field: u32, payload: &[u8]) {
        push_tag(buffer, field, 2);
        push_varint(buffer, payload.len() as u64);
        buffer.extend_from_slice(payload);
    }

    /// Hand-encode a `TensorProto` from the field numbers in `onnx.proto3`.
    ///
    /// Written byte by byte rather than through this module's own structs, so the test
    /// checks the declared tags against the real schema instead of against itself.
    fn encode_tensor(name: &str, dims: &[i64], values: &[f32]) -> Vec<u8> {
        let mut tensor = Vec::new();
        let mut packed_dims = Vec::new();
        for &dim in dims {
            push_varint(&mut packed_dims, dim as u64);
        }
        push_bytes(&mut tensor, 1, &packed_dims);
        push_tag(&mut tensor, 2, 0);
        push_varint(&mut tensor, TENSOR_DATA_TYPE_FLOAT as u64);
        push_bytes(&mut tensor, 8, name.as_bytes());
        let raw: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        push_bytes(&mut tensor, 9, &raw);
        tensor
    }

    fn encode_node(op_type: &str, inputs: &[&str]) -> Vec<u8> {
        let mut node = Vec::new();
        for input in inputs {
            push_bytes(&mut node, 1, input.as_bytes());
        }
        push_bytes(&mut node, 2, format!("{op_type}.out").as_bytes());
        push_bytes(&mut node, 4, op_type.as_bytes());
        node
    }

    fn encode_model(nodes: &[Vec<u8>], initializers: &[Vec<u8>]) -> Vec<u8> {
        let mut graph = Vec::new();
        for node in nodes {
            push_bytes(&mut graph, 1, node);
        }
        for initializer in initializers {
            push_bytes(&mut graph, 5, initializer);
        }
        let mut model = Vec::new();
        push_bytes(&mut model, 7, &graph);
        model
    }

    #[test]
    fn should_decode_nodes_and_initializers_from_onnx_wire_bytes() {
        let bytes = encode_model(
            &[
                encode_node("Conv", &["image", "conv.w", "conv.b"]),
                encode_node("Relu", &["Conv.out"]),
            ],
            &[encode_tensor("conv.w", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
        );

        let graph = OnnxGraph::decode(&bytes).expect("decode the hand-encoded model");

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].op_type, "Conv");
        assert_eq!(graph.nodes[0].inputs, vec!["image", "conv.w", "conv.b"]);
        assert_eq!(graph.nodes[0].outputs, vec!["Conv.out"]);
        assert_eq!(
            graph.consumer_of("Conv.out").map(|node| node.op_type.as_str()),
            Some("Relu"),
            "the Relu node consumes the Conv result"
        );
        assert_eq!(graph.op_count("Conv"), 1);
        assert_eq!(graph.op_count("LSTM"), 0);
        assert_eq!(
            graph.ops("Conv").next().expect("the Conv node is present").inputs,
            vec!["image", "conv.w", "conv.b"]
        );

        let weight = graph.initializer("conv.w").expect("the initializer is present");
        assert_eq!(weight.dims, vec![2, 3]);
        assert_eq!(weight.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn should_read_raw_data_as_little_endian_f32() {
        // 1.0f32 is 0x3F800000; little-endian on the wire regardless of host order. ~keep
        let mut tensor = Vec::new();
        let mut packed_dims = Vec::new();
        push_varint(&mut packed_dims, 1);
        push_bytes(&mut tensor, 1, &packed_dims);
        push_tag(&mut tensor, 2, 0);
        push_varint(&mut tensor, TENSOR_DATA_TYPE_FLOAT as u64);
        push_bytes(&mut tensor, 8, b"one");
        push_bytes(&mut tensor, 9, &[0x00, 0x00, 0x80, 0x3F]);
        let bytes = encode_model(&[], &[tensor]);

        let graph = OnnxGraph::decode(&bytes).expect("decode");

        assert_eq!(graph.initializer("one").expect("present").data, vec![1.0_f32]);
    }

    #[test]
    fn should_reject_an_initializer_that_is_not_f32() {
        let mut tensor = Vec::new();
        push_tag(&mut tensor, 2, 0);
        push_varint(&mut tensor, 7); // INT64 ~keep
        push_bytes(&mut tensor, 8, b"ints");
        let bytes = encode_model(&[], &[tensor]);

        let error = OnnxGraph::decode(&bytes).expect_err("a non-f32 initializer must be rejected");

        assert!(
            format!("{error}").contains("ints"),
            "the error must name the offending initializer: {error}"
        );
    }

    #[test]
    fn should_reject_an_initializer_whose_shape_disagrees_with_its_data() {
        let bytes = encode_model(&[], &[encode_tensor("mismatch", &[4], &[1.0, 2.0])]);

        let error = OnnxGraph::decode(&bytes).expect_err("a shape mismatch must be rejected");

        assert!(
            format!("{error}").contains("mismatch"),
            "the error must name the offending initializer: {error}"
        );
    }

    #[test]
    fn should_reject_bytes_that_are_not_a_protobuf_model() {
        let error = OnnxGraph::decode(&[0xff, 0xff, 0xff]).expect_err("garbage must not decode");
        assert!(matches!(error, OcrError::Inference { .. }));
    }

    #[test]
    fn should_reject_a_model_without_a_graph() {
        let error = OnnxGraph::decode(&[]).expect_err("an empty model has no graph");
        assert!(format!("{error}").contains("no graph"), "unexpected error: {error}");
    }

    #[test]
    fn should_report_a_missing_initializer_by_name() {
        let graph = OnnxGraph::decode(&encode_model(&[], &[])).expect("decode");
        let error = graph.initializer("absent").expect_err("the initializer is absent");
        assert!(format!("{error}").contains("absent"), "unexpected error: {error}");
    }
}
