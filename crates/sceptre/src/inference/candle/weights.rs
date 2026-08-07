//! Map ONNX initializers onto the [`VarBuilder`] paths the hand-written networks request.
//!
//! Initializer names are not used as paths. Roughly half of them are exporter-generated
//! counters (`onnx::Conv_299`, `onnx::MatMul_413`) that shift whenever the models are
//! re-exported, and ADR 0025 makes re-export a routine operation. Names are assigned
//! positionally instead — the `n`-th `Conv` node becomes `conv.{n}` — which depends only on
//! the graph's topology. Every tensor is then shape-checked when the network asks for it,
//! because `VarBuilder`'s map backend rejects a mismatched shape by name.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::candle_error;
use super::onnx_proto::{OnnxGraph, OnnxTensor};
use crate::error::{OcrError, Result};

/// Gates as ONNX orders them in an LSTM weight block: input, output, forget, cell.
///
/// `candle_nn::LSTM::step` chunks the same block as input, forget, cell, output, so the
/// blocks are re-stacked in this order when the weights are loaded.
const ONNX_GATE_ORDER_AS_CANDLE: [usize; 4] = [0, 2, 3, 1];

/// Gates per LSTM weight block.
const LSTM_GATE_COUNT: usize = 4;

/// Build the tensor map for `graph` and wrap it in a [`VarBuilder`].
pub(super) fn var_builder(graph: &OnnxGraph, device: &Device) -> Result<VarBuilder<'static>> {
    let tensors = tensor_map(graph, device)?;
    Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
}

/// Collect every trained tensor under its canonical path.
fn tensor_map(graph: &OnnxGraph, device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();
    collect_convolutions(graph, device, &mut tensors)?;
    collect_linears(graph, device, &mut tensors)?;
    collect_recurrent_layers(graph, device, &mut tensors)?;
    Ok(tensors)
}

/// `Conv` weights and biases keep ONNX's `[out, in, kh, kw]` layout, which candle shares.
fn collect_convolutions(graph: &OnnxGraph, device: &Device, tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    for (index, node) in graph.ops("Conv").enumerate() {
        let weight = operand(graph, node.inputs.get(1), "Conv", index, "weight")?;
        tensors.insert(format!("conv.{index}.weight"), to_tensor(weight, device)?);
        if let Some(bias) = node.inputs.get(2) {
            let bias = graph.initializer(bias)?;
            tensors.insert(format!("conv.{index}.bias"), to_tensor(bias, device)?);
        }
    }
    Ok(())
}

/// `MatMul` stores its weight as `[in, out]`; `candle_nn::Linear` wants `[out, in]`.
///
/// The bias is the `Add` that consumes the `MatMul` result, so it is found by following
/// the graph rather than by assuming the two node orders line up.
fn collect_linears(graph: &OnnxGraph, device: &Device, tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    for (index, node) in graph.ops("MatMul").enumerate() {
        let weight = operand(graph, node.inputs.get(1), "MatMul", index, "weight")?;
        let transposed = to_tensor(weight, device)?
            .t()
            .and_then(|weight| weight.contiguous())
            .map_err(|error| candle_error(&format!("transpose the linear.{index} weight"), error))?;
        tensors.insert(format!("linear.{index}.weight"), transposed);

        let produced = node
            .outputs
            .first()
            .ok_or_else(|| OcrError::inference(format!("the MatMul node at position {index} produces no output")))?;
        let add = graph.consumer_of(produced).filter(|node| node.op_type == "Add");
        let add = add.ok_or_else(|| {
            OcrError::inference(format!(
                "the MatMul node at position {index} is not followed by a bias Add"
            ))
        })?;
        let bias_name = add
            .inputs
            .iter()
            .find(|input| *input != produced)
            .ok_or_else(|| OcrError::inference(format!("the bias Add after MatMul {index} has no bias operand")))?;
        tensors.insert(
            format!("linear.{index}.bias"),
            to_tensor(graph.initializer(bias_name)?, device)?,
        );
    }
    Ok(())
}

/// Split each bidirectional ONNX `LSTM` into the two per-direction layers candle expects.
///
/// ONNX packs a layer as `W[dirs, 4H, in]`, `R[dirs, 4H, H]` and `B[dirs, 8H]`, where the
/// bias row holds the input-side and hidden-side biases back to back and the four gate
/// blocks run input, output, forget, cell. candle splits the biases into separate tensors
/// and orders the gates input, forget, cell, output, so both are rearranged here.
fn collect_recurrent_layers(graph: &OnnxGraph, device: &Device, tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    for (index, node) in graph.ops("LSTM").enumerate() {
        let input_weight = to_tensor(operand(graph, node.inputs.get(1), "LSTM", index, "weight")?, device)?;
        let hidden_weight = to_tensor(
            operand(graph, node.inputs.get(2), "LSTM", index, "recurrence weight")?,
            device,
        )?;
        let biases = to_tensor(operand(graph, node.inputs.get(3), "LSTM", index, "bias")?, device)?;

        let hidden = input_weight.dim(1).map_err(wrap("read the LSTM weight shape"))? / LSTM_GATE_COUNT;
        let directions = input_weight.dim(0).map_err(wrap("read the LSTM direction count"))?;

        for direction in 0..directions {
            // ONNX orders the directions forward then reverse; candle spells the second ~keep
            // one with a `_reverse` suffix on every weight name. ~keep
            let suffix = if direction == 0 { "" } else { "_reverse" };
            let per_direction = |packed: &Tensor, label: &str| -> Result<Tensor> {
                let slice = packed
                    .narrow(0, direction, 1)
                    .and_then(|slice| slice.squeeze(0))
                    .map_err(|error| candle_error(&format!("select the {label} for LSTM {index}"), error))?;
                Ok(slice)
            };

            let weight_ih = reorder_gates(&per_direction(&input_weight, "input weight")?, hidden, index)?;
            let weight_hh = reorder_gates(&per_direction(&hidden_weight, "hidden weight")?, hidden, index)?;
            tensors.insert(format!("rnn.{index}.weight_ih_l0{suffix}"), weight_ih);
            tensors.insert(format!("rnn.{index}.weight_hh_l0{suffix}"), weight_hh);

            let packed_bias = per_direction(&biases, "bias")?;
            let gate_width = LSTM_GATE_COUNT * hidden;
            for (offset, name) in [(0, "bias_ih"), (gate_width, "bias_hh")] {
                let half = packed_bias
                    .narrow(0, offset, gate_width)
                    .map_err(|error| candle_error(&format!("split the {name} for LSTM {index}"), error))?;
                tensors.insert(
                    format!("rnn.{index}.{name}_l0{suffix}"),
                    reorder_gates(&half, hidden, index)?,
                );
            }
        }
    }
    Ok(())
}

/// Re-stack the four leading gate blocks of `packed` from ONNX's order into candle's.
fn reorder_gates(packed: &Tensor, hidden: usize, index: usize) -> Result<Tensor> {
    let blocks: Vec<Tensor> = ONNX_GATE_ORDER_AS_CANDLE
        .iter()
        .map(|gate| packed.narrow(0, gate * hidden, hidden))
        .collect::<candle_core::Result<_>>()
        .map_err(|error| candle_error(&format!("split the gates of LSTM {index}"), error))?;
    Tensor::cat(&blocks, 0)
        .and_then(|reordered| reordered.contiguous())
        .map_err(|error| candle_error(&format!("reorder the gates of LSTM {index}"), error))
}

/// Resolve an operand name to its initializer, naming the node when it is absent.
fn operand<'graph>(
    graph: &'graph OnnxGraph,
    name: Option<&String>,
    op_type: &str,
    index: usize,
    label: &str,
) -> Result<&'graph OnnxTensor> {
    let name = name
        .ok_or_else(|| OcrError::inference(format!("the {op_type} node at position {index} has no {label} operand")))?;
    graph.initializer(name)
}

/// Copy an ONNX initializer into a candle tensor on `device`.
fn to_tensor(tensor: &OnnxTensor, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(tensor.data.clone(), tensor.dims.as_slice(), device)
        .map_err(wrap("build a candle tensor from an ONNX initializer"))
}

/// Adapt a candle error into an [`OcrError`] with a fixed operation description.
fn wrap(operation: &str) -> impl Fn(candle_core::Error) -> OcrError + '_ {
    move |error| candle_error(operation, error)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `narrow` the four gate blocks back out so a permutation is readable in a test.
    fn gate_blocks(tensor: &Tensor, hidden: usize) -> Vec<Vec<f32>> {
        (0..LSTM_GATE_COUNT)
            .map(|gate| {
                tensor
                    .narrow(0, gate * hidden, hidden)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn should_reorder_gates_from_onnx_iofc_into_candle_ifco() {
        // One row per gate, each row a distinguishable constant, so the permutation is
        // visible in the output rather than inferred. ~keep
        let packed = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (4, 1), &Device::Cpu).expect("build");

        let reordered = reorder_gates(&packed, 1, 0).expect("reorder");

        assert_eq!(
            gate_blocks(&reordered, 1),
            vec![vec![1.0], vec![3.0], vec![4.0], vec![2.0]],
            "ONNX input/output/forget/cell must become candle input/forget/cell/output"
        );
    }

    #[test]
    fn should_preserve_the_shape_when_reordering_gates() {
        let packed = Tensor::zeros((8, 3), DType::F32, &Device::Cpu).expect("zeros");

        let reordered = reorder_gates(&packed, 2, 0).expect("reorder");

        assert_eq!(reordered.dims(), &[8, 3]);
    }
}
