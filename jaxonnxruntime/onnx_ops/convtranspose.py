"""Define ONNX ConvTranspose operator."""

from collections.abc import Callable, Sequence
from typing import Any
import functools

import jax
import jax.numpy as jnp
from jax import jit

from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("ConvTranspose")
class ConvTranspose(handler.Handler):
  """Implementation of the ONNX ConvTranspose operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode):
    node.attrs_dict["strides"] = tuple(node.attrs.get("strides", [1, 1]))
    node.attrs_dict["pads"] = tuple(node.attrs.get("pads", [0, 0, 0, 0]))
    node.attrs_dict["dilations"] = tuple(node.attrs.get("dilations", [1, 1]))
    node.attrs_dict["group"] = int(node.attrs.get("group", 1))
    node.attrs_dict["output_padding"] = tuple(
        node.attrs.get("output_padding", [0, 0])
    )


  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    cls._prepare(node)
    return onnx_convtranspose


  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    cls._prepare(node)
    return onnx_convtranspose

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    cls._prepare(node)
    return onnx_convtranspose

def _crop_onnx_convtranspose_pads(y, pads):
  if pads is None:
    return y

  top, left, bottom, right = [int(v) for v in pads]

  h_end = None if bottom == 0 else -bottom
  w_end = None if right == 0 else -right

  return y[:, :, top:h_end, left:w_end]


@functools.partial(
    jit,
    static_argnames=("strides", "pads", "dilations", "group", "output_padding"),
)
def onnx_convtranspose(
    x,
    w,
    b=None,
    *,
    strides,
    pads,
    dilations,
    group,
    output_padding,
):
  assert x.ndim == 4
  assert w.ndim == 4

  if group != 1:
    raise NotImplementedError("ConvTranspose group != 1 not implemented yet")

  if any(v != 0 for v in output_padding):
    raise NotImplementedError("ConvTranspose output_padding not implemented yet")

  y = jax.lax.conv_transpose(
      lhs=x,
      rhs=w,
      strides=strides,
      padding="VALID",
      rhs_dilation=dilations,
      dimension_numbers=("NCHW", "OIHW", "NCHW"), #  onnx uses (N, C, H, W)
      transpose_kernel=True,
  )

  y = _crop_onnx_convtranspose_pads(y, pads) # onnx padding argument controls cropping

  if b is not None:
    y = y + jnp.reshape(b, (1, -1, 1, 1))

  return y