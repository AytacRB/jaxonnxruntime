

"""Define ONNX Floor operator."""

from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp

from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("Floor")
class Floor(handler.Handler):
  """Implementation of the ONNX Floor operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Floor op."""
    cls._prepare(node, inputs, onnx_floor)
    return onnx_floor

  @classmethod
  def version_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_6 Floor op."""
    cls._prepare(node, inputs, onnx_floor)
    return onnx_floor

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Floor op."""
    cls._prepare(node, inputs, onnx_floor)
    return onnx_floor


@functools.partial(jax.jit, static_argnames=())
def onnx_floor(x):
  """ impl for ONNX Floor."""
  return jnp.floor(x)