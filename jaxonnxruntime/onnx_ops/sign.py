


"""Define ONNX Sign operator."""

from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax.numpy as jnp
from jax import jit
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("Sign")
class Sign(handler.Handler):
  """Implementation of the ONNX Sign operator."""

  @classmethod
  def version_9(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_9 Sign op."""
    return onnx_sign

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Sign op."""
    return onnx_sign


@functools.partial(jit, static_argnames=())
def onnx_sign(*input_args):
  assert len(input_args) == 1
  x = input_args[0]
  return jnp.sign(x)