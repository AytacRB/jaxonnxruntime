


"""Define ONNX Resize and Upsample operators."""

from collections.abc import Callable, Sequence
from typing import Any
import functools

import jax.image
import jax.numpy as jnp
from jax import jit

from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


def _as_str(x):
  if isinstance(x, bytes):
    return x.decode()
  return x


def _method_from_mode(mode):
  mode = _as_str(mode)

  if mode == "nearest":
    return "nearest"
  if mode in ("linear", "bilinear"):
    return "linear"
  if mode == "cubic":
    return "cubic"

  raise NotImplementedError(f"Resize/Upsample mode not supported: {mode}")


def _static_tuple(x):
  x = jnp.asarray(x)
  return tuple(int(v) for v in x.tolist())


def _shape_from_scales(input_shape, scales):
  scales = jnp.asarray(scales)
  scales = [float(v) for v in scales.tolist()]
  return tuple(int(input_shape[i] * scales[i]) for i in range(len(input_shape)))


@functools.partial(jit, static_argnames=("out_shape", "method"))
def _resize_jit(x, *, out_shape, method):
  return jax.image.resize(
      x,
      shape=out_shape,
      method=method,
      antialias=False,
  )


@handler.register_op("Resize")
class Resize(handler.Handler):
  """Implementation of ONNX Resize."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode):
    node.attrs_dict["mode"] = _as_str(node.attrs.get("mode", "nearest"))
    node.attrs_dict["coordinate_transformation_mode"] = _as_str(
        node.attrs.get("coordinate_transformation_mode", "half_pixel")
    )
    node.attrs_dict["nearest_mode"] = _as_str(
        node.attrs.get("nearest_mode", "round_prefer_floor")
    )

  @classmethod
  def version_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    cls._prepare(node)
    return onnx_resize

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    cls._prepare(node)
    return onnx_resize

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    cls._prepare(node)
    return onnx_resize


def onnx_resize(
    x,
    roi=None,
    scales=None,
    sizes=None,
    *,
    mode,
    coordinate_transformation_mode,
    nearest_mode,
):
  if x.ndim != 4:
    raise NotImplementedError(
        f"Resize currently only supports 4D NCHW, got {x.shape}"
    )

  coordinate_transformation_mode = _as_str(coordinate_transformation_mode)
  nearest_mode = _as_str(nearest_mode)

  if coordinate_transformation_mode not in ("half_pixel", "asymmetric"):
    raise NotImplementedError(
        "Resize coordinate mode not supported yet: "
        f"{coordinate_transformation_mode}"
    )

  if nearest_mode not in ("floor", "round_prefer_floor"):
    raise NotImplementedError(
        f"Resize nearest_mode not supported yet: {nearest_mode}"
    )

  method = _method_from_mode(mode)

  if sizes is not None:
    out_shape = _static_tuple(sizes)
  elif scales is not None:
    out_shape = _shape_from_scales(x.shape, scales)
  else:
    raise ValueError("Resize requires either scales or sizes")

  return _resize_jit(x, out_shape=out_shape, method=method)


@handler.register_op("Upsample")
class Upsample(handler.Handler):
  """Implementation of deprecated ONNX Upsample."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode):
    node.attrs_dict["mode"] = _as_str(node.attrs.get("mode", "nearest"))

  @classmethod
  def version_7(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    cls._prepare(node)
    return onnx_upsample

  @classmethod
  def version_9(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    cls._prepare(node)
    return onnx_upsample


def onnx_upsample(x, scales, *, mode):
  if x.ndim != 4:
    raise NotImplementedError(
        f"Upsample currently only supports 4D NCHW, got {x.shape}"
    )

  method = _method_from_mode(mode)
  out_shape = _shape_from_scales(x.shape, scales)

  return _resize_jit(x, out_shape=out_shape, method=method)