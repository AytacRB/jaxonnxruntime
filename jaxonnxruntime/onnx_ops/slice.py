# Copyright 2026 The Jaxonnxruntime Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Define ONNX Slice operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any
import jax
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
import numpy as np

@handler.register_op('Slice')
class Slice(handler.Handler):
  """Implementation of the ONNX Slice operator."""

  @classmethod
  def _prepare_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    starts = node.attrs.get('starts')
    node.attrs_dict['starts'] = starts
    node.attrs_dict['ends'] = node.attrs.get('ends')
    node.attrs_dict['axes'] = node.attrs.get(
        'axes', tuple(i for i in range(len(starts)))
    )
    node.attrs_dict['steps'] = None

  @classmethod
  def _prepare_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    
    node.attrs_dict['starts'] = _to_static_tuple(inputs[1],node.name, "starts")
    node.attrs_dict['ends'] = _to_static_tuple(inputs[2],node.name, "ends")
    if len(inputs) >= 4:
      node.attrs_dict['axes'] = tuple(inputs[3].tolist())
    else:
      node.attrs_dict['axes'] = None
    if len(inputs) >= 5:
      node.attrs_dict['steps'] = tuple(inputs[4].tolist())
    else:
      node.attrs_dict['steps'] = None

  @classmethod
  def _prepare_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    cls._prepare_10(node, inputs, onnx_jax_impl)

  @classmethod
  def _prepare_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    cls._prepare_10(node, inputs, onnx_jax_impl)

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Slice op."""
    cls._prepare_1(node, inputs, onnx_slice)
    return onnx_slice

  @classmethod
  def version_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_10 Slice op."""
    cls._prepare_10(node, inputs, onnx_slice)
    return onnx_slice

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 Slice op."""
    cls._prepare_11(node, inputs, onnx_slice)
    return onnx_slice

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Slice op."""
    cls._prepare_13(node, inputs, onnx_slice)
    return onnx_slice


@functools.partial(jax.jit, static_argnames=('starts', 'ends', 'axes', 'steps'))
def onnx_slice(*input_args, starts, ends, axes, steps):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Slice."""
  x = input_args[0]
  if axes is None:
    axes = tuple(range(len(starts)))
  if steps is None:
    steps = [1] * len(starts)
  slices = tuple(
      slice(start, end, step) for start, end, step in zip(starts, ends, steps)
  )
  sub_indx = [slice(None)] * len(x.shape)
  for i, axis in enumerate(axes):
    sub_indx[axis] = slices[i]
  return x[tuple(sub_indx)]



def _to_static_tuple(x,node_name, name: str):
    # Normal eager JAX / numpy / Python array case.
    if hasattr(x, "tolist") and not x.__class__.__name__.endswith("Tracer"):
        try:
            y = x.tolist()
            if isinstance(y, (int, float)):
                y = [y]
            return tuple(int(v) for v in y)
        except Exception:
            pass

    # numpy fallback
    try:
        y = np.asarray(x).tolist()
        if isinstance(y, (int, float)):
            y = [y]
        return tuple(int(v) for v in y)
    except Exception:
        pass

    # JAX device_get fallback.
    try:
        y = jax.device_get(x).tolist()
        if isinstance(y, (int, float)):
            y = [y]
        return tuple(int(v) for v in y)
    except Exception:
        pass

    # IBPTracer with degenerate bounds.
    for attr in ("bounds", "val", "value"):
        if hasattr(x, attr):
            obj = getattr(x, attr)

            if hasattr(obj, "concrete"):
                lb, ub = obj.concrete
                lb = jax.device_get(lb)
                ub = jax.device_get(ub)

                if bool(np.all(np.asarray(lb) == np.asarray(ub))):
                    y = np.asarray(lb).tolist()
                    if isinstance(y, (int, float)):
                        y = [y]
                    return tuple(int(v) for v in y)

                raise ValueError(
                    f"{node_name}: Dynamic Slice {name} has non-degenerate IBP bounds: "
                    f"lb={lb}, ub={ub}"
                )

            try:
                y = jax.device_get(obj).tolist()
                if isinstance(y, (int, float)):
                    y = [y]
                return tuple(int(v) for v in y)
            except Exception:
                pass

    raise TypeError(
        f"Could not extract static Slice {name} from {type(x)}; "
        f"repr={x!r}; shape={getattr(x, 'shape', None)}; "
        f"dtype={getattr(x, 'dtype', None)}"
    )