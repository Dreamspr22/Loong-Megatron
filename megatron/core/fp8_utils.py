# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions related to FP8 that are used throughout Megatron core"""

import warnings
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, List, Optional, Set

import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_te_version, is_te_min_version

# Check if Transformer Engine is installed
HAVE_TE = False
try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    pass

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except ImportError:
    HAVE_PACKAGING = False

# Check if Transformer Engine has class for fp8 tensors.
HAVE_TE_FP8_TENSOR_CLASS = False
if HAVE_TE:
    if is_te_min_version("2.0"):
        # In TE2.x, QuantizedTensor is the base class for all different type of fp8 tensors,
        # including fp8 tensor for delayed scaling, current scaling and mxfp8, etc.
        from transformer_engine.pytorch.tensor import QuantizedTensor as FP8_TENSOR_CLASS
    else:
        from transformer_engine.pytorch.float8_tensor import Float8Tensor as FP8_TENSOR_CLASS

    HAVE_TE_FP8_TENSOR_CLASS = True
else:
    HAVE_TE_FP8_TENSOR_CLASS = False
    FP8_TENSOR_CLASS = None

# Check if Transformer Engine has MXFP8Tensor class

try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_TE_MXFP8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # MXFP8Tensor not found
    HAVE_TE_MXFP8TENSOR = False

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TELinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    TE_LINEAR_TYPES = (
        TELinear,
        TEColumnParallelLinear,
        TERowParallelLinear,
        TELayerNormColumnParallelLinear,
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )

else:
    TE_LINEAR_TYPES = ()

try:
    from megatron.core.extensions.transformer_engine import Fp8Padding, Fp8Unpadding
except ImportError:
    Fp8Padding = None
    Fp8Unpadding = None


_SELECTIVE_FP8_DEFAULT_ALLOWED_UB_NAMES = frozenset()
Fp8InitDecisionFn = Callable[..., bool]
_selective_fp8_init_decision_fn: Optional[Fp8InitDecisionFn] = None


def register_selective_fp8_init_decision(fn: Fp8InitDecisionFn) -> None:
    """Register a callback that decides per-module FP8 usage at init time.

    Args:
        fn: ``fn(config, *, te_cls, ub_name, init_kwargs) -> bool``.
            Return True to keep FP8 for the module, False to disable.
    """
    global _selective_fp8_init_decision_fn
    _selective_fp8_init_decision_fn = fn


@dataclass
class _SelectiveFp8StateEntry:
    """Per-context selective FP8 state stored in a module-level stack."""

    is_init: bool
    fp8_recipe: Optional[Any] = None
    fp8_group: Optional[Any] = None
    allowed_ub_names: Optional[Set[str]] = None


_SELECTIVE_FP8_STACK: List[_SelectiveFp8StateEntry] = []


def _get_selective_fp8_stack() -> List[_SelectiveFp8StateEntry]:
    return _SELECTIVE_FP8_STACK


def _get_current_selective_fp8_state() -> Optional[_SelectiveFp8StateEntry]:
    """Return the top of the selective FP8 state stack, or None if empty."""
    stack = _get_selective_fp8_stack()
    return stack[-1] if stack else None


def _is_selective_fp8_config(config: TransformerConfig) -> bool:
    return getattr(config, "selective_fp8", False)


def _get_allowed_ub_names_from_config(config: TransformerConfig) -> Set[str]:
    """Extract the set of allowed UB names from config, falling back to defaults."""
    ub_names = getattr(config, "selective_fp8_allowed_ub_names", None)
    return set(ub_names) if ub_names is not None else _SELECTIVE_FP8_DEFAULT_ALLOWED_UB_NAMES


def _push_selective_fp8_state(
    is_init: bool, fp8_recipe=None, fp8_group=None, allowed_ub_names=None,
) -> None:
    _get_selective_fp8_stack().append(
        _SelectiveFp8StateEntry(
            is_init=is_init,
            fp8_recipe=fp8_recipe,
            fp8_group=fp8_group,
            allowed_ub_names=allowed_ub_names,
        )
    )


def _pop_selective_fp8_state() -> None:
    stack = _get_selective_fp8_stack()
    if stack:
        stack.pop()


def _selective_fp8_active() -> bool:
    return _get_current_selective_fp8_state() is not None


def _selective_fp8_is_init() -> bool:
    entry = _get_current_selective_fp8_state()
    return entry is not None and entry.is_init


def _selective_fp8_get_allowed_ub_names() -> Set[str]:
    """Return the set of allowed UB names stored by the enclosing _SelectiveFp8Context."""
    entry = _get_current_selective_fp8_state()
    if entry and entry.allowed_ub_names is not None:
        return entry.allowed_ub_names
    return set(_SELECTIVE_FP8_DEFAULT_ALLOWED_UB_NAMES)


if HAVE_TE:
    # Use the non-deprecated autocast API to avoid DeprecationWarning overhead
    # from transformer_engine.pytorch.fp8_autocast on every call.
    try:
        from transformer_engine.pytorch.quantization import autocast as _te_autocast
    except ImportError:
        _te_autocast = transformer_engine.pytorch.fp8_autocast

    def _disable_fp8_context(is_init: bool):
        if is_init:
            return transformer_engine.pytorch.fp8_model_init(enabled=False)
        return _te_autocast(enabled=False)


    def _keep_fp8_for_ub_name(ub_name: Optional[str]) -> bool:
        return ub_name in _selective_fp8_get_allowed_ub_names()


    def _wrap_forward_with_selective_fp8_guard(cls) -> None:
        """Wrap forward to selectively enter fp8_autocast per module.

        The outer context intentionally does NOT enable fp8_autocast, so all
        modules default to BF16. This wrapper re-enables FP8 only for modules
        that were marked as FP8-eligible at init time
        (``_selective_fp8_disabled=False``).

        Fast path: modules with ``_selective_fp8_disabled=True`` (the majority)
        skip with a single ``getattr`` — no thread-local stack access needed.
        """
        if getattr(cls, "_selective_fp8_forward_wrapped", False):
            return
        original_forward = cls.forward

        @wraps(original_forward)
        def wrapped_forward(self, *args, **kwargs):
            # Fast path: init-time decision already marked this module as BF16.
            if getattr(self, "_selective_fp8_disabled", True):
                return original_forward(self, *args, **kwargs)
            # Only access the thread-local stack for FP8-eligible modules.
            state = _get_current_selective_fp8_state()
            if state is None or state.is_init or state.fp8_recipe is None:
                return original_forward(self, *args, **kwargs)
            with _te_autocast(
                enabled=True, recipe=state.fp8_recipe,
                amax_reduction_group=state.fp8_group,
            ):
                return original_forward(self, *args, **kwargs)

        cls.forward = wrapped_forward
        cls._selective_fp8_forward_wrapped = True


    def _should_disable_fp8_at_init(config, te_cls, ub_name, init_kwargs) -> bool:
        """Determine if FP8 should be disabled for this module at init time."""
        entry = _get_current_selective_fp8_state()
        if entry is None or not entry.is_init:
            return False
        # Use the registered decision callback if available.
        if _selective_fp8_init_decision_fn is not None:
            return not _selective_fp8_init_decision_fn(
                config, te_cls=te_cls, ub_name=ub_name, init_kwargs=init_kwargs)
        # Otherwise fall back to the static whitelist.
        return not _keep_fp8_for_ub_name(ub_name)


    def _wrap_init_with_selective_fp8_guard(
        cls, ub_name_arg: str = "tp_comm_buffer_name",
    ) -> None:
        if getattr(cls, "_selective_fp8_init_wrapped", False):
            return
        original_init = cls.__init__

        @wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            ub_name = kwargs.get(ub_name_arg)
            config = kwargs.get("config")
            should_disable = _should_disable_fp8_at_init(config, cls, ub_name, kwargs)

            ctx = _disable_fp8_context(is_init=True) if should_disable else nullcontext()
            with ctx:
                original_init(self, *args, **kwargs)

            # Persist init-time decision for runtime fast path.
            self._selective_fp8_disabled = should_disable

        cls.__init__ = wrapped_init
        cls._selective_fp8_init_wrapped = True


    def _install_selective_fp8_guards() -> None:
        linear_classes = (
            TELayerNormColumnParallelLinear,
            TEColumnParallelLinear,
            TERowParallelLinear,
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
            TELinear,  # MLA down-projection (parallel_mode='duplicated')
        )
        for linear_cls in linear_classes:
            _wrap_init_with_selective_fp8_guard(linear_cls)
            _wrap_forward_with_selective_fp8_guard(linear_cls)
        # TEDotProductAttention: only init guard needed.
        # At runtime the outer context no longer enables FP8, so attention
        # naturally runs in BF16 without a per-call disable wrapper.
        _wrap_init_with_selective_fp8_guard(TEDotProductAttention, ub_name_arg="__unused__")

        # Coverage check: ensure every TE_LINEAR_TYPES class is guarded.
        _unguarded = set(TE_LINEAR_TYPES) - set(linear_classes)
        if _unguarded:
            _unguarded_names = sorted(c.__name__ for c in _unguarded)
            warnings.warn(
                f"selective_fp8: the following TE linear classes are in "
                f"TE_LINEAR_TYPES but NOT covered by init/forward guards: "
                f"{_unguarded_names}.  This will cause 'quantized weights "
                f"without quantized compute' warnings and incorrect FP8 param "
                f"handling for these modules.  Please add them to "
                f"_install_selective_fp8_guards().",
                UserWarning,
                stacklevel=2,
            )


    _SELECTIVE_FP8_GUARDS_INSTALLED = False

    def _ensure_selective_fp8_guards() -> None:
        """Install selective FP8 guards lazily on first use.

        This avoids monkey-patching TE classes at import time, eliminating
        the per-forward function-call overhead when selective_fp8 is not used.
        """
        global _SELECTIVE_FP8_GUARDS_INSTALLED
        if not _SELECTIVE_FP8_GUARDS_INSTALLED:
            _install_selective_fp8_guards()
            _SELECTIVE_FP8_GUARDS_INSTALLED = True


    def validate_selective_fp8_coverage(model: torch.nn.Module) -> None:
        """Validate that all TE linear modules have selective FP8 attributes set.

        Call this after model construction (when selective_fp8 is enabled) to
        detect TE modules that were created outside the selective FP8 init
        guard, which would result in FP8-weight + BF16-compute mismatches.

        Emits a warning for each unguarded module; raises if any are found.
        """
        unguarded = []
        for name, module in model.named_modules():
            if isinstance(module, TE_LINEAR_TYPES) and not hasattr(
                module, "_selective_fp8_disabled"
            ):
                unguarded.append((name, type(module).__name__))

        if unguarded:
            details = "\n".join(
                f"  - {name} ({cls_name})" for name, cls_name in unguarded
            )
            msg = (
                f"selective_fp8: {len(unguarded)} TE module(s) lack the "
                f"_selective_fp8_disabled attribute, meaning they were created "
                f"without the selective FP8 init guard.  This causes FP8 "
                f"weights + BF16 compute (the 'quantized weights without "
                f"quantized compute' TE warning).  Unguarded modules:\n{details}\n"
                f"Fix: add the missing TE class to _install_selective_fp8_guards()."
            )
            raise RuntimeError(msg)


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor.

    Note that in TE2.x, in order to support more recipes, the design of the fp8 tensor class has
    changed. Now Float8Tensor is only used for current scaling and delayed scaling. And mxfp8
    and blockwise scaling have their own fp8 tensor classes. These different fp8 tensor classes
    are both inherited from QuantizedTensor. So, for TE1.x, FP8_TENSOR_CLASS is Float8Tensor,
    and for TE2.x, FP8_TENSOR_CLASS is QuantizedTensor.
    """
    return HAVE_TE_FP8_TENSOR_CLASS and isinstance(tensor, FP8_TENSOR_CLASS)


def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)


def dequantize_fp8_tensor(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a fp8 tensor to a higher precision tensor."""
    if is_te_min_version("2.0"):
        return fp8_tensor.dequantize()
    else:
        return fp8_tensor.from_float8()


def get_fp8_align_size(fp8_recipe: Fp8Recipe) -> int:
    """Get the alignment size required for fp8 GEMM."""
    if fp8_recipe == Fp8Recipe.mxfp8:
        return 32
    else:
        return 16


"""
The code below abstracts the functionalities needed for implementing "--fp8-param-gather" into
several functions. It provides different implementations for each function based on different
versions of TE, ensuring compatibility across various TE versions.

Currently, there are three functions:
    - modify_underlying_storage
        This function is used in DDP to place all parameters into a contiguous buffer. For
        non-fp8 tensors, replacing their data is simple, just using code like
        "tensor.data = new_data". However, for fp8 tensors, their raw data is not stored in the
        ".data" attribute, and it varies with different TE versions and different recipes. This
        function provides a unified interface to replace the underlying storage of a fp8 tensor.
    - quantize_param_shard
        This function is used in dist-opt to cast fp32 main params to fp8 params. For non-fp8
        params, this casting is as simple as "bf16_params.copy_(fp32_main_params)"; but for fp8
        params, the casting logic varies with different TE versions and different recipes. This
        function provides a unified interface to cast fp32 main params to fp8 params, and also
        updates the necessary attributes (like amax, scale, scale_inv or transpose cache) of the
        fp8 model params.
    - correct_amax_history_if_needed
        This function is used to correct the amax history of fp8 tensors. In TE1.x, some inplace
        copy operations will write unwanted values to the amax_history of fp8 tensors. This function
        corrects the amax_history back. For TE2.x, it's an empty function.
        Only useful for delayed scaling.
"""
if HAVE_TE and is_te_min_version("2.2"):
    # Supported TE versions: 2.2+
    from transformer_engine.pytorch.tensor import QuantizedTensor

    def _modify_underlying_storage_impl(
        fp8_tensor: QuantizedTensor, new_raw_data: torch.Tensor
    ) -> None:
        from transformer_engine.pytorch.tensor.utils import replace_raw_data

        replace_raw_data(fp8_tensor, new_raw_data)

    def _quantize_param_shard_impl(
        model_params: List[QuantizedTensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:
        if len(model_params) == 0:
            return

        from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_fp8

        args = [model_params, main_params, start_offsets, data_parallel_group]
        if fsdp_shard_model_params is not None:
            if not HAVE_PACKAGING:
                raise ImportError(
                    "packaging not found, please install it with `pip install packaging`"
                )
            if get_te_version() == PkgVersion("2.3.0.dev0+5fdd7bb") or is_te_min_version("2.3.0"):
                args.append(fsdp_shard_model_params)
            else:
                raise NotImplementedError(
                    f"FSDP with --fp8-param-gather is not supported in TE v{get_te_version()}"
                )
        cast_master_weights_to_fp8(*args)

    def _correct_amax_history_if_needed_impl(model: List[torch.nn.Module]) -> None:
        pass

elif HAVE_TE and is_te_min_version("2.0"):
    # Supported TE versions: 2.0
    from transformer_engine.pytorch.tensor import QuantizedTensor
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    def _modify_underlying_storage_impl(
        fp8_tensor: QuantizedTensor, new_raw_data: torch.Tensor
    ) -> None:
        old_raw_data = fp8_tensor._data
        assert old_raw_data.dtype == new_raw_data.dtype
        new_raw_data.detach().copy_(old_raw_data)
        fp8_tensor._data = new_raw_data
        del old_raw_data

    def _quantize_param_shard_impl(
        model_params: List[QuantizedTensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:
        # Avoid circular import
        from megatron.core.optimizer.optimizer import _multi_tensor_copy_this_to_that

        if len(model_params) == 0:
            return

        if fsdp_shard_model_params is None:
            fsdp_shard_model_params = [None] * len(model_params)

        for model_param, main_param, start_offset, fsdp_shard_model_param in zip(
            model_params, main_params, start_offsets, fsdp_shard_model_params
        ):
            if main_param is None:
                continue

            if fsdp_shard_model_param is not None:
                shard_model_param = fsdp_shard_model_param
            else:
                shard_model_param = model_param._data.view(-1)[
                    start_offset : start_offset + main_param.numel()
                ]

            quantizer = model_param._quantizer
            # When not using --fp8-param-gather, the main_param (fp32) is first cast to bf16/fp16,
            # and then cast to fp8 during forward.
            # Although it's not necessary when --fp8-param-gather is enabled, we still keep this
            # logic to keep numerical consistency. So here cast the main_param to model_param.dtype.
            main_param = main_param.to(model_param.dtype)
            out = Float8Tensor(
                shape=main_param.size(),
                dtype=model_param.dtype,
                requires_grad=False,
                data=shard_model_param,
                fp8_scale_inv=model_param._scale_inv,
                fp8_dtype=model_param._fp8_dtype,
                quantizer=quantizer,
            )
            quantizer.update_quantized(main_param, out)

        amaxes = []
        scales = []
        scale_invs = []
        for model_param in model_params:
            quantizer = model_param._quantizer
            amaxes.append(quantizer.amax.view(1))
            scales.append(quantizer.scale.view(1))
            scale_invs.append(model_param._scale_inv.view(1))
            model_param._reset_caches()

        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")

        # Update scaling factors.
        packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        _multi_tensor_copy_this_to_that(scales, packed_scale_views, dummy_overflow_buf)
        torch.reciprocal(packed_scales, out=packed_scales)
        _multi_tensor_copy_this_to_that(packed_scale_views, scale_invs, dummy_overflow_buf)

        # Reduce amaxes.
        # Note: Assume each param has a separate amax.
        packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
        torch.distributed.all_reduce(
            packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
        )
        _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)

    def _correct_amax_history_if_needed_impl(model: List[torch.nn.Module]) -> None:
        pass

elif HAVE_TE and is_te_min_version("1.0"):
    # Supported TE versions: 1.0 - 1.14
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    def _modify_underlying_storage_impl(tensor: Float8Tensor, new_raw_data: torch.Tensor) -> None:
        old_raw_data = tensor._data
        assert old_raw_data.dtype == new_raw_data.dtype
        new_raw_data.detach().copy_(old_raw_data)
        tensor._data = new_raw_data
        del old_raw_data

    def _quantize_param_shard_impl(
        model_params: List[Float8Tensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:
        # Avoid circular import
        from megatron.core.optimizer.optimizer import _multi_tensor_copy_this_to_that

        if len(model_params) == 0:
            return

        if fsdp_shard_model_params is None:
            fsdp_shard_model_params = [None] * len(model_params)

        for model_param, main_param, start_offset, fsdp_shard_model_param in zip(
            model_params, main_params, start_offsets, fsdp_shard_model_params
        ):
            if main_param is None:
                continue

            if fsdp_shard_model_param is not None:
                shard_model_param = fsdp_shard_model_param
            else:
                shard_model_param = model_param._data.view(-1)[
                    start_offset : start_offset + main_param.numel()
                ]

            # When not using --fp8-param-gather, the main_param (fp32) is first cast to bf16/fp16,
            # and then cast to fp8 during forward.
            # Although it's not necessary when --fp8-param-gather is enabled, we still keep this
            # logic to keep numerical consistency. So here cast the main_param to model_param.dtype.
            main_param = main_param.to(model_param.dtype)
            cast_to_fp8(
                main_param.view(1, -1),
                model_param._fp8_meta["scaling_fwd"],
                model_param._fp8_meta_index,
                model_param._fp8_dtype,
                out=shard_model_param.view(1, -1),
            )

        amaxes = []
        scales = []
        scale_invs = []
        for model_param in model_params:
            fp8_meta = model_param._fp8_meta["scaling_fwd"]
            fp8_meta_index = model_param._fp8_meta_index
            amaxes.append(fp8_meta.amax_history[0][fp8_meta_index].view(1))
            scales.append(fp8_meta.scale[fp8_meta_index].view(1))
            scale_invs.append(model_param._scale_inv.view(1))
            model_param._reset_caches()

        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")

        # Update scaling factors.
        packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        _multi_tensor_copy_this_to_that(scales, packed_scale_views, dummy_overflow_buf)
        torch.reciprocal(packed_scales, out=packed_scales)
        _multi_tensor_copy_this_to_that(packed_scale_views, scale_invs, dummy_overflow_buf)

        # Reduce amaxes.
        # Note: Assume each param has a separate amax.
        packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
        torch.distributed.all_reduce(
            packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
        )
        _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)

    def _correct_amax_history_if_needed_impl(model: List[torch.nn.Module]) -> None:
        for model_module in model:
            for param in model_module.parameters():
                if is_float8tensor(param) and param._fp8_meta is not None:
                    fp8_meta = param._fp8_meta["scaling_fwd"]
                    fp8_meta_index = param._fp8_meta_index
                    if hasattr(param, "get_high_precision_init_val"):
                        fp8_meta.amax_history[0][fp8_meta_index].copy_(
                            param.get_high_precision_init_val().abs().max()
                        )
                    else:
                        fp8_meta.amax_history[0][fp8_meta_index] = 0

else:
    # Fallback impl if TE version is invalid or TE is not installed.
    def _modify_underlying_storage_impl(*args, **kwargs):
        raise RuntimeError("Invalid Transformer Engine version for FP8 distributed optimizer")

    def _quantize_param_shard_impl(model_params, *args, **kwargs):
        if len(model_params) == 0:
            return
        else:
            # If TE is not installed, there shouldn't be any fp8 params.
            raise RuntimeError("Invalid Transformer Engine version for FP8 distributed optimizer")

    def _correct_amax_history_if_needed_impl(*args, **kwargs):
        # If TE is not installed, we are definitely not using fp8 for training, so no correction
        # is needed.
        pass


# Interface Function
def modify_underlying_storage(tensor: torch.Tensor, new_raw_data: torch.Tensor):
    """Replace the underlying raw data of a tensor with new data."""
    _modify_underlying_storage_impl(tensor, new_raw_data)


# Interface Function
def quantize_param_shard(
    model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params=None
):
    """Cast shard fp32 main params to fp8 model params."""
    _quantize_param_shard_impl(
        model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params
    )


# Interface Function
def correct_amax_history_if_needed(model: List[torch.nn.Module]):
    """Correct the amax history of fp8 tensors when it's necessary (i.e., in TE1.x)."""
    _correct_amax_history_if_needed_impl(model)


def is_first_last_bf16_layer(config: TransformerConfig, layer_no: int):
    """Check if the layer is in bf16."""
    num_bf16_layers_at_start = (
        config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
    )
    num_bf16_layers_at_end = (
        config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
    )
    # Since layer_no is a global layer index, additional checks on whether
    # we are in the first or last pipeline-parallel rank are not needed.
    is_first_layer = layer_no < num_bf16_layers_at_start
    is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

    return layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer)


if HAVE_TE:
    import inspect as _inspect

    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TEDelayedScaling

    # Cache inspect.signature results to avoid repeated reflection.
    _fp8_model_init_params = _inspect.signature(
        transformer_engine.pytorch.fp8_model_init
    ).parameters
    _fp8_model_init_has_recipe = "recipe" in _fp8_model_init_params
    _fp8_model_init_has_preserve = "preserve_high_precision_init_val" in _fp8_model_init_params
    del _fp8_model_init_params

    class _SelectiveFp8Context:
        """Context manager for selective FP8.

        For init (is_init=True): enters the base fp8_model_init context so TE creates
        FP8 params for whitelisted modules (qkv, proj).  Non-whitelisted modules get
        disabled by the per-module init guard.

        For runtime (is_init=False): does NOT enter fp8_autocast at the outer level.
        Instead, stores fp8_recipe/fp8_group so that the per-module forward guard can
        create individual fp8_autocast(enabled=True) contexts only for whitelisted
        modules.  This reduces context switches from 6 disables → 2 enables per layer.
        """

        def __init__(self, base_context, is_init: bool,
                     fp8_recipe=None, fp8_group=None, allowed_ub_names=None):
            self.base_context = base_context
            self.is_init = is_init
            self.fp8_recipe = fp8_recipe
            self.fp8_group = fp8_group
            self.allowed_ub_names = allowed_ub_names

        def __enter__(self):
            _ensure_selective_fp8_guards()
            _push_selective_fp8_state(
                self.is_init, self.fp8_recipe, self.fp8_group, self.allowed_ub_names,
            )
            if self.is_init:
                # Init phase: enter base fp8_model_init context as before
                try:
                    return self.base_context.__enter__()
                except Exception:
                    _pop_selective_fp8_state()
                    raise
            # Runtime: don't enter fp8_autocast; whitelisted modules enter individually
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                if self.is_init:
                    return self.base_context.__exit__(exc_type, exc_val, exc_tb)
            finally:
                _pop_selective_fp8_state()

    def get_fp8_recipe(config: TransformerConfig):
        """Return fp8 recipe.

        Arguments:
            config (TransformerConfig): Configuration object.

        Returns:
            FP8 recipe.
        """
        if config.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif config.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        # Select fp8 recipe (TE version >= 2.1.0).
        fp8_recipe = None
        if is_te_min_version("2.1.0"):
            if config.fp8_recipe == Fp8Recipe.delayed:
                fp8_recipe = TEDelayedScaling(
                    config=config,
                    fp8_format=fp8_format,
                    override_linear_precision=(False, False, not config.fp8_wgrad),
                )
            elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
                fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                    fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention
                )
            elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
                fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(
                    fp8_format=fp8_format
                )
            elif config.fp8_recipe == Fp8Recipe.mxfp8:
                fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
                    fp8_format=fp8_format
                )
            else:
                raise ValueError(
                    "Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and "
                    "DelayedScaling are the only supported FP8 recipes. Please also make sure "
                    "you are using a compatible TE version."
                )
        else:
            # Assert that the user is using delayed scaling.
            assert config.fp8_recipe == Fp8Recipe.delayed, (
                "Please make sure to use TransformerEngine version >= 2.2.0.dev0 for "
                "Float8CurrentScaling, >= 2.1.0 for MXFP8BlockScaling, and >= 2.3.0.dev0 for "
                "Float8BlockScaling."
            )
            fp8_recipe = TEDelayedScaling(
                config=config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not config.fp8_wgrad),
            )
        return fp8_recipe

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Return fp8 context manager.

        Arguments:
            config (TransformerConfig): Configuration object.
            layer_no (int): *Global* layer index (including layers on other
                pipeline-parallel ranks).
            is_init (bool): Whether the context is fp8_model_init (True) or fp8_autocast (False).

        Returns:
            FP8 context.
            If layer_no < 0, we return a fp8 context for all layers regardless of layer_no.
            We return nullcontext() when: a) not using fp8 to train, b) layer_no is a layer
            that needs to be trained in bf16.
        """

        need_fp8_context = config.fp8 if not is_init else config.fp8_param

        if not need_fp8_context or is_first_last_bf16_layer(config, layer_no):
            # BF16 path: no outer FP8 context is needed.
            # For selective-FP8 init, still enter a selective state scope so
            # per-module init guards can run and persist runtime markers.
            fp8_context = nullcontext()
            if _is_selective_fp8_config(config) and is_init:
                allowed_ub_names = _get_allowed_ub_names_from_config(config)
                return _SelectiveFp8Context(
                    fp8_context, is_init=True, allowed_ub_names=allowed_ub_names
                )
            return fp8_context

        fp8_recipe = get_fp8_recipe(config)

        # First / last layer in bf16 isn't supported with delayed scaling since it
        # requires entering/exiting fp8 context per layer, causing incorrect amax
        # reduction behavior.
        assert not (
            config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
        ), "Delayed scaling does not support first / last layer in BF16."

        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(
                with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
            )

        # Selective FP8 runtime: don't create the outer fp8_autocast context.
        # Whitelisted modules (qkv, proj) will enter fp8_autocast individually,
        # avoiding N disable context switches per layer (6 → 2 enable switches).
        if _is_selective_fp8_config(config) and not is_init:
            allowed_ub_names = _get_allowed_ub_names_from_config(config)
            return _SelectiveFp8Context(
                nullcontext(), is_init=False,
                fp8_recipe=fp8_recipe, fp8_group=fp8_group,
                allowed_ub_names=allowed_ub_names,
            )

        if not is_init:
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                    enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
                )
        else:
            context_args = {"enabled": True}
            if _fp8_model_init_has_recipe:
                context_args["recipe"] = fp8_recipe
            if _fp8_model_init_has_preserve:
                context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()
            fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

        if _is_selective_fp8_config(config):
            allowed_ub_names = _get_allowed_ub_names_from_config(config)
            return _SelectiveFp8Context(fp8_context, is_init=is_init,
                                        allowed_ub_names=allowed_ub_names)
        return fp8_context

else:

    def get_fp8_recipe(config: TransformerConfig):
        """Returns None since TE is not available."""
        return None

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Returns dummy fp8 context manager since TE is not available."""
        return nullcontext()


if HAVE_TE:
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    # Modules that have been wrapped for inference for fp8
    _fp8_inference_wrapped_modules = weakref.WeakSet()

    def _wrap_te_linear_for_padding(module: torch.nn.Module):
        """Wrap a TE linear module to automatically pad sequences for FP8 inference.

        Modifies the module's forward method to:
        1. Pad input sequences to FP8 alignment requirements
        2. Run the original forward pass
        3. Unpad outputs to original sequence length

        Args:
            module: A Transformer Engine linear layer (TELinear, TEColumnParallelLinear, etc.)
        """
        if module in _fp8_inference_wrapped_modules:
            return
        _pad_func = Fp8Padding(1)
        _unpad_func = Fp8Unpadding(1)

        original_forward = module.forward

        @wraps(original_forward)
        def padded_forward(input_tensor, *args, **kwargs):
            # Only do padding for fp8 if we are in fp8 context
            if not FP8GlobalStateManager.is_fp8_enabled():
                return original_forward(input_tensor, *args, **kwargs)

            seq_len, batch_size, hidden_size = input_tensor.shape
            # Reshape to (S, B*H) to pad sequence dimension
            input_2d = input_tensor.reshape(seq_len, -1)
            # Pad the sequence dimension
            padded_input_2d, _ = _pad_func(input_2d, [seq_len])
            padded_seq_len = padded_input_2d.shape[0]

            # Reshape back to (padded_S, B, H)
            padded_input_3d = padded_input_2d.view(padded_seq_len, batch_size, hidden_size)
            output = original_forward(padded_input_3d, *args, **kwargs)

            # Handle output
            if isinstance(output, tuple):
                output_tensor = output[0]
                other_outputs = output[1:]
            else:
                output_tensor = output
                other_outputs = ()

            # Unpad output - reshape to 2D, unpad, reshape back
            _, _, output_hidden_size = output_tensor.shape
            output_2d = output_tensor.reshape(padded_seq_len, -1)
            unpadded_output_2d = _unpad_func(output_2d, [seq_len])
            unpadded_output = unpadded_output_2d.reshape(seq_len, batch_size, output_hidden_size)

            if other_outputs:
                return (unpadded_output,) + other_outputs
            else:
                return unpadded_output

        module.forward = padded_forward
        _fp8_inference_wrapped_modules.add(module)

    def prepare_model_for_fp8_inference(model):
        """Prepare a model for FP8 inference by wrapping TE linear layers with padding support.

        FP8 TE Gemms have specific shape requirements. This function wraps all Transformer
        Engine linear layers in the model to automatically pad/unpad sequences during inference.

        Args:
            model (model (GPTModel): Model containing TE linear layers.

        Returns:
            GPTModel: The same model with wrapped linear layers (modified in-place).

        """
        assert Fp8Padding and Fp8Unpadding, "TE version does not have FP8 padding functions"
        # Find and wrap all TE linear layers
        for module in model.modules():
            if isinstance(module, TE_LINEAR_TYPES):
                _wrap_te_linear_for_padding(module)

        return model

else:

    def prepare_model_for_fp8_inference(model):
        """If trys using prepare_model_for_fp8_inference without TE we error"""
        raise RuntimeError(
            "prepare_model_for_fp8_inference requires Transformer Engine to be installed. "
            "Please install transformer-engine to use FP8 inference."
        )
