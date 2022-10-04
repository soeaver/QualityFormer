import copy
import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from yacs.config import CfgNode

import torch
import torch.nn as nn

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class Optimizer(object):
    def __init__(self, model: nn.Module, solver: CfgNode) -> None:
        self.model = model
        self.optimizer_type = solver.OPTIMIZER

        # lr
        self.base_lr = solver.BASE_LR
        self.bias_lr_factor = solver.BIAS_LR_FACTOR
        self.backbone_lr_factor = solver.BACKBONE_LR_FACTOR
        # weight decay
        self.weight_decay = solver.WEIGHT_DECAY
        self.weight_decay_bias = solver.WEIGHT_DECAY_BIAS
        self.weight_decay_norm = solver.WEIGHT_DECAY_NORM
        self.weight_decay_embed = solver.WEIGHT_DECAY_EMBED
        self.backbone_wd_factor = solver.BACKBONE_WEIGHT_DECAY_FACTOR
        # momentum
        self.momentum = solver.MOMENTUM
        # clip gradients
        self.clip_gradients = solver.CLIP_GRADIENTS
        if self.clip_gradients.ENABLED:
            assert self.optimizer_type in ("SGD", "ADAMW"), "only SGD and ADAMW support clip gradients."

        self.norm_module_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.LocalResponseNorm,
        )

    def get_params(self) -> List[Dict[str, Any]]:
        overrides = {}

        bias_overrides = {}
        if self.bias_lr_factor is not None and self.bias_lr_factor != 1.0:
            bias_overrides["lr"] = self.base_lr * self.bias_lr_factor
        if self.weight_decay_bias is not None and self.weight_decay_bias != self.weight_decay:
            bias_overrides["weight_decay"] = self.weight_decay_bias
        if len(bias_overrides):
            overrides["bias"] = bias_overrides

        defaults = {"lr": self.base_lr, "weight_decay": self.weight_decay}
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in self.model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone." in module_name and self.backbone_lr_factor != 1.0:
                    hyperparams["lr"] = hyperparams["lr"] * self.backbone_lr_factor
                if "backbone." in module_name and self.backbone_wd_factor != 1.0:
                    hyperparams["weight_decay"] = hyperparams["weight_decay"] * self.backbone_wd_factor
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, self.norm_module_types):
                    hyperparams["weight_decay"] = self.weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    print(module_name)
                    hyperparams["weight_decay"] = self.weight_decay_embed
                hyperparams.update(overrides.get(module_param_name, {}))
                params.append({"params": [value], **hyperparams})
        return params

    def build(self) -> torch.optim.Optimizer:
        """
        Returns:
            Optimizer
        """
        def _maybe_add_full_model_gradient_clipping(_optim, clip_grad_cfg):
            clip_norm_val = clip_grad_cfg.CLIP_VALUE
            clip_norm_type = clip_grad_cfg.NORM_TYPE
            enable = clip_grad_cfg.ENABLED and clip_grad_cfg.CLIP_TYPE == "full_model" and clip_norm_val > 0.0

            class FullModelGradientClippingOptimizer(_optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val, norm_type=clip_norm_type)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else _optim

        def _maybe_add_gradient_clipping(_optim, clip_grad_cfg):
            clip_norm_val = clip_grad_cfg.CLIP_VALUE
            enable = clip_grad_cfg.ENABLED and clip_grad_cfg.CLIP_TYPE != "full_model" and clip_norm_val > 0.0
            if not enable:
                return _optim

            if isinstance(_optim, torch.optim.Optimizer):
                optimizer_type = type(_optim)
            else:
                assert issubclass(_optim, torch.optim.Optimizer), _optim
                optimizer_type = _optim

            grad_clipper = _create_gradient_clipper(
                clip_grad_cfg.CLIP_TYPE, clip_grad_cfg.CLIP_VALUE, clip_grad_cfg.NORM_TYPE
            )
            OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
                optimizer_type, per_param_clipper=grad_clipper
            )
            if isinstance(_optim, torch.optim.Optimizer):
                _optim.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
                return _optim
            else:
                return OptimizerWithGradientClip

        params = self.get_params()
        if self.optimizer_type == "SGD":    # for supporting full model clip
            optimizer = _maybe_add_full_model_gradient_clipping(torch.optim.SGD, self.clip_gradients)(
                params, self.base_lr, momentum=self.momentum
            )
        elif self.optimizer_type == "ADAMW":    # for supporting full model grad clip
            optimizer = _maybe_add_full_model_gradient_clipping(torch.optim.AdamW, self.clip_gradients)(
                params, self.base_lr
            )
        elif self.optimizer_type == "ADAM":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == 'RMSPROP':
            optimizer = torch.optim.RMSprop(
                self.get_params(),
                momentum=self.momentum,
            )
        else:
            raise NotImplementedError(f"no optimizer type {self.optimizer_type}")

        if not self.clip_gradients.CLIP_TYPE == "full_model" and self.optimizer_type in ("SGD", "ADAMW"):
            # for supporting value / norm grad clip
            optimizer = _maybe_add_gradient_clipping(optimizer, self.clip_gradients)

        return optimizer


def _create_gradient_clipper(clip_type, clip_value, norm_type=None) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm.
    """
    def clip_grad_norm(p: _GradientClipperInput):
        assert norm_type is not None
        torch.nn.utils.clip_grad_norm_(p, clip_value, norm_type)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, clip_value)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        "value": clip_grad_value,
        "norm": clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[clip_type]


def _generate_optimizer_class_with_gradient_clipping(
        optimizer: Type[torch.optim.Optimizer],
        *,
        per_param_clipper: Optional[_GradientClipper] = None,
        global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert (
            per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
        )
    return OptimizerWithGradientClip
