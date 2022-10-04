import torch
from torch import nn
from torch.nn import functional as F

from lib.layers import make_conv, make_norm, make_fc, make_act

from qem.modeling.parsing_head.cdg import heads
from qem.modeling.parsing_head.cdg.loss import cdg_loss_evaluator
from qem.modeling import registry


class CDG(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(CDG, self).__init__()
        self.dim_in = dim_in
        self.spatial_in = spatial_in

        head = registry.CDG_HEADS[cfg.PARSING.CDG.CDG_HEAD]
        self.Head = head(cfg, self.dim_in, self.spatial_in)

        self.loss_evaluator = cdg_loss_evaluator(cfg)

        self.dim_out = self.Head.dim_out[-1:]
        self.spatial_out = self.Head.spatial_out[-1:]

    def forward(self, features, parsing_targets=None):
        """
        Arguments:
            features (Tensor): feature-maps from possibly several levels
            parsing_targets (Tensor, optional): the ground-truth parsing targets.

        Returns:
            losses (Tensor): During training, returns the losses for the
                head. During testing, returns an empty dict.
            encoded_features (Tensor): during training, returns None. During testing, the predicted parsingiou.
        """
        row_pred, col_pred, cdg_features = self.Head(features)

        if self.training:
            return self._forward_train(row_pred, col_pred, cdg_features, parsing_targets)
        else:
            return self._forward_test(cdg_features)

    def _forward_train(self, row_pred, col_pred, cdg_features, parsing_targets=None):
        loss_cdg = self.loss_evaluator(row_pred, col_pred, parsing_targets)
        return loss_cdg, [cdg_features]

    def _forward_test(self, cdg_features):
        return {}, [cdg_features]
