import torch
import torch.nn as nn

import qem.modeling.backbone
import qem.modeling.fpn
from qem.modeling import registry
from qem.modeling.parsing_head.parsing import Parsing


class Generalized_CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Backbone
        conv_body = registry.BACKBONES[self.cfg.BACKBONE.CONV_BODY]
        self.Conv_Body = conv_body(self.cfg)
        self.dim_in = self.Conv_Body.dim_out
        self.spatial_in = self.Conv_Body.spatial_out

        # Feature Pyramid Networks
        if self.cfg.MODEL.FPN_ON:
            fpn_body = registry.FPN_BODY[self.cfg.FPN.BODY]
            self.Conv_Body_FPN = fpn_body(self.cfg, self.dim_in, self.spatial_in)
            self.dim_in = self.Conv_Body_FPN.dim_out
            self.spatial_in = self.Conv_Body_FPN.spatial_out
        else:
            self.dim_in = self.dim_in[-1:]
            self.spatial_in = self.spatial_in[-1:]

        if self.cfg.MODEL.PARSING_ON:
            self.Parsing = Parsing(self.cfg, self.dim_in, self.spatial_in)

    def forward(self, x, targets=None):
        # Backbone
        conv_features = self.Conv_Body(x)

        # FPN
        if self.cfg.MODEL.FPN_ON:
            conv_features = self.Conv_Body_FPN(conv_features)
        else:
            conv_features = [conv_features[-1]]

        results = []
        losses = {}
        if self.cfg.MODEL.PARSING_ON:
            result_parsing, loss_parsing = self.Parsing(conv_features, targets)
            results.append(result_parsing)
            losses.update(loss_parsing)

        if self.training:
            outputs = {'metrics': {}, 'losses': {}}
            outputs['losses'].update(losses)
            return outputs

        return results

    def conv_body_net(self, x):
        conv_features = self.Conv_Body(x)

        if self.cfg.MODEL.FPN_ON:
            conv_features = self.Conv_Body_FPN(conv_features)
        else:
            conv_features = [conv_features[-1]]
        return conv_features

    def parsing_net(self, conv_features, targets=None):
        result_parsing, loss_parsing = self.Parsing(conv_features, targets)
        return result_parsing
