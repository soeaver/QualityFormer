import torch.nn as nn
import torch.nn.functional as F

from qem.modeling import registry


@registry.PARSING_OUTPUTS.register("conv1x1_outputs")
class Conv1x1Outputs(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super().__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in
        self.edge_on = cfg.PARSING.EDGE_ON

        self.classify = nn.Conv2d(self.dim_in, cfg.PARSING.NUM_PARSING, kernel_size=1, stride=1, padding=0)
        if self.edge_on:
            self.edge = nn.Conv2d(self.dim_in, 2, kernel_size=1, stride=1, padding=0)

        self.dim_out = [cfg.PARSING.NUM_PARSING]
        self.spatial_out = [1.0]
        if self.edge_on:
            self.dim_out += [2]
            self.spatial_out += [1.0]

    def forward(self, x):
        x = x[-1]
        x_parsing = self.classify(x)
        if self.edge_on:
            x_edge = self.edge(x)

        up_scale = int(1 / self.spatial_in[0])
        if up_scale > 1:
            x_parsing = F.interpolate(x_parsing, scale_factor=up_scale, mode="bilinear", align_corners=False)
            if self.edge_on:
                x_edge = F.interpolate(x_edge, scale_factor=up_scale, mode="bilinear", align_corners=False)
        if self.edge_on:
            return [x_parsing, x_edge]

        return [x_parsing]
