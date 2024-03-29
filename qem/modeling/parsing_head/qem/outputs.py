from torch import nn
from torch.nn import functional as F

from qem.modeling import registry


@registry.QEM_OUTPUTS.register("qem_output")
class QEMOutput(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(QEMOutput, self).__init__()

        dim_in_p, dim_in_i = dim_in
        self.spatial_in_p, _ = spatial_in

        self.parsing_score = nn.Conv2d(dim_in_p, cfg.PARSING.NUM_PARSING, kernel_size=1, stride=1, padding=0)
        self.parsing_iou = nn.Linear(dim_in_i, 1)

        self.dim_out = [cfg.PARSING.NUM_PARSING, 1]
        self.spatial_out = [self.spatial_in_p, (1, 1)]

        self._init_weights()

    def _init_weights(self):
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xp, xi = x

        xp = self.parsing_score(xp)
        xi = self.parsing_iou(xi.view(xi.size(0), -1))

        return xp, xi
