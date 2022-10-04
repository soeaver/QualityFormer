import torch
from torch import nn
from torch.nn import functional as F

from lib.layers import make_conv, make_norm, make_act

from qem.modeling import registry


@registry.CDG_HEADS.register("cdg_head")
class CDGHead(nn.Module):
    """
    CDG head.
    """

    def __init__(self, cfg, dim_in, spatial_in):
        super(CDGHead, self).__init__()

        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in[-1]

        num_parsing = cfg.PARSING.NUM_PARSING
        h, w = cfg.PARSING.CDG.FEAT_HW
        conv_dim = cfg.PARSING.CDG.CONV_DIM
        up_kernel_size = cfg.PARSING.CDG.UP_KERNEL_SIZE

        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.row_pool = nn.AdaptiveAvgPool2d((h, 1))
        self.col_pool = nn.AdaptiveAvgPool2d((1, w))
        self.row_conv1d = nn.Sequential(
            nn.Conv1d(self.dim_in, conv_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
        )
        self.col_conv1d = nn.Sequential(
            nn.Conv1d(self.dim_in, conv_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
        )

        self.row_pred = nn.Sequential(
            nn.Conv1d(conv_dim, num_parsing, 3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.col_pred = nn.Sequential(
            nn.Conv1d(conv_dim, num_parsing, 3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.row_up = nn.Sequential(
            nn.Conv1d(conv_dim, self.dim_in, up_kernel_size, stride=1, padding=up_kernel_size // 2, bias=True),
            nn.Sigmoid(),
        )
        self.col_up = nn.Sequential(
            nn.Conv1d(conv_dim, self.dim_in, up_kernel_size, stride=1, padding=up_kernel_size // 2, bias=True),
            nn.Sigmoid(),
        )

        self.fusion = make_conv(
            self.dim_in * 3, self.dim_in, kernel_size=3, stride=1,
            norm=make_norm(self.dim_in, norm="BN"),
            act=make_act()
        )

        self.dim_out = [num_parsing, num_parsing, dim_in[-1]]
        self.spatial_out = [(h, 1), (1, w), spatial_in[-1]]

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
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x[-1]
        N, C, H, W = x.size()

        x_row = self.row_pool(x).squeeze(3)  # N, C, H
        x_col = self.col_pool(x).squeeze(2)  # N, C, W
        x_row = self.row_conv1d(x_row)
        x_col = self.col_conv1d(x_col)

        row_pred = self.row_pred(x_row)  # N, PARSING_NUM, H
        col_pred = self.col_pred(x_col)  # N, PARSING_NUM, W

        x_row = self.row_up(x_row)
        x_col = self.col_up(x_col)
        x_row_up = x_row.unsqueeze(3)
        x_col_up = x_col.unsqueeze(2)
        x_row_up = F.interpolate(x_row_up, (H, W), mode='bilinear', align_corners=True)  # N, C, H, W
        x_col_up = F.interpolate(x_col_up, (H, W), mode='bilinear', align_corners=True)  # N, C, H, W
        x_row_col = self.beta * x_row_up + self.gamma * x_col_up
        x_aug = x * x_row_col

        x = torch.cat([x, x_aug, x_row_col], dim=1)
        x = self.fusion(x)

        return row_pred, col_pred, x
