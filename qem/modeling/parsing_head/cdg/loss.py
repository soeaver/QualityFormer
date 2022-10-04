import torch
from torch.nn import functional as F


def generate_cdg_gt(target, num_parsing=20, device=None):
    h, w = target.shape
    target_c = target.clone()
    target_c[target_c == 255] = 0
    target_c = target_c.long()
    target_c = target_c.view(h * w)
    target_c = target_c.unsqueeze(1)
    target_onehot = torch.zeros(h * w, num_parsing).to(device)
    target_onehot.scatter_(1, target_c, 1)  # h*w, class_num
    target_onehot = target_onehot.transpose(0, 1)
    target_onehot = target_onehot.view(num_parsing, h, w)

    # h distribution ground truth
    row_gt = (torch.sum(target_onehot, dim=2)).float()
    row_gt[0, :] = 0
    max = torch.max(row_gt, dim=1)[0]
    max = max.unsqueeze(1)
    row_gt = row_gt / (max + 1e-5)

    # w distribution gound truth
    col_gt = (torch.sum(target_onehot, dim=1)).float()
    col_gt[0, :] = 0
    max = torch.max(col_gt, dim=1)[0]
    max = max.unsqueeze(1)
    col_gt = col_gt / (max + 1e-5)

    return row_gt, col_gt

    # # ===========================================================
    # hwgt = torch.matmul(hgt.transpose(0, 1), wgt)
    # max = torch.max(hwgt.view(-1), dim=0)[0]
    # hwgt = hwgt / (max + 1.0e-5)
    # # ====================================================================
    # return hgt, wgt, hwgt


class CDGLossComputation(object):
    def __init__(self, cfg):
        self.device = torch.device(cfg.DEVICE)
        self.num_parsing = cfg.PARSING.NUM_PARSING
        self.cdg_loss_weight = cfg.PARSING.CDG.LOSS_WEIGHT

    def __call__(self, row_pred, col_pred, parsing_targets):
        N, H, W = parsing_targets.shape

        row_targets, col_targets = [], []
        for idx in range(N):
            row_tgt, col_tgt = generate_cdg_gt(parsing_targets[idx], self.num_parsing, device=self.device)
            row_targets.append(row_tgt)
            col_targets.append(col_tgt)
        row_targets = torch.stack(row_targets, dim=0).to(self.device, dtype=torch.float)
        col_targets = torch.stack(col_targets, dim=0).to(self.device, dtype=torch.float)

        row_pred = row_pred.unsqueeze(3)
        row_pred = F.interpolate(input=row_pred, size=(H, 1), mode='bilinear', align_corners=True)
        row_pred = row_pred.squeeze(3)

        col_pred = col_pred.unsqueeze(2)
        col_pred = F.interpolate(input=col_pred, size=(1, W), mode='bilinear', align_corners=True)
        col_pred = col_pred.squeeze(2)

        row_loss = torch.mean((row_targets - row_pred) * (row_targets - row_pred))
        col_loss = torch.mean((col_targets - col_pred) * (col_targets - col_pred))

        cdg_loss = (row_loss + col_loss) * self.cdg_loss_weight

        return dict(loss_cdg=cdg_loss)


def cdg_loss_evaluator(cfg):
    loss_evaluator = CDGLossComputation(cfg)
    return loss_evaluator
