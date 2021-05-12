import numpy as np
import torch
from torch.nn import functional as F

from lib.ops import lovasz_softmax_loss


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def cal_one_mean_iou(image_array, label_array, num_parsing):
    hist = fast_hist(label_array, image_array, num_parsing).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / union
    return iu


def prepare_edge_targets(parsing_targets, device, edge_width=3):
    N, H, W = parsing_targets.shape
    edge = torch.zeros(parsing_targets.shape).to(device, dtype=torch.float)

    # right
    edge_right = edge[:, 1:H, :]
    edge_right[(parsing_targets[:, 1:H, :] != parsing_targets[:, :H - 1, :]) & (parsing_targets[:, 1:H, :] != 255) &
               (parsing_targets[:, :H - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :, :W - 1]
    edge_up[(parsing_targets[:, :, :W - 1] != parsing_targets[:, :, 1:W]) & (parsing_targets[:, :, :W - 1] != 255) &
            (parsing_targets[:, :, 1:W] != 255)] = 1

    # up-right
    edge_upright = edge[:, :H - 1, :W - 1]
    edge_upright[(parsing_targets[:, :H - 1, :W - 1] != parsing_targets[:, 1:H, 1:W])
                 & (parsing_targets[:, :H - 1, :W - 1] != 255) & (parsing_targets[:, 1:H, 1:W] != 255)] = 1

    # bottom-right
    edge_bottomright = edge[:, :H - 1, 1:W]
    edge_bottomright[(parsing_targets[:, :H - 1, 1:W] != parsing_targets[:, 1:H, :W - 1])
                     & (parsing_targets[:, :H - 1, 1:W] != 255) & (parsing_targets[:, 1:H, :W - 1] != 255)] = 1

    kernel = torch.ones((1, 1, edge_width, edge_width)).to(device, dtype=torch.float)
    with torch.no_grad():
        edge = edge.unsqueeze(1)
        edge = F.conv2d(edge, kernel, stride=1, padding=1)
    edge[edge != 0] = 1
    edge = edge.squeeze()
    return edge


class ParsingLossComputation(object):
    def __init__(self, cfg):
        self.device = torch.device(cfg.DEVICE)
        self.edge_on = cfg.PARSING.EDGE_ON
        self.parsingiou_on = cfg.PARSING.PARSINGIOU_ON
        self.num_parsing = cfg.PARSING.NUM_PARSING
        self.loss_weight = cfg.PARSING.LOSS_WEIGHT
        self.lovasz_loss_weight = cfg.PARSING.LOVASZ_LOSS_WEIGHT
        self.edge_loss_weight = cfg.PARSING.EDGE_LOSS_WEIGHT
        self.edge_width = cfg.PARSING.EDGE_WIDTH

    def __call__(self, logits, parsing_targets):
        parsing_logits = logits[0]
        losses = dict()

        if self.parsingiou_on:
            pred_parsings_np = parsing_logits.detach().argmax(dim=1).cpu().numpy()
            parsing_targets_np = parsing_targets.cpu().numpy()

            N = parsing_targets_np.shape[0]
            parsingiou_targets = np.zeros(N, dtype=np.float)

            for _ in range(N):
                parsing_iou = cal_one_mean_iou(parsing_targets_np[_], pred_parsings_np[_], self.num_parsing)
                parsingiou_targets[_] = np.nanmean(parsing_iou)
            parsingiou_targets = torch.from_numpy(parsingiou_targets).to(self.device, dtype=torch.float)
        else:
            parsingiou_targets = None

        parsing_targets = parsing_targets.to(self.device)
        parsing_loss = F.cross_entropy(parsing_logits, parsing_targets, reduction='mean', ignore_index=255)
        parsing_loss *= self.loss_weight
        losses["loss_parsing"] = parsing_loss

        if self.lovasz_loss_weight:
            lovasz_loss = lovasz_softmax_loss(parsing_logits, parsing_targets)
            lovasz_loss *= self.lovasz_loss_weight
            losses["loss_lovasz"] = lovasz_loss

        if self.edge_on:
            edge_logits = logits[-1]
            edge_targets = prepare_edge_targets(parsing_targets, self.device, edge_width=self.edge_width)
            edge_targets = edge_targets.to(self.device, dtype=torch.long)

            pos_num = torch.sum(edge_targets == 1, dtype=torch.float)
            neg_num = torch.sum(edge_targets == 0, dtype=torch.float)
            weight_pos = neg_num / (pos_num + neg_num)
            weight_neg = pos_num / (pos_num + neg_num)
            weights = torch.tensor([weight_neg, weight_pos])  # edge loss weight
            weights = weights.to(self.device)

            edge_loss = F.cross_entropy(edge_logits, edge_targets, weight=weights, reduction='mean', ignore_index=255)
            edge_loss *= self.edge_loss_weight
            losses["loss_edge"] = edge_loss

        return losses, parsingiou_targets


def parsing_loss_evaluator(cfg):
    loss_evaluator = ParsingLossComputation(cfg)
    return loss_evaluator
