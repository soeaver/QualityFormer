import torch
import torch.nn.functional as F

from qem.modeling import registry
from qem.modeling.parsing_head import heads, outputs
from qem.modeling.parsing_head.loss import parsing_loss_evaluator
from qem.modeling.parsing_head.parsingiou.parsingiou import ParsingIoU
from qem.modeling.parsing_head.qem.qem import QEM


class Parsing(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(Parsing, self).__init__()
        self.dim_in = dim_in
        self.spatial_in = spatial_in
        self.parsingiou_on = cfg.PARSING.PARSINGIOU_ON
        self.qem_on = cfg.PARSING.QEM_ON
        self.qem_num = cfg.PARSING.QEM.STACK_NUM

        head = registry.PARSING_HEADS[cfg.PARSING.PARSING_HEAD]
        self.Head = head(cfg, self.dim_in, self.spatial_in)
        self.dim_in = self.Head.dim_out

        if self.qem_on:
            if self.qem_num == 1:
                self.QEM = QEM(cfg, self.dim_in, self.spatial_in)
                self.dim_in = self.QEM.dim_out
            else:
                for i in range(1, self.qem_num + 1):
                    setattr(self, 'QEM' + str(i), QEM(cfg, self.dim_in, self.spatial_in))
                    self.dim_in = getattr(self, 'QEM' + str(i)).dim_out

        output = registry.PARSING_OUTPUTS[cfg.PARSING.PARSING_OUTPUT]
        self.Output = output(cfg, self.dim_in, self.Head.spatial_out)

        self.loss_evaluator = parsing_loss_evaluator(cfg)

        if self.parsingiou_on:
            self.ParsingIoU = ParsingIoU(cfg, self.Head.dim_out, self.Head.spatial_out)

        self.dim_out = self.Output.dim_out
        self.spatial_out = self.Output.spatial_out

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features)

    def _forward_train(self, conv_features, targets=None):
        losses = dict()

        x = self.Head(conv_features)

        if self.qem_on:
            if self.qem_num == 1:
                loss_qem, x = self.QEM(x, targets['parsing'])
                losses.update(loss_qem)
            else:
                for i in range(1, self.qem_num + 1):
                    loss_qem, x = getattr(self, 'QEM' + str(i))(x, targets['parsing'])
                    for l_k, l_v in loss_qem.items():
                        losses['{}_{}'.format(l_k, i)] = l_v * i

        logits = self.Output(x)

        parsing_loss, parsingiou_targets = self.loss_evaluator(logits, targets['parsing'])
        losses.update(parsing_loss)

        if self.parsingiou_on:
            parsingiou_losses, _ = self.ParsingIoU(x, parsingiou_targets)
            losses.update(parsingiou_losses)

        return None, losses

    def _forward_test(self, conv_features):
        x = self.Head(conv_features)

        if self.qem_on:
            if self.qem_num == 1:
                _, x = self.QEM(x, None)
            else:
                for i in range(1, self.qem_num + 1):
                    _, x = getattr(self, 'QEM' + str(i))(x, None)

        logits = self.Output(x)

        output = F.softmax(logits[0], dim=1)
        results = dict(
            probs=output,
            parsing_iou_scores=torch.ones(output.size()[0], dtype=torch.float32, device=output.device)
        )

        if self.parsingiou_on:
            _, parsingiou = self.ParsingIoU(x, None)
            results.update(dict(parsing_iou_scores=parsingiou.squeeze(1)))

        return results, {}
