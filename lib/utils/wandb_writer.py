import os
import torch

try:
    import wandb
except:
    pass


class WandbLogger(object):
    def __init__(self, cfg):
        wandb.init(
            config=cfg,
            name=cfg.WANDB.NAME if cfg.WANDB.NAME != "" else os.path.basename(cfg.CKPT.strip('/')),
            entity=cfg.WANDB.ENTITY,
            project=cfg.WANDB.PROJECT,
        )

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, metrics):
        log_dict = dict()
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            log_dict[k] = v

        wandb.log(log_dict, step=self.step)

    def flush(self):
        pass
