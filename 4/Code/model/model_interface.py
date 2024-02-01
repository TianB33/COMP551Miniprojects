import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from .model_utils import cuda
import torch.nn as nn
import os
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import logging

def fmax(probs, labels):
    thresholds = np.arange(0, 1, 0.01)
    f_max = 0.0

    for threshold in thresholds:
        precision = 0.0
        recall = 0.0
        precision_cnt = 0
        recall_cnt = 0
        for idx in range(probs.shape[0]):
            prob = probs[idx]
            label = labels[idx]
            pred = (prob > threshold).astype(np.int32)
            correct_sum = np.sum(label*pred)
            pred_sum = np.sum(pred)
            label_sum = np.sum(label)
            if pred_sum > 0:
                precision += correct_sum/pred_sum
                precision_cnt += 1
            if label_sum > 0:
                recall += correct_sum/label_sum
            recall_cnt += 1
        if recall_cnt > 0:
            recall = recall / recall_cnt
        else:
            recall = 0
        if precision_cnt > 0:
            precision = precision / precision_cnt
        else:
            precision = 0
        f = (2.*precision*recall)/max(precision+recall, 1e-8)
        f_max = max(f, f_max)

    return f_max

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class MInterface(pl.LightningModule):
    def __init__(self, loss_weights = None, model_name=None, steps_per_epoch=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.loss_weights = loss_weights
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)
        

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)


        logpath = str(os.path.join(self.hparams.res_dir, self.hparams.ex_name, "log.log"))
        file_handler = logging.FileHandler(logpath)
        file_handler.setLevel(logging.INFO)


        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)


        logger.addHandler(file_handler)
        self.textlogger = logger
        
        

    def forward(self, batch):
        # batch = batch.to(self.device)
        logits = self.model(batch)
        if self.hparams.dataset in ['ec','go']:
            return {"logits": logits, "pred_probs":logits.sigmoid()}
        
        if self.hparams.dataset in ['fold', 'func', 'go']:
            return {"logits": logits, "pred_probs":logits.log_softmax(dim=-1)}
        

    def training_step(self, batch, batch_idx):
        # batch = batch.to(self.device)
        results = self(batch)
        if self.hparams.dataset in ['ec','go']:
            pred_probs = results['pred_probs']
            y = batch.y
            loss = self.loss_fn(pred_probs, y)
        
        if self.hparams.dataset in ['fold', 'func']:
            logits = results['logits']
            loss = self.loss_fn(logits.log_softmax(dim=-1), batch.y)
        
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            
        return loss

    def validation_step(self, batch, batch_idx, test=False):
        # batch = batch.to(self.device)
        with torch.no_grad():
            results = self(batch)
            
        if self.hparams.dataset in ['ec','go']:
            pred_probs = results['pred_probs']
            y = batch.y
            valid_fmax = fmax(pred_probs.cpu().numpy(), y.cpu().numpy())
            if not test:
                self.log_dict({"valid_fmax": valid_fmax}, on_epoch=True)
            else:
                self.log_dict({f"test_fmax": valid_fmax}, on_epoch=True)
        
        if self.hparams.dataset in ['fold', 'func']:
            logits = results['logits']
            pred = logits.max(1)[1]
            acc = (pred == batch.y).float().mean()
            if not test:
                self.log_dict({"valid_acc": acc}, on_epoch=True)
                
            else:
                self.log_dict({f"test_acc": acc}, on_epoch=True)
                
        return self.log_dict

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx, test=True)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
        # scheduler = self.lr_schedulers()
        # if self.hparams.lr_scheduler == 'plateau':
        #     scheduler.step(metrics=self.monitor_metric, epoch=self.current_epoch)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        
        optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.parameters()), lr=self.hparams.lr, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, self.parameters()), lr=self.hparams.lr, weight_decay=weight_decay,  momentum=self.hparams.momentum)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                lr_weights = []
                for i, milestone in enumerate(self.hparams.lr_milestones):
                    if i == 0:
                        lr_weights += [np.power(self.hparams.lr_gamma, i)] * milestone
                    else:
                        lr_weights += [np.power(self.hparams.lr_gamma, i)] * (milestone - self.hparams.lr_milestones[i-1])
                if self.hparams.lr_milestones[-1] < self.hparams.epoch:
                    lr_weights += [np.power(self.hparams.lr_gamma, len(self.hparams.lr_milestones))] * (self.hparams.epoch + 1 - self.hparams.lr_milestones[-1])
                lambda_lr = lambda epoch: lr_weights[epoch]
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
                return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
                return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
            elif self.hparams.lr_scheduler == 'onecycle':
                scheduler = lrs.OneCycleLR(optimizer, max_lr=self.hparams.lr, total_steps=self.hparams.epoch * self.hparams.steps_per_epoch, three_phase=True)
                # scheduler = lrs.OneCycleLR(optimizer, max_lr=self.hparams.lr, total_steps=self.hparams.epoch * self.hparams.steps_per_epoch)
                return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
            if self.hparams.lr_scheduler == 'plateau':
                scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
                return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": self.hparams.monitor_metric}]
            else:
                raise ValueError('Invalid lr_scheduler type!')
        
    def lr_scheduler_step(self, *args, **kwargs):
        scheduler = self.lr_schedulers()
        if self.hparams.lr_scheduler == 'plateau':
            metric = self.trainer.logged_metrics[self.hparams.monitor_metric]
            scheduler.step(metrics=metric)
        else:
            scheduler.step()
    
    def configure_devices(self):
        self.device = torch.device(self.hparams.device)

    def configure_loss(self):
        if self.hparams.dataset in ['ec','go']:
            if self.loss_weights is not None:
                self.loss_fn = torch.nn.BCELoss(weight=torch.as_tensor(self.loss_weights).to(self.device))
            else:
                self.loss_fn = torch.nn.BCELoss()
        
        if self.hparams.dataset in ['fold', 'func']:
            self.loss_fn = torch.nn.NLLLoss()

    def load_model(self):
        from model.models import Model
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = list(inspect.signature(Model.__init__).parameters)[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)