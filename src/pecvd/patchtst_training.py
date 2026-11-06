"""Original rich-SSL objective and explicit convergence controls for STEP 1."""
from dataclasses import dataclass
import math
import torch


def legacy_rich_objective(model,x,mask,config):
    """Restore the backed-up masked + visible/detail training objective.

Visible detail supervision is auxiliary, NOT blind reconstruction performance.
Only separately evaluated blind validation MSE selects checkpoints.
"""
    target,patches=model.patchify(x)
    pred=model(x,mask).float();clean=model(x).float();target=target.float()
    masked=((pred-target).square()*mask[...,None]).sum()/(mask.sum().clamp_min(1)*target.shape[-1])
    visible=(clean-target).square().mean()
    slope=(clean.diff(dim=-1)-target.diff(dim=-1)).square().mean()
    clean4=clean.reshape(len(x),model.n_sensors,patches,model.patch_len)
    truth4=target.reshape_as(clean4)
    def relation(z):
        centered=z-z.mean(-1,keepdim=True)
        unit=centered/centered.square().sum(-1,keepdim=True).clamp_min(1e-4).sqrt()
        return unit.permute(0,2,1,3)@unit.permute(0,2,3,1)
    relationship=(relation(clean4)-relation(truth4)).square().mean()
    from .pretrained_patchtst import detail_loss
    variability,peak=detail_loss(clean,target)
    parts={'masked_mse':masked,'visible_mse':visible,'slope':slope,
           'relationship':relationship,'variability':variability,'peak':peak}
    total=(masked+config.clean_weight*visible+config.slope_weight*slope+
           config.relationship_weight*relationship+config.variability_weight*variability+config.peak_weight*peak)
    return total,parts


@dataclass
class ValidationSelection:
    """Validation-only patience. Checks, not epochs, increment the stale count."""
    min_epochs:int=40
    patience:int=12
    min_delta:float=1e-6
    best:float=math.inf
    best_epoch:int=0
    stale:int=0

    def observe(self,value,epoch):
        if not math.isfinite(value):raise ValueError('Finite validation blind MSE required')
        improved=value < self.best
        significant=value < self.best-self.min_delta
        if improved:self.best=value;self.best_epoch=epoch
        if significant:self.stale=0
        else:self.stale+=1
        return improved,epoch>=self.min_epochs and self.stale>=self.patience


def apply_warmup(optimizer,base_rates,epoch,warmup_epochs):
    if len(base_rates)!=len(optimizer.param_groups):raise ValueError('Optimizer group mismatch')
    if warmup_epochs<0 or epoch<1:raise ValueError('Valid warmup/epoch required')
    if warmup_epochs and epoch<=warmup_epochs:
        for group,base in zip(optimizer.param_groups,base_rates):group['lr']=base*epoch/warmup_epochs
