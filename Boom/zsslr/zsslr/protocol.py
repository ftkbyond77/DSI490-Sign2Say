"""Seen-only optimization, candidate selection, calibration and accuracy metrics."""
import torch
from torch import Tensor
import torch.nn.functional as F
from .model import ZSSLRCompatibilityModel

def validate_ids(ids: Tensor, total: int) -> None:
    """Require nonempty unique long indices into a global class table."""
    if ids.ndim != 1 or ids.dtype != torch.long or ids.numel()==0:
        raise ValueError('Class IDs must be a nonempty long vector')
    if ids.unique().numel()!=ids.numel() or (ids<0).any() or (ids>=total).any():
        raise ValueError('Duplicate or out-of-range class IDs')

def local_targets(labels: Tensor, candidates: Tensor) -> Tensor:
    """Map global class IDs to candidate-column indices, rejecting absent labels."""
    if labels.ndim!=1 or labels.dtype!=torch.long:
        raise ValueError('Labels must be a long vector')
    matches=labels[:,None]==candidates[None,:]
    if not (matches.sum(1)==1).all():
        raise ValueError('Every target must appear exactly once in candidates')
    return matches.long().argmax(1)

def training_loss(model: ZSSLRCompatibilityModel, body: Tensor, hands: Tensor,
                  text: Tensor, attributes: Tensor, labels: Tensor,
                  seen_ids: Tensor, lambda_reg: float=1e-4) -> dict[str,Tensor]:
    """Only seen prototypes enter CE; labels are global row IDs, not local columns."""
    validate_ids(seen_ids,len(text))
    if lambda_reg<0: raise ValueError('lambda_reg must be nonnegative')
    targets=local_targets(labels,seen_ids)
    logits=model(body,hands,text[seen_ids],attributes[seen_ids])
    ce=F.cross_entropy(logits,targets)
    reg=lambda_reg*model.W.square().sum()
    return {'loss':ce+reg,'ce':ce,'regularization':reg}

def calibrate(logits: Tensor, candidates: Tensor, seen_ids: Tensor, gamma: float) -> Tensor:
    """Subtract gamma from seen columns without modifying original logits."""
    return logits-gamma*torch.isin(candidates,seen_ids).to(logits.dtype)[None,:]

@torch.no_grad()
def predict(model: ZSSLRCompatibilityModel, body: Tensor, hands: Tensor,
            text: Tensor, attributes: Tensor, seen_ids: Tensor, unseen_ids: Tensor,
            mode: str='zsslr', gamma: float=0.) -> tuple[Tensor,Tensor]:
    """Return logits and corresponding global IDs; temporarily use evaluation mode."""
    validate_ids(seen_ids,len(text)); validate_ids(unseen_ids,len(text))
    if torch.isin(seen_ids,unseen_ids).any(): raise ValueError('Seen/unseen overlap')
    if mode not in ('zsslr','gzsslr'): raise ValueError('Unknown mode')
    ids=unseen_ids if mode=='zsslr' else torch.cat((seen_ids,unseen_ids))
    was_training=model.training
    try:
        model.eval()
        logits=model(body,hands,text[ids],attributes[ids])
        return (calibrate(logits,ids,seen_ids,gamma) if mode=='gzsslr' else logits),ids
    finally:
        model.train(was_training)

def accuracy(logits: Tensor, labels: Tensor, candidates: Tensor, k: int=1,
             macro: bool=False) -> float:
    """Fraction correct; macro averages represented classes. Top-5 uses min(5,K)."""
    if len(labels)==0: raise ValueError('Empty evaluation subset')
    targets=local_targets(labels,candidates)
    hits=(logits.topk(min(k,len(candidates)),dim=1).indices==targets[:,None]).any(1).float()
    if macro:
        return torch.stack([hits[labels==c].mean() for c in labels.unique()]).mean().item()
    return hits.mean().item()

def evaluate(logits: Tensor, labels: Tensor, candidates: Tensor,
             seen_ids: Tensor | None=None) -> dict[str,float]:
    """Top-1/5 micro and macro; GZSL S/U/H use per-class top-1 accuracy."""
    result={f'{kind}top{k}':accuracy(logits,labels,candidates,k,macro)
            for kind,macro in (('',False),('macro_',True)) for k in (1,5)}
    if seen_ids is not None:
        mask=torch.isin(labels,seen_ids)
        if not mask.any() or mask.all(): raise ValueError('H requires both seen and unseen samples')
        s=accuracy(logits[mask],labels[mask],candidates,1,True)
        u=accuracy(logits[~mask],labels[~mask],candidates,1,True)
        result.update(S=s,U=u,H=2*s*u/(s+u) if s+u else 0.)
    return result
