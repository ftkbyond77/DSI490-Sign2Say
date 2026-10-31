"""Counterfactual attribute sensitivity, not causal or calibrated confidence."""
import torch
from torch import Tensor
from .model import ZSSLRCompatibilityModel
from .protocol import calibrate

@torch.no_grad()
def FlipDifferenceOperator(model: ZSSLRCompatibilityModel, visual_features: Tensor,
                           text: Tensor, attributes: Tensor, candidates: Tensor,
                           seen_ids: Tensor, gamma: float=0.) -> dict[str,Tensor]:
    """Flip one candidate's one bit at a time, re-normalizing its semantic vector.

    text/attributes contain ONLY candidate rows in candidate order.
    delta[b,j,a,k] = P(k | attribute a of class j flipped) - P(k | original).
    self_delta[b,j,a] tracks the perturbed class's probability. All alternatives
    remain in the softmax denominator. Cost/storage scales with B*K*K*53.
    Caller supplies evaluation-mode visual features to avoid stochastic encoders.
    """
    if len(candidates)!=len(text) or candidates.unique().numel()!=len(candidates):
        raise ValueError('Candidate IDs must uniquely identify prototype rows')
    was_training=model.training
    try:
        model.eval()
        base=calibrate(model.score_features(visual_features,text,attributes),candidates,seen_ids,gamma)
        probability=base.softmax(-1)
        delta=base.new_empty((len(base),len(text),53,len(text)))
        for j in range(len(text)):
            for a in range(53):
                flipped=attributes[j:j+1].clone()
                flipped[:,a]=1-flipped[:,a]
                score=model.score_features(visual_features,text[j:j+1],flipped).squeeze(1)
                if torch.isin(candidates[j],seen_ids): score=score-gamma
                changed=base.clone()
                changed[:,j]=score
                delta[:,j,a,:]=changed.softmax(-1)-probability
        own=torch.stack([delta[:,j,:,j] for j in range(len(text))],dim=1)
        return {'baseline_probability':probability,'probability_shift':delta,
                'self_probability_shift':own,'mean_absolute_impact':own.abs().mean((0,1))}
    finally:
        model.train(was_training)
