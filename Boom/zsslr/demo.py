"""Offline deterministic CPU smoke pipeline; random metrics are NOT research results."""
import json
import torch
from zsslr import ZSSLRCompatibilityModel, training_loss, predict, evaluate, FlipDifferenceOperator

def main() -> None:
    """Train one seen-only step, evaluate both modes, and inspect attribute shifts."""
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model=ZSSLRCompatibilityModel(width=8)
    body=torch.randn(4,4,3,32,32)
    hands=torch.randn(4,4,3,24,24)
    text=torch.randn(8,768)
    attrs=torch.randint(0,2,(8,53)).float()
    seen=torch.tensor([4,0,6,2]); unseen=torch.tensor([1,3,5,7])
    labels=torch.tensor([4,0,6,2])
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    losses=training_loss(model,body,hands,text,attrs,labels,seen)
    optimizer.zero_grad(); losses['loss'].backward(); optimizer.step()
    z,zi=predict(model,body,hands,text,attrs,seen,unseen)
    g,gi=predict(model,body,hands,text,attrs,seen,unseen,'gzsslr',.1)
    model.eval()
    with torch.no_grad(): v=model.visual(body,hands); c=model.semantic(text,attrs)
    flips=FlipDifferenceOperator(model,v,text[gi],attrs[gi],gi,seen,.1)
    assert v.shape==(4,512) and c.shape==(8,117) and model.W.shape==(512,117)
    assert z.shape==(4,4) and g.shape==(4,8)
    assert flips['probability_shift'].shape==(4,8,53,8)
    assert torch.allclose(v.norm(dim=-1),torch.ones(4),atol=1e-5)
    assert torch.allclose(c.norm(dim=-1),torch.ones(8),atol=1e-5)
    print(json.dumps({'synthetic_only':True,'shapes':{'v':list(v.shape),'c':list(c.shape),
       'W':list(model.W.shape),'flip':list(flips['probability_shift'].shape)},
       'loss':{k:x.item() for k,x in losses.items()},
       'zsslr':evaluate(z,unseen,zi),
       'gzsslr':evaluate(g,torch.tensor([4,0,1,3]),gi,seen)},indent=2))

if __name__=='__main__': main()
