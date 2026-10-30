"""SignBERT-inspired hand-stream implementation without MANO. See METHOD.md for differences."""
import math
import torch
from torch import nn

class GraphLayer(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        adjacency = torch.eye(21)
        for finger in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)):
            for a,b in zip(finger,finger[1:]): adjacency[a,b]=adjacency[b,a]=1
        adjacency /= adjacency.sum(-1,keepdim=True)
        self.register_buffer('adjacency',adjacency)
        self.linear=nn.Linear(cin,cout)
    def forward(self,x):
        return torch.relu(self.linear(torch.einsum('ij,...jc->...ic',self.adjacency,x)))

class HandTransformer(nn.Module):
    def __init__(self, classes, dim=128, layers=3, heads=4, frames=32):
        super().__init__()
        self.config=dict(classes=classes,dim=dim,layers=layers,heads=heads,frames=frames)
        self.graph=nn.Sequential(GraphLayer(3,64),GraphLayer(64,dim))
        self.chirality=nn.Embedding(2,dim)
        times=torch.arange(frames).float().unsqueeze(1)
        frequencies=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000)/dim))
        positional=torch.zeros(frames,dim)
        positional[:,0::2]=torch.sin(times*frequencies)
        positional[:,1::2]=torch.cos(times*frequencies)
        self.register_buffer('positional',positional)
        layer=nn.TransformerEncoderLayer(dim,heads,dim*4,dropout=0.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.attention=nn.Linear(dim,1)
        self.classifier=nn.Linear(dim,classes)
    def encode(self,xy,visible):
        # B,T,2,21,2; third graph channel distinguishes missing joints from zeros.
        b,t=xy.shape[:2]
        x=torch.cat((xy,visible.unsqueeze(-1).float()),-1)
        h=self.graph(x).mean(-2)
        h=h+self.positional[:t][None,:,None,:]+self.chirality.weight[None,None,:,:]
        return self.encoder(h.reshape(b,t*2,-1)).reshape(b,t,2,-1)
    def forward(self,xy,visible):
        h=self.encode(xy,visible)
        b,t,_,d=h.shape
        h=h.reshape(b,t*2,d)
        valid=visible.any(-1).reshape(b,t*2)
        if not valid.any(-1).all(): raise ValueError('No observed hand tokens in one or more clips')
        scores=self.attention(h).squeeze(-1).masked_fill(~valid,-1e4)
        pooled=(h*scores.softmax(-1).unsqueeze(-1)).sum(1)
        return self.classifier(pooled)

def masked_input(x, visible, generator=None):
    """50% hand tokens: equal joint/frame/identity branches (2021 paper)."""
    target=x.clone(); masked=x.clone(); observed=visible.clone()
    selected=(torch.rand(x.shape[:3],device=x.device,generator=generator)<0.5)&visible.any(-1)
    if not selected.any():
        candidates=visible.any(-1).nonzero()
        if len(candidates): selected[tuple(candidates[0])]=True
    operation=torch.randint(3,x.shape[:3],device=x.device,generator=generator)
    for index in selected.nonzero():
        b,t,h=index.tolist(); op=int(operation[b,t,h])
        if op==0:
            count=int(torch.randint(1,22,(),device=x.device,generator=generator))
            joints=torch.randperm(21,device=x.device,generator=generator)[:count]
            if float(torch.rand((),device=x.device,generator=generator))<0.5:
                masked[b,t,h,joints]=0; observed[b,t,h,joints]=False
            else:
                masked[b,t,h,joints]+=0.05*torch.randn((count,2),device=x.device,generator=generator)
        elif op==1:
            masked[b,t,h]=0; observed[b,t,h]=False
    loss_mask=selected.unsqueeze(-1)&visible
    return masked,observed,target,loss_mask

def reconstruction_loss(pred,target,mask):
    if not mask.any(): raise ValueError('No observed joints available for reconstruction')
    return (pred-target).abs()[mask].mean()

class CoordinateDecoder(nn.Module):
    """Direct 2D joint reconstruction, replacing the MANO decoder explicitly."""
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,42))
    def forward(self,h):
        return self.net(h).reshape(*h.shape[:-1],21,2)
