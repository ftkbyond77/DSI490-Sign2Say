"""MANO-based decoder; model assets must be provided by the user."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

# MANO's 16 skeletal joints followed by five tips -> MediaPipe's wrist/finger ordering.
JOINT_ORDER=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
# Paper vertex set, reordered thumb/index/middle/ring/little. Visually verify against licensed assets.
TIP_VERTICES=[734,333,443,555,678]

def asset_paths(directory):
    directory=Path(directory)
    paths=[directory/'MANO_LEFT.pkl',directory/'MANO_RIGHT.pkl']
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing licensed MANO assets: '+', '.join(missing)+'. Obtain from https://mano.is.tue.mpg.de/ . No fallback decoder is used.')
    return paths

class ManoDecoder(nn.Module):
    def __init__(self,dim,directory):
        super().__init__()
        paths=asset_paths(directory)
        from smplx import MANO
        self.hands=nn.ModuleList([MANO(str(path),is_rhand=(i==1),use_pca=True,num_pca_comps=25,
            flat_hand_mean=False,create_betas=False,create_global_orient=False,
            create_hand_pose=False,create_transl=False) for i,path in enumerate(paths)])
        for layer in self.hands:
            layer.requires_grad_(False)
        # 25 PCA pose + 10 shape + 6D rotation + xy translation + positive scale.
        self.parameters_head=nn.Linear(dim,44)
        nn.init.normal_(self.parameters_head.weight,std=0.001)
        nn.init.zeros_(self.parameters_head.bias)
        with torch.no_grad():
            self.parameters_head.bias[35:41]=torch.tensor([1.,0.,0.,0.,1.,0.])
    def forward(self,h):
        b,t,_,_=h.shape
        parameters=self.parameters_head(h)
        pose=parameters[...,:25]; beta=parameters[...,25:35]
        rotation=parameters[...,35:41].reshape(b,t,2,2,3)
        a=F.normalize(rotation[...,0,:],dim=-1)
        c=rotation[...,1,:]; c=F.normalize(c-(a*c).sum(-1,keepdim=True)*a,dim=-1)
        rotation=torch.stack([a,c,torch.cross(a,c,dim=-1)],dim=-1)
        offset=parameters[...,41:43]; scale=F.softplus(parameters[...,43:44])+1e-6
        points=[]
        for hand,layer in enumerate(self.hands):
            output=layer(betas=beta[:,:,hand].reshape(-1,10),hand_pose=pose[:,:,hand].reshape(-1,25),
                         global_orient=h.new_zeros((b*t,3)),return_verts=True)
            joints=torch.cat([output.joints[:,:16],output.vertices[:,TIP_VERTICES]],dim=1)[:,JOINT_ORDER]
            joints=joints-joints[:,:1]
            joints=joints.reshape(b,t,21,3)
            rotated=torch.einsum('btij,btnj->btni',rotation[:,:,hand],joints)
            points.append(scale[:,:,hand,None,:]*rotated[...,:2]+offset[:,:,hand,None,:])
        return torch.stack(points,dim=2),pose,beta

def mano_regularization(pose,beta,selected):
    if not selected.any(): return pose.sum()*0
    magnitude=pose.square().sum(-1)+10*beta.square().sum(-1)
    # Adjacent same-hand shape regularization; never compare the end of one hand with the other.
    temporal=(beta[:,1:]-beta[:,:-1]).square().sum(-1)
    pairs=selected[:,1:]&selected[:,:-1]
    derivative=temporal[pairs].mean() if pairs.any() else beta.sum()*0
    return magnitude[selected].mean()+100*derivative
