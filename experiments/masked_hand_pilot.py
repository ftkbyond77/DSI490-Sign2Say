"""Local pilot on an explicit prepared cache; never overwrites serving weights.
Cache NPZ (no pickle): body/left/right/face_all [B,T,V,3], valid [B,T] bool,
observed_left/right [B,T,21] bool, labels [B], clip_id [B], role [B] train/val.
Observed masks must originate before interpolation, remain aligned through prep.
This pilot uses SupCon, NOT the full production CosFace+SupCon sampler baseline.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from modules.encoder import load_encoder, supcon
from modules.masked_hand import HandReconstruction, mask_hands, reconstruction_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--mt5-layers', type=int, default=2)
    p.add_argument('--weight', type=float, default=0.1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.steps < 1 or a.weight < 0:
        p.error('steps must be positive and weight nonnegative')
    if a.out.exists():
        p.error('output already exists; use a fresh experiment directory')
    torch.manual_seed(a.seed)
    with np.load(a.cache, allow_pickle=False) as f:
        required = ['body','left','right','face_all','valid','observed_left','observed_right','labels','clip_id','role']
        missing = set(required)-set(f.files)
        if missing:
            p.error('cache missing: '+str(sorted(missing)))
        data = {k: f[k].copy() for k in required}
    if data['valid'].dtype != bool or any(data['observed_'+h].dtype != bool for h in ('left','right')):
        p.error('valid/observed masks must be boolean, not inferred confidence')
    train = np.flatnonzero(data['role'] == 'train')
    val = np.flatnonzero(data['role'] == 'val')
    if len(train) < 2 or not len(val):
        p.error('need at least two train clips and separate validation clips')
    if set(data['clip_id'][train]) & set(data['clip_id'][val]):
        p.error('clip_id overlaps train and val; split original clips before augmentation')
    if not set(data['labels'][val]).issubset(set(data['labels'][train])):
        p.error('validation contains labels absent from training gallery')
    _, counts = np.unique(data['labels'][train], return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        p.error('pilot needs at least two labels with two training views each for SupCon')
    model = load_encoder(a.models, a.models/'encoder.pt', a.device)
    if not 0 <= a.mt5_layers <= len(model.mt5.encoder.block):
        p.error('invalid mt5 layer count')
    model.n_train_layers = a.mt5_layers
    head = HandReconstruction().to(a.device)
    groups = model.trainable_groups()
    groups.append(dict(params=list(head.parameters()), lr=5e-4))
    opt = torch.optim.AdamW(groups)
    parts = {k: torch.tensor(data[k], device=a.device) for k in ('body','left','right','face_all')}
    valid = torch.tensor(data['valid'], device=a.device)
    observed = {h: torch.tensor(data['observed_'+h], device=a.device) for h in ('left','right')}
    labels = torch.tensor(data['labels'], dtype=torch.long, device=a.device)
    ti = torch.tensor(train, device=a.device)
    vi = torch.tensor(val, device=a.device)
    history = []

    @torch.no_grad()
    def evaluate():
        model.eval()
        z = model(parts, valid)
        predicted = labels[ti][(z[vi] @ z[ti].T).argmax(1)]
        return float((predicted == labels[vi]).float().mean())

    before = evaluate()
    for step in range(a.steps):
        model.train(); head.train()
        x = {k:v[ti] for k,v in parts.items()}
        m = valid[ti]
        z = model(x, m)
        supervised = supcon(z, labels[ti])
        rec = z.sum()*0
        if a.weight:
            masked, target, selected = mask_hands(x, m, {k:v[ti] for k,v in observed.items()})
            rec = reconstruction_loss(head(model.frames(masked, m)), target, selected)
        loss = supervised + a.weight*rec
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([v for g in groups for v in g['params']], 1)
        opt.step()
        row = dict(step=step+1, loss=float(loss.detach()), supcon=float(supervised.detach()), reconstruction=float(rec.detach()))
        history.append(row); print(json.dumps(row), flush=True)
    after = evaluate()
    a.out.mkdir(parents=True)
    torch.save(model.state_dict(), a.out/'encoder.experimental.pt')
    torch.save(head.state_dict(), a.out/'reconstruction.experimental.pt')
    (a.out/'results.json').write_text(json.dumps(dict(kind='pilot_not_production_baseline', args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}, validation_r1_before=before, validation_r1_after=after, history=history), indent=2))

if __name__ == '__main__':
    main()
