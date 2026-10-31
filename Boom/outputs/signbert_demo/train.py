from pathlib import Path
import argparse,json,random
import numpy as np
import torch
from torch import nn
from model import HandTransformer,masked_input,reconstruction_loss
from model import CoordinateDecoder

ROOT=Path(__file__).parent

def load_data(manifest,frames=32):
    records=json.loads(Path(manifest).read_text())['records']
    records=[r for r in records if r['split']=='train']
    if len(records)<2: raise ValueError('No usable training dataset: successfully run prepare.py on at least two real clips first')
    labels=sorted({r['label'] for r in records})
    xs=[]; masks=[]; ys=[]
    for r in records:
        with np.load(r['features'],allow_pickle=False) as data:
            if str(data['sha256'])!=r['sha256']: raise ValueError('Feature/source hash mismatch')
            idx=np.linspace(0,len(data['xy'])-1,frames).round().astype(int)
            xs.append(data['xy'][idx]); masks.append(data['visible'][idx]); ys.append(labels.index(r['label']))
    return torch.tensor(np.stack(xs)),torch.tensor(np.stack(masks)),torch.tensor(ys),labels

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['pretrain','finetune'])
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manifest.json')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/pretrain.pt')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT/'runs')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1: p.error('epochs and batch-size must be positive')
    if args.stage=='finetune' and not args.checkpoint.is_file():
        raise FileNotFoundError('Run coordinate reconstruction pretraining first: '+str(args.checkpoint))
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    manifest_metadata=json.loads(args.manifest.read_text())
    x,mask,y,labels=load_data(args.manifest)
    model=HandTransformer(len(labels)); decoder=None
    if args.stage=='pretrain': decoder=CoordinateDecoder(model.config['dim'])
    else:
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if checkpoint.get('stage')!='pretrain' or checkpoint.get('decoder')!='coordinates': raise ValueError('Expected coordinate-pretrained checkpoint')
        compatible={k:v for k,v in checkpoint['model'].items() if not k.startswith('classifier.')}
        model.load_state_dict(compatible,strict=False)
    parameters=list(model.parameters())+(list(decoder.parameters()) if decoder else [])
    optimizer=torch.optim.Adam(parameters,lr=0.001,betas=(0.9,0.999),weight_decay=0.0001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)
    args.output.mkdir(parents=True,exist_ok=True)
    logs=[]
    for epoch in range(args.epochs):
        model.train(); total=0.; batches=0
        for indices in torch.randperm(len(x)).split(args.batch_size):
            xb,mb,yb=x[indices],mask[indices],y[indices]
            optimizer.zero_grad()
            if decoder:
                corrupted,observed,target,lmask=masked_input(xb,mb)
                predicted=decoder(model.encode(corrupted,observed))
                loss=reconstruction_loss(predicted,target,lmask)
            else: loss=nn.functional.cross_entropy(model(xb,mb),yb)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; stopping without saving this step')
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.0); optimizer.step()
            total+=float(loss.detach()); batches+=1
        scheduler.step(); logs.append(dict(epoch=epoch+1,loss=total/batches))
        print(json.dumps(logs[-1]),flush=True)
    # Save a self-contained coordinate model; no MANO dependency.
    result=dict(stage=args.stage,decoder='coordinates',method='SignBERT-inspired without MANO',model=model.state_dict(),config=model.config,labels=labels,
        evaluation='No independent validation/test data. Training-only demo.',source_manifest=str(args.manifest.resolve()),
        extractor_pipeline=manifest_metadata.get('pipeline_version','unspecified'),
        training_source_hashes=[r['sha256'] for r in manifest_metadata['records'] if r['split']=='train'])
    if decoder: result['decoder_head']=decoder.state_dict()
    else:
        model.eval()
        with torch.no_grad():
            scores=torch.cat([model(a,b) for a,b in zip(x.split(args.batch_size),mask.split(args.batch_size))])
            result['training_top1']=float((scores.argmax(-1)==y).float().mean())
            result['training_top5']=float((scores.topk(min(5,len(labels)),dim=-1).indices==y[:,None]).any(-1).float().mean())
    torch.save(result,args.output/(args.stage+'.pt'))
    (args.output/(args.stage+'_log.json')).write_text(json.dumps(logs,indent=2))
    print('Saved',args.output/(args.stage+'.pt'))

if __name__=='__main__': main()
