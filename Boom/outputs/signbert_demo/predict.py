from pathlib import Path
import argparse,json
import numpy as np
import torch
from model import HandTransformer

def main():
    p=argparse.ArgumentParser(); p.add_argument('video',type=Path)
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).parent/'runs/finetune.pt')
    args=p.parse_args()
    if not args.checkpoint.is_file(): p.error('No trained model. Complete MANO pretraining and fine-tuning first.')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('stage')!='finetune' or checkpoint.get('decoder')!='coordinates': p.error('Expected a fine-tuned no-MANO classifier checkpoint')
    from prepare import features
    from extract_keypoints import extract_video_keypoints
    model=HandTransformer(**checkpoint['config']); model.load_state_dict(checkpoint['model']); model.eval()
    x,mask,coverage=features(extract_video_keypoints(str(args.video)),frames=checkpoint['config']['frames'])
    with torch.no_grad(): probabilities=model(torch.from_numpy(x)[None],torch.from_numpy(mask)[None]).softmax(-1)[0]
    scores,indices=probabilities.topk(min(5,len(probabilities)))
    print(json.dumps(dict(evaluation=checkpoint['evaluation'],hand_coverage=coverage,
        results=[dict(word=checkpoint['labels'][i],softmax_score=float(s)) for s,i in zip(scores,indices.tolist())]),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
