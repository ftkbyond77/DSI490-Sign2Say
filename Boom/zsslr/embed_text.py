"""Extract final-layer raw CLS embeddings with pinned local multilingual BERT."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
from prepare_semantics import OUT

def main() -> None:
    """Freeze model, batch descriptions, save aligned float32 N,768 and provenance."""
    torch.set_num_threads(4)
    revision='3f076fdb1ab68d5b2880cb87a0886f315b8146f8'
    path=Path.home()/'.cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots'/revision
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModel.from_pretrained(path,local_files_only=True).eval()
    texts=json.loads((OUT/'descriptions.json').read_text()); result=[]
    lengths=[len(tokenizer.encode(t)) for t in texts]
    with torch.inference_mode():
        for start in range(0,len(texts),16):
            batch=tokenizer(texts[start:start+16],padding=True,truncation=True,max_length=256,return_tensors='pt')
            result.append(model(**batch).last_hidden_state[:,0,:].numpy())
            print(f'{min(start+16,len(texts))}/{len(texts)}',flush=True)
    matrix=np.concatenate(result).astype(np.float32)
    assert matrix.shape==(len(texts),768) and np.isfinite(matrix).all()
    np.save(OUT/'text_embeddings.npy',matrix)
    (OUT/'embedding_metadata.json').write_text(json.dumps({'model':'bert-base-multilingual-cased','revision':revision,'pooling':'last_hidden_state[:,0,:] raw CLS','shape':list(matrix.shape),'truncated_rows':sum(n>256 for n in lengths),'max_tokens':256,'source_sha256':hashlib.sha256((OUT/'descriptions.json').read_bytes()).hexdigest()},indent=2))
if __name__=='__main__': main()
