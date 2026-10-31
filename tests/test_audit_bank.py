"""Synthetic fixtures: bank filtering must preserve indices and held-out separation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('audit_bank',Path(__file__).resolve().parents[1]/'scripts/audit_bank.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'models').mkdir();(self.root/'artifacts/eval').mkdir(parents=True)
        self.cp=self.root/'models/concepts.json';a.write_json(self.cp,['one','two','three'])
        self.bp=self.root/'models/bank.npz';self.ep=self.root/'artifacts/eval/bank_eval.npz'
        self.bank=dict(Z=np.eye(3,dtype=np.float16),c=np.array([0,1,2],np.int32),clip=np.array(['train-a','train-b','heldout']),dataset=np.array(['x']*3),signer=np.array(['a','b','c']))
        np.savez(self.bp,**self.bank);np.savez(self.ep,**{k:v[:2] for k,v in self.bank.items()})
        self.review=self.root/'review.json'
        a.write_json(self.review,dict(bank_sha256=a.digest(self.bp),concepts_sha256=a.digest(self.cp),eval_bank_sha256=a.digest(self.ep),decisions=[dict(clip_id='train-a',action='exclude',reason='test fixture')]))
    def test_preserves_indices_and_no_heldout_leakage(self):
        before=a.digest(self.bp);a.build_review(self.root,self.review,self.root/'out')
        with np.load(self.root/'out/bank_eval.npz',allow_pickle=False) as b:
            self.assertEqual(b['clip'].tolist(),['train-b']);self.assertEqual(b['c'].tolist(),[1])
            np.testing.assert_array_equal(b['Z'],self.bank['Z'][1:2])
        self.assertEqual(before,a.digest(self.bp))
    def test_stale_review_rejected(self):
        self.cp.write_text('["different"]')
        with self.assertRaisesRegex(ValueError,'fingerprint'):a.build_review(self.root,self.review,self.root/'out')
    def test_pending_review_does_not_build(self):
        r=json.loads(self.review.read_text());r['decisions'][0]['action']='pending';a.write_json(self.review,r)
        with self.assertRaisesRegex(ValueError,'No reviewed exclusions'):a.build_review(self.root,self.review,self.root/'out')
    def test_unknown_id_rejected(self):
        r=json.loads(self.review.read_text());r['decisions'][0]['clip_id']='unknown';a.write_json(self.review,r)
        with self.assertRaisesRegex(ValueError,'Unknown'):a.build_review(self.root,self.review,self.root/'out')
    def test_no_overwrite(self):
        out=self.root/'out';out.mkdir();(out/'keep').write_text('keep')
        with self.assertRaisesRegex(ValueError,'empty experiment'):a.build_review(self.root,self.review,out)
    def test_ambiguous_join_not_used(self):
        self.assertEqual(a.unique_index([{'id':'x'},{'id':'x'}],lambda x:x['id']),{})
    def test_path_normalization(self):
        self.assertEqual(a.raw_key('keypoint_samples/word_level/x/a.mp4'),'raw_data/word_level/x/a.mp4')
    def test_conflicting_eval_ids_rejected(self):
        b={k:v[:2].copy() for k,v in self.bank.items()};b['c'][0]=2;np.savez(self.ep,**b)
        r=json.loads(self.review.read_text());r['eval_bank_sha256']=a.digest(self.ep);a.write_json(self.review,r)
        with self.assertRaisesRegex(ValueError,'indices disagree'):a.build_review(self.root,self.review,self.root/'out')

if __name__=='__main__':unittest.main()
