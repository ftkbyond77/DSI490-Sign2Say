"""Contract tests for metadata and public API; real replay is separate."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from webapp.api.app import _public, TranslateRequest
from webapp.api.request_cache import save_request
from modules.runtime import variant_lut

class Contract(unittest.TestCase):
    def test_public_and_cache_keep_model_and_client_metadata(self):
        req=TranslateRequest(frames=[dict(t=0),dict(t=100)],width=640,height=480,perf_mode='balanced',client_fps=10)
        out=_public(dict(sentence='สวัสดี',segments=[],face={},timings_ms={},model='v8.1',model_version='v8.1',mirrored_to_right_hand=False))
        self.assertEqual(out['model_version'],'v8.1')
        for key in ('sentence','segments','face','timings_ms'):self.assertIn(key,out)
        with tempfile.TemporaryDirectory() as d:
            save_request(d,req,out,{'mirrored':False},.5)
            record=json.load(gzip.open(next(Path(d).glob('*.json.gz')),'rt'))
        self.assertEqual(record['result']['model_version'],'v8.1')
        self.assertEqual(record['perf_mode'],'balanced')
        self.assertEqual(record['client_fps'],10)
        self.assertIn('hand_canonicalization',record['result'])
    def test_variant_map_keeps_unmapped_words(self):
        self.assertEqual(variant_lut(['a','b','a2'],{'a2':'a','missing':'a'}).tolist(),[0,1,0])

if __name__=='__main__':unittest.main()
