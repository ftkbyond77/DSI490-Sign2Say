import gzip,json,tempfile,unittest
from types import SimpleNamespace
from webapp.api.request_cache import save_request
class TestCameraCache(unittest.TestCase):
    def test_optional_metadata(self):
        for metadata in [None, {'mode':'wide','capabilities':{},'resizeModeNoneSupported':None}]:
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                req=SimpleNamespace(frames=[],width=1280,height=720,vocab='conversation',alphabet='en',user_bank=[])
                if metadata is not None: req.camera_capture=metadata
                save_request(directory,req,{'segments':[]},None,.5)
                from pathlib import Path
                paths=list(Path(directory).glob('*.json.gz'));self.assertEqual(len(paths),1)
                with gzip.open(paths[0],'rt') as f: result=json.load(f)
                self.assertEqual(result.get('camera_capture'),metadata)
                if metadata is None:self.assertNotIn('camera_capture',result)

class TestRequestSchema(unittest.TestCase):
    def test_old_and_wide_requests(self):
        from webapp.api.app import TranslateRequest
        base=dict(frames=[],width=1280,height=720)
        self.assertIsNone(TranslateRequest(**base).model_dump().get('camera_capture'))
        metadata={'mode':'wide','capabilities':{},'resizeModeCapabilityPresent':False,'resizeModeNoneSupported':None}
        self.assertEqual(TranslateRequest(**base,camera_capture=metadata).model_dump().get('camera_capture'),metadata)
