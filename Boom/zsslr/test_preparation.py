"""Conservative parsing and split alignment regression tests."""
import unittest,json
import numpy as np
from prepare_semantics import parse_entry,OUT,SPEC

class PreparationTests(unittest.TestCase):
    """Unknown handling and alias-group isolation."""
    def test_unknown_not_static(self) -> None:
        """No motion arrow is not evidence for a static gesture."""
        a,m,_=parse_entry({'movement':'ไม่มีลูกศรบอกวิถีในภาพ','num_hands':2})
        self.assertEqual(a[32],0); self.assertEqual(m[32],0)
        self.assertEqual(m[44],1); self.assertEqual(a[44],0); self.assertEqual(a[45],1)
    def test_no_label_leakage(self) -> None:
        """Word identity does not generate a guessed physical attribute."""
        a,_,_=parse_entry({'concept':'กำหมัด','handshape':'ระบุไม่ได้ชัดจากภาพ'})
        self.assertEqual(a.sum(),0)
    def test_artifacts(self) -> None:
        """All prototype rows have one split and no group crosses partitions."""
        rows=json.loads((OUT/'classes.json').read_text())
        a=np.load(OUT/'attributes_matrix.npy'); m=np.load(OUT/'attributes_known_mask.npy')
        self.assertEqual(a.shape,(473,53)); self.assertEqual(len(SPEC),53)
        self.assertTrue(np.all(a<=m))
        groups={}
        for row in rows: groups.setdefault(row['split_group'],set()).add(row['split'])
        self.assertTrue(all(len(s)==1 for s in groups.values()))
if __name__=='__main__': unittest.main()
