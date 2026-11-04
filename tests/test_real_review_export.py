"""Independent replay oracle rejects corrupt exported research records."""
import copy
import unittest
import hashlib
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from keyloom_mania.generator_density import ExecutionLedger, execution_diagnostics
from scripts.export_real_review import verify_record
from scripts.review_real_song import packaged_chart


class RealReviewExportTests(unittest.TestCase):
    def fixture(self):
        rows=[[0,1,0,0],[100,0,1,1],[200,1,1,0],[300,0,0,1],[400,1,1,1],[450,0,0,0]]
        events=[[100,0],[300,0],[450,0]]
        ledger=ExecutionLedger()
        for row in rows:ledger.record(*row)
        desired=np.zeros((1,4,2));desired[0,0]=[3,3]
        trace=execution_diagnostics(desired,desired,None,[{'start_ms':0,'end_ms':2000,'quiet':False}],500,ledger,events,{})
        return {'committed_ledger':rows,'exported_events':events,'beats':trace['bars'][0]['beats'],
                'summary':trace['summary'],'style_bars':[{'eligible':True,'pass':False}],
                'outcome':{'adherence':{'eligible_bars':1,'passed_bars':0}}}

    def test_independent_oracle_recovers_net_zero_but_nonzero_changes(self):
        record=self.fixture()
        result=verify_record(record)
        self.assertEqual(result['exported_notes'],3)
        self.assertEqual(result['recalculated_mae'],[0,0])
        self.assertEqual(record['summary']['export_dropped_rows'],1)
        self.assertEqual(record['summary']['prepare_added_notes'],1)

    def test_oracle_rejects_stage_and_loss_corruption(self):
        for target in ('stage','loss','summary','duplicate','lane','style'):
            record=copy.deepcopy(self.fixture())
            if target=='stage':record['beats'][0]['stage_counts']['layout'][0]+=1
            if target=='loss':record['beats'][0]['losses']['export_removed_notes']+=1
            if target=='summary':record['summary']['notes_beat_mae']+=.1
            if target=='duplicate':record['committed_ledger'].append(record['committed_ledger'][0])
            if target=='lane':record['exported_events'].append(record['exported_events'][0])
            if target=='style':record['outcome']['adherence']['passed_bars']=1
            with self.subTest(target=target),self.assertRaises(AssertionError):
                verify_record(record)

    def test_recovers_cleaned_chart_from_unchanged_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package=root/'generated.osz'
            with zipfile.ZipFile(package,'w') as z:z.writestr('chart.osu',b'original chart')
            digest=hashlib.sha256(package.read_bytes()).hexdigest()
            target=packaged_chart(root,digest)
            self.assertEqual(target.read_bytes(),b'original chart')
            target.write_bytes(b'other chart')
            with self.assertRaises(ValueError):packaged_chart(root,digest)

    def test_recovery_rejects_changed_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package=root/'generated.osz'
            with zipfile.ZipFile(package,'w') as z:z.writestr('chart.osu',b'chart')
            with self.assertRaises(ValueError):packaged_chart(root,'0'*64)


if __name__=='__main__':unittest.main()
