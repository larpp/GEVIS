"""Reused equipment Job IDs must not discard genuinely heldout source Runs."""
from types import SimpleNamespace
import copy
import unittest

import numpy as np

from pecvd.experiments import ald_preprocessing as prep, decision_training as d
from pecvd.experiments.peald_training import Prepared


def reused_jobs():
    partitions = dict(train=[f'run-{i}' for i in range(8)],
                      validation=['run-8', 'run-9'], test=['run-10', 'run-11'])
    meta = dict(sensor_columns=['pressure', 'ValveAct_1:1'], window_length=8,
                mean=np.zeros(2), std=np.ones(2), cadence_seconds=1.)
    pools = {}
    for split, runs in partitions.items():
        pools[split] = []
        for run in runs:
            i = int(run.split('-')[1])
            values = np.stack([np.sin(np.arange(8) / 2) + i * .001, np.arange(8) % 2], -1).astype(np.float32)
            pools[split].append(dict(run_id=run, lineage_id=f'job-{i % 4}', pair_id=run + '/window',
                current=values, sensor_columns=meta['sensor_columns'], split=split, step='phase',
                cohort=('recipe', '', 0, 0)))
    return Prepared({'phase': pools}, {'phase': meta}, {}, partitions,
                    dict(source_directory='data/mt_nit/data_complete', split_input_scope='data_complete_only'))


def support(prepared):
    runtime = SimpleNamespace(prepared=prepared, encoders={'phase': (None, prepared.metadata['phase'])})
    reference = d.fit_references(runtime, d.Config())
    cases = []
    for window in prepared.windows['phase']['validation']:
        cases.append(dict(d._common(window), current=window['current'], original_current=window['current'],
            centers=window['sensor_columns'], targets=np.zeros((2, 3), dtype=np.int64),
            kind='equal', injected=False, pair_id=window['pair_id']))
    return d.supported_cases(runtime, cases, reference)


class AldRunLineageTests(unittest.TestCase):
    def test_reproduces_empty_validation_and_repairs_without_recomputing_arrays(self):
        original = reused_jobs()
        old_cases, old_audit = support(original)
        self.assertEqual(old_cases, [])
        self.assertTrue(all(row['reason'] in ('reference_basis_lineage_not_independent',
                                             'heldout_lineage_overlaps_TRAIN') for row in old_audit))
        fixed = prep.canonicalize_run_lineages(original)
        cases, audit = support(fixed)
        self.assertEqual(len(cases), 2)
        self.assertTrue(all(row['supported'] for row in audit))
        self.assertEqual(fixed.identity['lineage_policy'], prep.LINEAGE_POLICY)
        self.assertEqual(original.windows['phase']['validation'][0]['lineage_id'], 'job-0')
        for split in ('train', 'validation', 'test'):
            for a, b in zip(original.windows['phase'][split], fixed.windows['phase'][split]):
                self.assertIs(a['current'], b['current'])
                self.assertEqual(b['lineage_id'], b['run_id'])

    def test_true_source_run_overlap_is_still_rejected(self):
        prepared = reused_jobs()
        prepared.split_runs['validation'].append(prepared.split_runs['train'][0])
        with self.assertRaisesRegex(ValueError, 'more than one partition'):
            prep.canonicalize_run_lineages(prepared)

    def test_windows_from_same_canonical_run_keep_same_lineage(self):
        prepared = reused_jobs()
        alias = copy.copy(prepared.windows['phase']['train'][0]); alias['pair_id'] += '/alias'
        prepared.windows['phase']['train'].append(alias)
        fixed = prep.canonicalize_run_lineages(prepared)
        self.assertEqual(fixed.windows['phase']['train'][0]['lineage_id'],
                         fixed.windows['phase']['train'][-1]['lineage_id'])


if __name__ == '__main__':
    unittest.main()
