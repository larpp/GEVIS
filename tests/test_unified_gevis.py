import ast,json,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import asdict
from pecvd.experiments.gevis_system import ready_gevis_runtime,run_gevis_experiment,SYSTEM

ROOT=Path(__file__).resolve().parents[1]


def optional_branch_name_leaks(cells):
    """Audit top-level optional RUN_* bindings, not merely Python syntax.

    This bounded notebook check does not interpret function bodies or arbitrary
    Python conditions; execution tests cover the actual reporting entry cells.
    """
    definitions={};loads=[]
    def walk(nodes,guards=frozenset()):
        for node in nodes:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                definitions.setdefault(node.name,[]).append(guards)
                continue
            if isinstance(node,ast.If):
                walk([ast.Expr(value=node.test)],guards)
                flags={n.id for n in ast.walk(node.test) if isinstance(n,ast.Name) and n.id.startswith('RUN_')}
                walk(node.body,guards|flags)
                walk(node.orelse,guards|{'NOT_'+name for name in flags})
                continue
            if isinstance(node,(ast.For,ast.While,ast.With,ast.Try)):
                for field in ('body','orelse','finalbody'):walk(getattr(node,field,[]),guards)
                for handler in getattr(node,'handlers',[]):walk(handler.body,guards)
                if isinstance(node,ast.For):walk([ast.Assign(targets=[node.target],value=node.iter)],guards)
                continue
            for n in ast.walk(node):
                if isinstance(n,ast.Name):
                    if isinstance(n.ctx,ast.Store):definitions.setdefault(n.id,[]).append(guards)
                    elif isinstance(n.ctx,ast.Load):loads.append((n.id,guards))
                elif isinstance(n,ast.Import):
                    for alias in n.names:definitions.setdefault(alias.asname or alias.name.split('.')[0],[]).append(guards)
                elif isinstance(n,ast.ImportFrom):
                    for alias in n.names:definitions.setdefault(alias.asname or alias.name,[]).append(guards)
    for source in cells:walk(ast.parse(source).body)
    return sorted({name for name,context in loads if name in definitions
        and all(definitions[name]) and not any(guard<=context for guard in definitions[name])})


class UnifiedGevisTests(unittest.TestCase):
    def test_no_optional_branch_only_names_escape_in_all_step_exp_notebooks(self):
        paths=list((ROOT/'notebooks').rglob('*.ipynb'))
        self.assertEqual(len(paths),11)
        from pecvd.experiments.pipeline_audit import NOTEBOOKS
        self.assertEqual({p.relative_to(ROOT/'notebooks').as_posix() for p in paths},set(NOTEBOOKS))
        self.assertEqual(len(NOTEBOOKS),len(set(NOTEBOOKS)))
        for path in paths:
            code=[''.join(c['source']) for c in json.loads(path.read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code']
            self.assertEqual(optional_branch_name_leaks(code),[],str(path))
        # Prove that the checker rejects the exact old failure, not a vacuous scan.
        self.assertEqual(optional_branch_name_leaks([
            'RUN_ANSWER_HEAD_TRAINING=False\nif RUN_ANSWER_HEAD_TRAINING:\n ANSWER_CHECKPOINT_PATH="optional.pt"',
            'print(ANSWER_CHECKPOINT_PATH)']),['ANSWER_CHECKPOINT_PATH'])

    def test_retired_threshold_fixture_completes_without_optional_answer_variables(self):
        import contextlib,io
        from tests.test_threshold_comparison import notebook_comparison_fixture
        n=json.loads((ROOT/'tests/fixtures/step3_legacy_diagnostics.json').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in n['cells'] if c.get('id')=='paired-threshold-comparison'))
        for enabled in (False,True):
            shown=[];env=dict(RUN_STRUCTURED_EVALUATION=enabled,RUN_ANSWER_HEAD_TRAINING=False,
                PIPELINE_ROOT=ROOT,PIPELINE_SETTINGS={'comparison_split':'validation'},
                RUN_THRESHOLD_SENSITIVITY=True,display=shown.append)
            with patch('importlib.reload',side_effect=lambda module:module), \
                 patch('pecvd.experiments.routing_audit.refresh_evaluation_modules'), \
                 patch('pecvd.experiments.threshold_comparison.run_threshold_comparison',side_effect=notebook_comparison_fixture) as compare, \
                 contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source,'STEP3:actual-final-cell','exec'),env)
            self.assertEqual(compare.call_count,int(enabled))
            self.assertNotIn('ANSWER_CHECKPOINT_PATH',env)
            self.assertEqual('THRESHOLD_COMPARISON' in env,enabled)

    def test_budget_adds_active_operating_point_without_changing_audit_quantiles(self):
        from pecvd.experiments.threshold_comparison import fit_comparison
        from pecvd.experiments.window_policy import WindowConfig
        from tests.test_position_rank_policy import records
        cfg=WindowConfig(gate_scope='position_rank_v4',operation_profile='core',nominal_window_alert_rate=.03)
        policies=fit_comparison(records('train'),records('validation'),lambda c:{},cfg)
        self.assertEqual(set(policies),{.90,.95,.99,.97})
        context=next(iter(policies[.97].payload['groups']['phase']['contexts'].values()))
        self.assertTrue(context['insufficient_tail_support'])
        self.assertFalse(context['control_budget_resolved'])
        self.assertAlmostEqual(context['requested_control_alert_budget'],.03)

    def test_hybrid_path_is_not_answer_head_or_source_threshold_fallback(self):
        runtime=SimpleNamespace(module=SimpleNamespace(DISCOVERY_POLICY_MODE='window_calibrated_v2',
            DISCOVERY_POLICY=SimpleNamespace(config=SimpleNamespace(gate_scope='position_rank_v4'))))
        report=SimpleNamespace(metadata={})
        with patch('pecvd.experiments.discovery_reporting.run_discovery_experiment',return_value=report) as run, \
             patch('pecvd.experiments.reporting.ready_answer_runtime',side_effect=AssertionError('No answer detector')):
            self.assertIs(run_gevis_experiment(ROOT,runtime=runtime,max_cases=8),report)
        run.assert_called_once_with(ROOT,runtime=runtime,max_cases=8)
        self.assertEqual(report.metadata['gevis_system'],SYSTEM)
        self.assertFalse(report.metadata['learned_answer_head_is_detector'])
        runtime.external_pairs={};runtime.module.DISCOVERY_POLICY=None
        with self.assertRaisesRegex(RuntimeError,'process-specific'):ready_gevis_runtime(ROOT,runtime)

    def test_all_default_notebooks_declare_one_system(self):
        paths=list((ROOT/'notebooks').rglob('*.ipynb'))
        self.assertEqual(len(paths),11)
        from pecvd.experiments.pipeline_audit import NOTEBOOKS
        self.assertEqual({p.relative_to(ROOT/'notebooks').as_posix() for p in paths},set(NOTEBOOKS))
        self.assertEqual(len(NOTEBOOKS),len(set(NOTEBOOKS)))
        for path in paths:
            cells=json.loads(path.read_text(encoding='utf-8'))['cells']
            code='\n'.join(''.join(c['source']) for c in cells if c['cell_type']=='code')
            self.assertIn('GEVIS_SYSTEM',code,str(path))
            for c in cells:
                if c['cell_type']=='code':ast.parse(''.join(c['source']))
        exp=lambda n: ''.join((ROOT/f'notebooks/{n}').read_text(encoding='utf-8'))
        self.assertIn('run_generalization',exp('research/CVD_synthetic_change_generalization.ipynb'))
        self.assertIn('run_gevis_predictions',exp('CVD_evaluate.ipynb'))
        self.assertIn('run_feature_controls',exp('research/CVD_feature_ablation.ipynb'))


if __name__=='__main__':unittest.main()
