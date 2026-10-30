"""Offline model/document contract check; never opens process data or weights."""
import ast
import hashlib
import json
from pathlib import Path


def constant(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing executable setting: ' + name)


def contract_fingerprint(root):
    root = Path(root)
    paths = ['src/pecvd/experiments/' + name for name in (
        'fusion_answers.py','peald_training.py','peald_readout.py',
        'peald_step_heads.py','peald_encoder_routing.py','external_test.py','inference_benchmark.py',
        'online_inference.py','evaluation_audit.py','fusion_metrics.py','fast_experiments.py','pipeline_audit.py',
        'pecvd_external.py','generated_test_scoring.py','fast_assessment.py')]
    paths += ['config/experiments/gevis_pipeline.json','config/experiments/peald_fast.json']
    sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}
    # Every canonical notebook is fingerprinted; inventory is shared with readiness.
    NOTEBOOKS = constant(root/'src/pecvd/experiments/pipeline_audit.py', 'NOTEBOOKS')
    for name in NOTEBOOKS:
        notebook = json.loads((root/'notebooks'/name).read_text(encoding='utf-8'))
        sources[name] = [(c.get('id'), ''.join(c.get('source', [])))
                         for c in notebook['cells'] if c['cell_type'] == 'code']
    return hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()


def check_docs(root):
    root = Path(root)
    errors = []
    try:
        settings = json.loads((root/'config/experiments/gevis_pipeline.json').read_text(encoding='utf-8'))
        recipe = constant(root/'src/pecvd/experiments/peald_encoder_routing.py', 'RECIPE')
        fingerprint = contract_fingerprint(root)
        notebook = json.loads((root/'notebooks/ALD_evaluate.ipynb').read_text(encoding='utf-8'))
        setup = next(''.join(c['source']) for c in notebook['cells'] if c.get('id') == 'setup')
        # Read the actual AST, not notebook output or a disconnected documentation setting.
        assignments = {t.id: ast.literal_eval(n.value) for n in ast.parse(setup).body
                       if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                       for t in n.targets if isinstance(t, ast.Name)}
        if settings.get('decision_system') != 'fusion_answers':
            errors.append('Current learned-answer contract changed; update and review documentation')
        if (assignments.get('ENCODER_SELECTION_MODE') != 'single'
                or assignments.get('SOURCE_ENCODER_STEP') != 'DEP.'):
            errors.append('PEALD evaluation default is not documented single DEP encoder mode')
        for name in ['README.md','_docs/README.md','_blueprint/experiment-plan.md','notebooks/README.md']:
            content = (root/name).read_text(encoding='utf-8')
            if '<!-- peald-encoder-routing: ' + recipe + ' -->' not in content:
                errors.append(name + ': encoder routing recipe documentation is stale')
            if '<!-- gevis-contract-sha256: ' + fingerprint + ' -->' not in content:
                errors.append(name + ': executable model/notebook changed without document synchronization')
            if not all(word in content for word in ['Validation','TRAIN','PECVD','PEALD']):
                errors.append(name + ': missing domain or fitting/selection boundary')
        plan = (root/'_blueprint/experiment-plan.md').read_text(encoding='utf-8')
        if 'Direction D0011' not in plan or 'decision_system=fusion_answers' not in plan:
            errors.append('Blueprint does not match active learned-answer contract')
        adapters = [root/'.agents/skills/project-harness/SKILL.md', root/'.claude/skills/project-harness/SKILL.md']
        if all(path.is_file() for path in adapters) and adapters[0].read_bytes() != adapters[1].read_bytes():
            errors.append('Project Harness adapters disagree; keep AGENTS.md as shared authority')
    except (OSError, ValueError, StopIteration, KeyError, SyntaxError) as error:
        errors.append('Document check prerequisite failed: ' + type(error).__name__)
    return errors


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    failures = check_docs(args.root)
    for failure in failures:
        print(failure)
    print('GEVIS document contract: ' + ('NEEDS_REVISION' if failures else 'PASS'))
    raise SystemExit(bool(failures))
