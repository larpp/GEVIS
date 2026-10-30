"""Read-only canonical notebook and local artifact readiness checks. No fitting."""
import ast
from contextlib import contextmanager
import hashlib,json,os
from pathlib import Path

NOTEBOOKS=('STEP_1_train_patchtst_encoder.ipynb','STEP_2_train_gevis_backbone.ipynb',
    'STEP_3_train_gevis_decision.ipynb','STEP_3.1_run_gevis_inference.ipynb','CVD_evaluate.ipynb',
    'ALD_evaluate.ipynb','CVD_predict_external_csv.ipynb','ALD_predict_external_csv.ipynb','research/CVD_question_sensitivity.ipynb',
    'research/CVD_synthetic_change_generalization.ipynb','research/CVD_feature_ablation.ipynb')


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def notebook_variant(root):
    settings_path=Path(root)/'config/experiments/gevis_pipeline.json'
    if settings_path.is_file():
        configured=json.loads(settings_path.read_text(encoding='utf-8')).get('model_variant')
        if configured is not None:return configured
    doc=json.loads((Path(root)/'notebooks'/NOTEBOOKS[0]).read_text(encoding='utf-8'))
    code=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='01-config'))
    tree=ast.parse(code)
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NOTEBOOK_MODEL_VARIANT' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('STEP 1 must declare NOTEBOOK_MODEL_VARIANT explicitly')


@contextmanager
def selected_profile(variant):
    previous=os.environ.get('GEVIS_MODEL_VARIANT')
    os.environ['GEVIS_MODEL_VARIANT']=variant
    try:yield
    finally:
        if previous is None:os.environ.pop('GEVIS_MODEL_VARIANT',None)
        else:os.environ['GEVIS_MODEL_VARIANT']=previous


def audit_pipeline(root,*,variant=None,verify_hashes=True,include_discovery=True):
    """Return readiness, never silently fall back to another trained variant.

    Explicit variant permits historical comparison inspection. Default follows
    the canonical STEP 1 source, not an old notebook's cached Python variables.
    Results contain paths/contracts only, no metrics, tensors or source rows.
    """
    from pecvd.model_variants import active_config,active_run_tag
    from pecvd.process_config import load_process_config
    from .process_inputs import candidate_backbone_path,configured_process_path
    root=Path(root).resolve();variant=variant or notebook_variant(root);rows=[]
    def add(stage,status,path,detail):
        path=Path(path)
        if not path.resolve().is_relative_to(root):
            display_path='outside-audit-root/'+path.name
            detail='Environment override outside audited root; '+detail
        else:display_path=path.relative_to(root).as_posix()
        rows.append({'stage':stage,'status':status,'variant':variant,
                     'path':display_path,'detail':detail})
    for name in NOTEBOOKS:
        path=root/'notebooks'/name
        if not path.is_file():add('notebook','MISSING',path,'canonical entry missing');continue
        try:
            doc=json.loads(path.read_text(encoding='utf-8'))
            for cell in doc['cells']:
                if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
            add('notebook','READY',path,'code cells parse; execution not inferred from saved outputs')
        except (ValueError,SyntaxError,KeyError):add('notebook','FAIL',path,'invalid notebook source')
    with selected_profile(variant):
        config=active_config(load_process_config(configured_process_path(root)))
        encoder=root/'artifacts'/config.artifacts.encoder
        backbone=candidate_backbone_path(root,config,active_run_tag())
    manifest_path=encoder/'encoder_manifest.json';manifest=None;backbone_ck=None
    if not manifest_path.is_file():add('STEP1','MISSING',manifest_path,'run PECVD STEP 1; no historical fallback')
    else:
        try:
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest['schema']!='gevis.encoder_manifest.rich_ssl.v1':raise ValueError()
            if manifest['artifact_version']!=config.artifacts.encoder:raise ValueError()
            if set(manifest['step_checkpoints'])!=set(config.step_specs):raise ValueError()
            for item in manifest['step_checkpoints'].values():
                path=(encoder/item['path']).resolve()
                if not path.is_relative_to(encoder.resolve()) or not path.is_file():raise ValueError()
                if verify_hashes and file_sha256(path)!=item['sha256']:raise ValueError()
            add('STEP1','READY',manifest_path,'manifest/config/encoder identities verified' if verify_hashes else 'paths checked; hashes NOT verified')
        except (ValueError,KeyError):
            manifest=None;add('STEP1','FAIL',manifest_path,'manifest/path/hash/config mismatch')
    if not backbone.is_file():add('STEP2','MISSING',backbone,'train matching PECVD STEP 2 after STEP 1')
    elif manifest is None:add('STEP2','BLOCKED',backbone,'STEP 1 identity is unavailable')
    else:
        import torch
        backbone_ck=torch.load(backbone,map_location='cpu',weights_only=False)
        valid=(backbone_ck.get('schema')=='gevis.backbone.generic.v10' and
               backbone_ck.get('step1_bridge',{}).get('manifest_sha256')==file_sha256(manifest_path))
        add('STEP2','READY' if valid else 'FAIL',backbone,'STEP 1 bridge hash verified' if valid else 'backbone/manifest mismatch')
        if not valid:backbone_ck=None
    settings_path=root/'config/experiments/gevis_pipeline.json'
    decision=json.loads(settings_path.read_text(encoding='utf-8')).get('decision_system') if settings_path.is_file() else 'hybrid'
    if decision=='fusion_answers':
        # The active STEP3 requires a trained readout, never retired numerical
        # policies. Verify its exact identity without loading E5 or process rows.
        from types import SimpleNamespace
        from . import fusion_answers as fa
        module=SimpleNamespace(ROOT=root,ENCODER_ROOT=encoder,BACKBONE_PATH=backbone,DEVICE='cpu')
        path=backbone.parent/'fusion_answers_<identity>.pt'
        if backbone_ck is None:
            add('STEP3','BLOCKED',path,'matching STEP 1/2 required for the trained Fusion readout')
        else:
            try:
                fa.attach(module)
                path=module.FUSION_ANSWER_PATH
                ready=getattr(module,'FUSION_ANSWER_HEAD',None) is not None
                add('STEP3','READY' if ready else 'MISSING',path,
                    'trained Fusion readout and model/cache/source identities verified' if ready
                    else 'run STEP3 answer-head training; no numerical fallback')
            except (ValueError,KeyError,FileNotFoundError,RuntimeError):
                add('STEP3','FAIL',getattr(module,'FUSION_ANSWER_PATH',path),
                    'Fusion readout identity/state/training evidence mismatch; no numerical fallback')
        return rows
    policy_path=backbone.parent/'reference_policy_v2.json'
    if not policy_path.is_file():add('STEP3','MISSING',policy_path,'prepare matching TRAIN reference criteria; do not retrain external data')
    elif backbone_ck is None:add('STEP3','BLOCKED',policy_path,'matching STEP 1/2 required')
    else:
        try:
            from .reference_policy_v2 import ReferenceConfig,identity_for,load_policy
            config_rule=ReferenceConfig(**json.loads(policy_path.read_text(encoding='utf-8'))['config'])
            load_policy(policy_path,identity_for(root,encoder,backbone,config_rule))
            add('STEP3','READY',policy_path,'model/manifest/cache/policy source identities verified')
        except (ValueError,KeyError,FileNotFoundError):add('STEP3','FAIL',policy_path,'criteria identity mismatch; prepare a matching candidate, do not silently reuse')
    if include_discovery and os.environ.get('GEVIS_DISCOVERY_MODE','window_calibrated_v2')=='window_calibrated_v2':
        from .discovery_calibration import policy_path as window_path,identity_for as window_identity,load_policy as load_window
        from .window_policy import WindowConfig
        path=window_path(backbone)
        if not path.is_file():add('STEP3-window','MISSING',path,'run STEP 3 window-evidence-build; frozen STEP 1/2 need no retraining')
        elif backbone_ck is None:add('STEP3-window','BLOCKED',path,'matching STEP 1/2 required')
        else:
            try:
                cfg=WindowConfig(**json.loads(path.read_text(encoding='utf-8'))['config'])
                policy=load_window(path,window_identity(encoder,backbone,cfg))
                ready=set(policy.payload['groups'])==set(config.step_specs) and all(g['supported'] for g in policy.payload['groups'].values())
                add('STEP3-window','READY' if ready else 'BLOCKED',path,'window calibration identity verified; inspect finite-sample warnings' if ready else 'insufficient independent calibration controls')
            except (ValueError,KeyError,FileNotFoundError):add('STEP3-window','FAIL',path,'window policy model/cache/code/config identity mismatch')
    return rows
