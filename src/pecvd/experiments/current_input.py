"""Current-only GEVIS: frozen PatchTST/text/Fusion + learned input projection/head.

No reference argument, reference bank, paired numeric_state, or test-original
input. Nominal originals are normal by the user's dataset premise. This is a
new learning contract; a paired head cannot be restored as a current-only head.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from . import fusion_answers as fa, peald_training as peald, normal_context as normal
from .reporting import ExperimentReport

SCHEMA = 'gevis.current_input_normal_profile_typed.v2'
TRAINING_RECIPE = 'normal_variation_positive_f1_v4'
HANDOFF = '.cache/external/peald/current_input_head.json'


@dataclass(frozen=True)
class Config:
    seed: int = 42
    epochs: int = 20
    patience: int = 5
    width: int = 96
    center_batch_size: int = 8
    learning_batch_size: int = 32
    cache_frozen_features: bool = True
    learning_rate: float = .001
    max_seconds: float = 600.
    normal_variation_windows_per_step: int = 4
    train_windows_per_step: int = 8
    validation_windows_per_step: int = 4
    test_windows_per_step: int = 4
    original_runs_normal: bool = True
    recipe_suffix: str = 'HD02'
    excluded_months: tuple = (4, 7)
    max_interval_seconds: float = 1.5
    normal_epochs: int = 10
    normal_max_seconds: float = 180.
    normal_batch_size: int = 4096
    normal_windows_per_run: int = 2
    normal_width: int = 64
    normal_learning_rate: float = .001

    def validate(self):
        if type(self.cache_frozen_features) is not bool:raise ValueError('cache_frozen_features must be Boolean')
        if type(self.learning_batch_size) is not int or self.learning_batch_size<2 or self.learning_batch_size>128 or self.learning_batch_size%2:
            raise ValueError('An even supervised sensor batch size in 2..128 is required')
        if not math.isfinite(self.max_interval_seconds) or self.max_interval_seconds<=0:raise ValueError('Finite positive native timestamp-gap limit required')
        if not isinstance(self.recipe_suffix,str) or not self.recipe_suffix:
            raise ValueError('One explicit supported recipe suffix required')
        if any(type(m) is not int or not 1<=m<=12 for m in self.excluded_months):
            raise ValueError('Excluded months must be calendar month integers')
        if self.original_runs_normal is not True:
            raise ValueError('Explicit nominal-original normality premise required')
        if any(type(v) is not int or v < 1 for v in (self.epochs, self.patience, self.width,
                self.center_batch_size, self.train_windows_per_step,
                self.validation_windows_per_step, self.test_windows_per_step,self.normal_epochs,self.normal_batch_size,
                self.normal_windows_per_run,self.normal_width)):
            raise ValueError('Positive integer current-only learning budgets required')
        if type(self.normal_variation_windows_per_step) is not int or self.normal_variation_windows_per_step<0:
            raise ValueError('Nonnegative normal variation budget required')
        if self.center_batch_size > 32 or not all(math.isfinite(v) and v > 0 for v in (self.learning_rate, self.max_seconds,self.normal_max_seconds,self.normal_learning_rate)):
            raise ValueError('Bounded center batch and finite positive learning budgets required')


def _json(value):
    def convert(v):
        if isinstance(v, np.ndarray):return v.tolist()
        if isinstance(v, np.generic):return v.item()
        if isinstance(v, Path):return str(v)
        raise TypeError(type(v).__name__)
    return json.dumps(value, sort_keys=True, default=convert)


def _sha(value):return hashlib.sha256(_json(value).encode()).hexdigest()


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode());digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_identity(runtime, steps=None):
    selected = set(runtime.encoders if steps is None else steps)
    import inspect
    from pecvd import gevis_backbone, gevis_fast
    from . import questions, peald_current_data
    code = {Path(m.__file__).name:peald.digest(m.__file__) for m in (gevis_backbone,gevis_fast,questions,peald_current_data)}
    import importlib.metadata
    versions={'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__}
    try:versions['transformers']=importlib.metadata.version('transformers')
    except importlib.metadata.PackageNotFoundError:versions['transformers']=None
    seen = {}; encoders = {}
    for step, (encoder, _) in runtime.encoders.items():
        if step not in selected:continue
        if id(encoder) not in seen:seen[id(encoder)] = _state_digest(encoder)
        encoders[step] = seen[id(encoder)]
        for cls in type(encoder).__mro__:
            try:source_file=inspect.getsourcefile(cls)
            except TypeError:source_file=None
            if source_file and Path(source_file).is_file():code[cls.__module__+'.'+cls.__qualname__]=peald.digest(source_file)
    return dict(fusion_sha256=_state_digest(runtime.module.GEVIS_MODEL), encoders=encoders,
        runtime_code_sha256=runtime.module.CURRENT_SOURCE_RECIPE_SHA256,code_sha256=code,versions=versions)


def _steps(runtime, steps=None):
    selected = list(runtime.encoders) if steps is None else list(steps)
    if not selected or len(selected) != len(set(selected)) or any(s not in runtime.encoders for s in selected):
        raise ValueError('Unique supported current-input Steps required')
    widths = {int(encoder.d_model) for s, (encoder, _) in runtime.encoders.items() if s in selected}
    if len(widths) != 1:raise ValueError('Shared current head requires one encoder latent width')
    return selected


def _source_tree(source):
    """Unwrap only the canonical interactive mode guard, not arbitrary branches.

    Programmatic legacy loaders still execute original paired cells with no
    STEP_EXECUTION_MODE. The current-only source bootstrap needs just their
    frozen definitions/load prefix, never the interactive current model loader.
    """
    import ast
    tree=ast.parse(source)
    head=[];body=list(tree.body)
    while body and isinstance(body[0],ast.ImportFrom) and body[0].module=='__future__':
        head.append(body.pop(0))
    expected=ast.parse("globals().get('STEP_EXECUTION_MODE', 'paired_legacy') == 'paired_legacy'",mode='eval').body
    if len(body)==1 and isinstance(body[0],ast.If) and ast.dump(body[0].test)==ast.dump(expected):
        if body[0].orelse:raise ValueError('Unexpected source bootstrap mode fallback')
        return ast.Module(body=head+body[0].body,type_ignores=tree.type_ignores)
    return tree


def load_frozen_source(root):
    """Load STEP1/2 model artifacts, without paired head attach or NPZ cache reads."""
    import ast
    import sys
    from types import ModuleType
    from pecvd.gevis_ui import notebook_runtime_sources
    from .pipeline_settings import configure_pipeline
    import os
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    root=Path(root).resolve()
    if Path(__file__).resolve() != root/'src/pecvd/experiments/current_input.py':
        raise ValueError('Current-input helper belongs to another project copy')
    configure_pipeline(root)
    sources=notebook_runtime_sources(root/'notebooks/STEP_3.1_run_gevis_inference.ipynb')
    types_tree=_source_tree(sources['03-types'])
    roots=0
    for node in types_tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROOT' for t in node.targets):
            node.value=ast.Name(id='_CURRENT_ROOT',ctx=ast.Load());roots+=1
    if roots!=1:raise ValueError('Unexpected frozen source bootstrap; review before loading')
    load_tree=_source_tree(sources['03-load']);prefix=[];found=False
    for node in load_tree.body:
        prefix.append(node)
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='load_runtime':
            found=True;break
    if not found:raise ValueError('Frozen source model loader not found')
    descriptor=[node for node in _source_tree(sources['03-function']).body
                if isinstance(node,ast.FunctionDef) and node.name=='numeric_token_descriptions']
    if len(descriptor)!=1:raise ValueError('Frozen descriptor helper not found')
    module=ModuleType('_gevis_current_input_source');module.__file__=str(root/'notebooks/STEP_3.1_run_gevis_inference.ipynb')
    module._CURRENT_ROOT=root;sys.modules[module.__name__]=module
    exec(compile(ast.fix_missing_locations(types_tree),module.__file__+':current-types','exec'),module.__dict__)
    prefix_tree=ast.Module(body=prefix,type_ignores=[])
    exec(compile(ast.fix_missing_locations(prefix_tree),module.__file__+':frozen-models-only','exec'),module.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=descriptor,type_ignores=[])),module.__file__+':descriptions','exec'),module.__dict__)
    module.CURRENT_SOURCE_RECIPE_SHA256=hashlib.sha256((ast.dump(types_tree)+ast.dump(prefix_tree)+ast.dump(descriptor[0])).encode()).hexdigest()
    module.DECISION_SYSTEM=fa.SYSTEM;module.FUSION_ANSWER_HEAD=None;module.FUSION_ANSWER_TRAINING=None
    return SimpleNamespace(module=module,encoders=module.STEP_ENCODERS)


def prepare(root,process=None,*,patch_length,config=Config(),preprocessing=None,delivery_package=None,max_memory_mb=4096):
    """All data_complete files; native Main/SubCycle policy, no merge CSV."""
    from . import peald_current_data as native
    from .ald_delivery_evaluation import discover_package
    config.validate();package=delivery_package or discover_package(root)
    prepared=native.prepare(root,config=config,patch_length=patch_length,
        max_window_length=(preprocessing.max_window_length if preprocessing else 64),delivery_package=package,
        sensor_columns=process.sensor_columns if process is not None else (),progress=print,max_memory_mb=max_memory_mb)
    prepared.identity['current_input_policy']=asdict(config)
    return prepared


def runtime_for(source,prepared,*,source_step):
    encoder,original=source.encoders[source_step]
    if not getattr(encoder,'supports_native_sensor_transfer',False) and not callable(getattr(encoder,'channel_independent_tokens',None)):
        raise ValueError('Native sensor-transfer encoder required')
    if not prepared.windows:raise ValueError('No supported native training windows; inspect preparation audit')
    # Source weights stay unchanged; reject incompatible native grids before encoding.
    hf_config=getattr(encoder,'hf_config',{})
    exact_context=hf_config.get('context_length') if isinstance(hf_config,dict) else None
    for meta in prepared.metadata.values():
        width=int(meta['window_length']);patch=int(encoder.patch_len)
        if width%patch or not 2<=width//patch<=64:
            raise ValueError('Prepared native window exceeds reviewed source patch capacity')
        if exact_context is not None and width!=int(exact_context):
            raise ValueError('Frozen source needs an exact context length; no implicit resampling/padding is allowed')
    encoders={s:(encoder,dict(meta)) for s,meta in prepared.metadata.items()}
    module=SimpleNamespace(**source.module.__dict__);module.STEP_ENCODERS=encoders
    module.FUSION_ANSWER_HEAD=None;module.FUSION_ANSWER_TRAINING=None
    return SimpleNamespace(module=module,encoders=encoders,source_runtime=source,source_step=source_step,prepared=prepared)


@torch.no_grad()
def encode_current(runtime, step, current, sensor_columns=None):
    """Exactly one current encoder call. No second waveform is constructed."""
    encoder, metadata = runtime.encoders[step]
    module = runtime.module; encoder = encoder.to(module.DEVICE).requires_grad_(False).eval()
    names=list(metadata['sensor_columns'] if sensor_columns is None else sensor_columns)
    if not names or len(names)!=len(set(names)) or any(name not in metadata['sensor_columns'] for name in names):raise ValueError('Known observed current sensors required')
    values = np.asarray(current, np.float32)
    if values.shape != (metadata['window_length'], len(names)) or not np.isfinite(values).all():
        raise ValueError('One complete finite current window is required')
    tensor = torch.as_tensor(np.array(values, copy=True)[None], device=module.DEVICE)
    token_method = getattr(encoder, 'channel_independent_tokens', None)
    if callable(token_method):
        tokens = token_method(tensor)
    else:
        if values.shape[1] != encoder.n_sensors and not getattr(encoder, 'supports_native_sensor_transfer', False):
            raise ValueError('Encoder has no native sensor-transfer capability')
        tokens = encoder.tokens(tensor)
    patches, count = encoder.patchify(tensor)
    differences = patches.diff(dim=-1)
    statistics = torch.stack([patches.mean(-1), patches.std(-1, unbiased=False),
        patches.amin(-1), patches.amax(-1), patches[..., 0], patches[..., -1],
        differences.mean(-1), differences.std(-1, unbiased=False)], -1)
    if tokens.shape[1] != len(names)*count:
        raise ValueError('Complete current sensor/patch grid required')
    result = torch.cat([tokens, statistics], -1).detach().cpu()
    if not torch.isfinite(result).all():raise ValueError('Nonfinite current features')
    return result


class CurrentHead(nn.Module):
    """Normal model freezes after stage1; projection/readout train in stage2."""
    def __init__(self, runtime, *, width=96):
        super().__init__()
        source = runtime.module.GEVIS_MODEL.requires_grad_(False).eval().to(runtime.module.DEVICE)
        # Do not register/source-copy frozen weights in the new checkpoint or optimizer.
        object.__setattr__(self, '_runtime', runtime)
        object.__setattr__(self, '_source', source)
        from collections import OrderedDict
        self._text_context = {}
        self._descriptor_vectors = OrderedDict()
        self._question_vectors = None
        self.text_cache_stats = dict(descriptor_hits=0,descriptor_misses=0,text_encoder_calls=0)
        if not hasattr(runtime,'normal_profile'):raise ValueError('Train/load the normal-only context model before the typed head')
        self.normal_profile=runtime.normal_profile.requires_grad_(False).eval()
        latent = next(iter(runtime.encoders.values()))[0].d_model
        hidden = int(source.type_emb.embedding_dim)
        self.projection = nn.Sequential(nn.Linear(latent+8, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.readout = fa.ContextReadout(4*hidden+2+normal.EVIDENCE_DIM, width)

    @torch.no_grad()
    def _frozen_descriptors(self, texts, device):
        # Exact-string reuse only. Vectors are detached, model-local, CPU-resident,
        # bounded (8192 entries) and never parameters/checkpoint state.
        pool=self._descriptor_vectors;local={};missing=[]
        for text in texts:
            if text in pool:
                local[text]=pool[text];pool.move_to_end(text)
                self.text_cache_stats['descriptor_hits']+=1
            else:missing.append(text)
        if missing:
            values=self._source.descriptor_norm(self._source._text(missing,device)).detach().cpu()
            self.text_cache_stats['text_encoder_calls']+=1
            self.text_cache_stats['descriptor_misses']+=len(missing)
            for text,value in zip(missing,values):
                # clone prevents one retained row from pinning the full batch storage.
                local[text]=value.clone();pool[text]=local[text]
                while len(pool)>8192:pool.popitem(last=False)
        return torch.stack([local[text] for text in texts]).to(device)

    @torch.no_grad()
    def _frozen_questions(self, tasks, device):
        if self._question_vectors is None:
            text=self._source._text([fa.QUESTIONS[k] for k in fa.KINDS],device)
            kind_ids=torch.arange(len(fa.KINDS),device=device)
            self._question_vectors=(text.detach(),(text+self._source.type_emb(kind_ids)).detach())
            self.text_cache_stats['text_encoder_calls']+=1
        text,query=self._question_vectors
        ids=torch.tensor([fa.KINDS.index(k) for _,k in tasks],device=device)
        return text.index_select(0,ids),query.index_select(0,ids)

    def current_keys(self, encoded):
        keys=self._source.numeric_key_norm(self.projection(encoded))
        return torch.cat([keys,encoded[...,-normal.FEATURES:]],-1)

    def forward(self, keys, step, centers, *, offset_samples=0, sensor_columns=None, recipe_key=None,return_features=False,timings=None,normal_evidence=None):
        from pecvd.gevis_backbone import attention_inputs
        from .questions import center_descriptions
        metadata = dict(self._runtime.encoders[step][1])
        names=list(metadata['sensor_columns'] if sensor_columns is None else sensor_columns)
        if not names or len(names)!=len(set(names)) or any(v not in metadata['sensor_columns'] for v in names):raise ValueError('Unknown observed sensor grid')
        metadata['sensor_columns']=names
        if not centers or len(set(centers)) != len(centers) or any(c not in names for c in centers):
            raise ValueError('Unique supported analysis centers required')
        if keys.shape[0] != 1 or keys.shape[1] % len(names):raise ValueError('One current sensor grid required')
        statistics=keys[...,-normal.FEATURES:]
        keys=keys[...,:-normal.FEATURES]
        patches = keys.shape[1]//len(names); source = self._source; device = keys.device
        tasks = [(c, k) for c in centers for k in fa.KINDS]
        cache_key = (step, patches, tuple(names), tuple(centers))
        if cache_key not in self._text_context:
            descriptions = self._runtime.module.numeric_token_descriptions(step, metadata, patches, 1)[0]
            contexts = [center_descriptions(descriptions, names, patches, center) for center, _ in tasks]
            flattened = [v for row in contexts for v in row]
            unique = list(dict.fromkeys(flattened)); lookup = {s:i for i,s in enumerate(unique)}
            with torch.no_grad():
                ids = torch.tensor([lookup[v] for v in flattened], device=device)
                descriptor_values = self._frozen_descriptors(unique,device)
                text,query = self._frozen_questions(tasks,device)
            if len(self._text_context)>=128:self._text_context.pop(next(iter(self._text_context)))
            self._text_context[cache_key] = (descriptor_values.detach(), ids, text.detach(), query.detach())
        descriptor_values, ids, text, query = self._text_context[cache_key]
        descriptors = descriptor_values.index_select(0, ids).reshape(len(tasks), keys.shape[1], -1)
        numeric = keys.expand(len(tasks), -1, -1)
        attention_query, attention_keys = attention_inputs(numeric, descriptors, text, query,
            source.fusion_gate, getattr(source, 'fusion_mode', 'legacy'))
        answer, attention = source.cross(attention_query[:, None], attention_keys, attention_keys, need_weights=True)
        answer = source.answer_norm(answer[:, 0]+query)
        grid = keys.reshape(1, len(names), patches, -1)[0]
        weights = attention[:, 0].reshape(len(tasks), len(names), patches)
        selected = torch.stack([grid[names.index(c)].mean(0) for c, _ in tasks])
        local = torch.stack([(grid[names.index(c)]*weights[i, names.index(c), :, None]).sum(0)/
            weights[i, names.index(c)].sum().clamp_min(1e-8) for i, (c, _) in enumerate(tasks)])
        # Elapsed native samples are known at arrival; never use final Step duration.
        cadence = float(metadata.get('cadence_seconds', 1.))
        coordinate = keys.new_tensor([[math.log1p(offset_samples*cadence),
            math.log1p(metadata['window_length']*cadence)]]).expand(len(tasks), -1)
        if normal_evidence is None:
            if timings is not None:
                from .online_inference import sync
                sync(self._runtime.module);normal_started=time.perf_counter()
            statistic_grid=statistics.reshape(len(names),patches,normal.FEATURES)
            center_grid=torch.stack([statistic_grid[names.index(center)] for center in centers])
            normal_values=self.normal_profile.evidence(center_grid,step,recipe_key,centers,offset_samples)
            if timings is not None:
                sync(self._runtime.module)
                timings['normal_profile_ms']+=1000*(time.perf_counter()-normal_started)
        else:
            if any(center not in normal_evidence for center in centers):raise ValueError('Incomplete per-request normal evidence')
            normal_values=torch.stack([normal_evidence[center] for center in centers])
            if normal_values.shape!=(len(centers),normal.EVIDENCE_DIM) or not torch.isfinite(normal_values).all():raise ValueError('Invalid normal evidence')
        features = torch.cat([answer, query, selected, local, coordinate,normal_values.repeat_interleave(3,dim=0)], -1)
        grouped={kind:features[i::3] for i,kind in enumerate(fa.KINDS)}
        return grouped if return_features else {kind:self.readout(grouped[kind],kind) for kind in fa.KINDS}


def cases(runtime, split, *, steps=None, config=Config()):
    config.validate(); selected = _steps(runtime, steps)
    if split not in ('train', 'validation', 'test'):raise ValueError('Known split required')
    partitions = runtime.prepared.split_runs
    if any(set(partitions[a]) & set(partitions[b]) for a,b in
            (('train','validation'),('train','test'),('validation','test'))):raise ValueError('Run partitions overlap')
    limit = getattr(config, split+'_windows_per_step')
    seed = config.seed+dict(train=0, validation=100003, test=200003)[split]
    rows = []; index = 0
    for step in selected:
        names = list(runtime.encoders[step][1]['sensor_columns'])
        allowed=runtime.encoders[step][1].get('allowed_recipe_keys')
        pool = [w for w in runtime.prepared.windows[step][split] if not allowed or w['cohort'][0] in allowed]
        if any(w['run_id'] not in partitions[split] or w.get('split',split) != split for w in pool):
            raise ValueError('Window split does not match its original Run')
        for window in peald._select(pool, limit, seed):
            names=list(window.get('sensor_columns',runtime.encoders[step][1]['sensor_columns']))
            base = np.array(window['current'], dtype=np.float32, copy=True)
            common = dict(step=step, split=split, run_id=window['run_id'],
                offset_samples=int(window['cohort'][-1]),recipe_key=window['cohort'][0], sensor_columns=names, original_id=window['pair_id'])
            rows.append(dict(common, current=base, centers=names, targets=np.zeros((len(names),3),np.int64),
                kind='equal', injected=False, pair_id=window['pair_id']+'/current-original'))
            sid = (index*17)%len(names)
            family = tuple(fa.TYPE_TARGET)[index%len(fa.TYPE_TARGET)]
            candidates = [v for v in fa.paired_variants(base, sid, index, seed) if v['kind']==family and v['mask'].any()]
            if candidates:
                variant = candidates[(index//len(fa.TYPE_TARGET))%len(candidates)]
                observed = np.asarray(variant['observed'], np.float32)
                mask = observed != base
                target = variant.get('sensor', sid)
                if mask[:, target].any():
                    ids = [target]+([(target+1)%len(names)] if len(names)>1 else [])
                    truth = np.zeros((len(ids),3),np.int64)
                    for i,sensor in enumerate(ids):
                        if mask[:,sensor].any():truth[i] = (1, fa.TYPE_TARGET[family], variant['level'])
                    rows.append(dict(common, current=observed, centers=[names[j] for j in ids], targets=truth,
                        kind=family, injected=True, pair_id=window['pair_id']+'/current-injected'))
            index += 1
    return rows



def normal_variation_cases(runtime, steps, base_cases, *, config=Config()):
    """Extra measured TRAIN normals; emphasize real switching/variation windows.

    No synthetic jitter, averaged traces, relabelled injections or held-out inputs.
    Within each native Step, round-robin distinct Runs after ranking each Run's
    actual windows by standardized first-difference energy. Raw arrays stay intact.
    """
    config.validate();selected=_steps(runtime,steps);rows=[];audit=[]
    used={(case['step'],case['original_id']) for case in base_cases}
    partitions=runtime.prepared.split_runs
    if any(set(partitions['train']) & set(partitions[k]) for k in ('validation','test')):
        raise ValueError('Normal variation Run partitions overlap')
    for step in selected:
        encoder,meta=runtime.encoders[step];allowed=meta.get('allowed_recipe_keys')
        by_run={};unsupported=0;available=0
        for window in runtime.prepared.windows[step]['train']:
            if window['run_id'] not in partitions['train'] or window.get('split','train')!='train' or window.get('injected',False):
                raise ValueError('Normal variation requires original TRAIN windows only')
            if (step,window['pair_id']) in used or (allowed and window['cohort'][0] not in allowed):continue
            names=list(window.get('sensor_columns',meta['sensor_columns']))
            base=np.asarray(window['current'],np.float32)
            if base.shape!=(meta['window_length'],len(names)) or not np.isfinite(base).all():
                raise ValueError('Complete finite normal TRAIN waveform required')
            available+=1
            if runtime.normal_profile.support(step,window['cohort'][0],names,int(window['cohort'][-1]),len(base)//encoder.patch_len):
                unsupported+=1;continue
            # Inputs are already standardized using TRAIN scalars. Squared changes
            # rank measured variation only; this score is never an anomaly label.
            roughness=float(np.square(np.diff(base.astype(np.float64),axis=0)).mean()) if len(base)>1 else 0.
            by_run.setdefault(window['run_id'],[]).append((roughness,window))
        for values in by_run.values():values.sort(key=lambda item:(-item[0],str(item[1]['pair_id'])))
        order=sorted(by_run,key=lambda run:(-by_run[run][0][0],str(run)))
        chosen=[];depth=0
        while len(chosen)<config.normal_variation_windows_per_step:
            level=[by_run[run][depth][1] for run in order if depth<len(by_run[run])]
            if not level:break
            chosen.extend(level[:config.normal_variation_windows_per_step-len(chosen)]);depth+=1
        for window in chosen:
            names=list(window.get('sensor_columns',meta['sensor_columns']))
            rows.append(dict(step=step,split='train',run_id=window['run_id'],
                offset_samples=int(window['cohort'][-1]),recipe_key=window['cohort'][0],sensor_columns=names,
                original_id=window['pair_id'],current=np.array(window['current'],np.float32,copy=True),
                centers=names,targets=np.zeros((len(names),3),np.int64),kind='normal_original_variation',
                injected=False,pair_id=window['pair_id']+'/current-normal-variation'))
        audit.append(dict(step=step,additional_candidates=available,unsupported_normal_windows=unsupported,
            selected_windows=len(chosen),selected_run_groups=len({w['run_id'] for w in chosen})))
    return rows,audit


def _negative_population(case):
    if case.get('kind')=='normal_original_variation':return 'normal_variation'
    return 'unchanged' if case['injected'] else 'normal_original'


def _feature_identity(runtime,examples,source):
    import ast
    tree=ast.parse(Path(__file__).read_text(encoding='utf8'))
    recipe=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='encode_current')
    digest=hashlib.sha256()
    for case in examples:
        digest.update(_json({k:v for k,v in case.items() if k not in ('current','targets','encoded')}).encode())
        for name in ('current','targets'):
            values=np.ascontiguousarray(case[name]);digest.update(name.encode())
            digest.update(str((values.dtype,values.shape)).encode());digest.update(values.tobytes())
    steps=sorted({case['step'] for case in examples})
    return dict(schema='gevis.current_frozen_features.v1',source=source,cases_sha256=digest.hexdigest(),
        encoder_recipe_sha256=hashlib.sha256(ast.dump(recipe).encode()).hexdigest(),
        metadata={step:runtime.encoders[step][1] for step in steps})


def _feature_content(identity,features):
    digest=hashlib.sha256(_json(identity).encode())
    for value in features:
        digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _encoded(runtime,examples,progress=None,*,cache_root=None,source=None):
    """Cache immutable encoder outputs independently of head/optimizer changes."""
    cache_path=None;identity=None;module_digest=peald.digest(__file__)
    if cache_root is not None:
        if source is None:source=source_identity(runtime,sorted({case['step'] for case in examples}))
        identity=_feature_identity(runtime,examples,source)
        cache_path=Path(cache_root)/(_sha(identity)[:24]+'.pt')
        if cache_path.is_file():
            before=peald.digest(cache_path);record=torch.load(cache_path,map_location='cpu',weights_only=False)
            features=record['features']
            if _sha(record.get('identity'))!=_sha(identity) or record.get('content_sha256')!=_feature_content(identity,features):
                raise ValueError('Frozen feature cache identity/content differs; no fallback to a different case set')
            if len(features)!=len(examples):raise ValueError('Incomplete frozen feature cache')
            for case,value in zip(examples,features):
                encoder,meta=runtime.encoders[case['step']]
                names=case.get('sensor_columns',meta['sensor_columns'])
                expected=(1,len(names)*(meta['window_length']//encoder.patch_len),encoder.d_model+8)
                if value.dtype!=torch.float32 or tuple(value.shape)!=expected or not torch.isfinite(value).all():
                    raise ValueError('Frozen feature grid differs from current cases')
            if peald.digest(cache_path)!=before:raise ValueError('Frozen feature cache changed while loading')
            if progress:progress(f'Reused current-only frozen features: {len(features)} cases; no encoder replay',flush=True)
            return [dict(case,encoded=value) for case,value in zip(examples,features)]
    encoded=[]
    for i,case in enumerate(examples):
        encoded.append(dict(case,encoded=encode_current(runtime,case['step'],case['current'],case.get('sensor_columns'))))
        if progress and (i==0 or (i+1)%25==0 or i+1==len(examples)):
            progress(f'Current-only frozen encoder: {i+1}/{len(examples)} cases',flush=True)
    if cache_path is not None:
        if module_digest!=peald.digest(__file__):raise ValueError('Encoding source changed during feature preparation')
        features=[case['encoded'] for case in encoded]
        record=dict(identity=identity,features=features,content_sha256=_feature_content(identity,features))
        _atomic_torch(record,cache_path)
        if progress:progress(f'Frozen features saved for same-case reuse: {cache_path}',flush=True)
    return encoded


class NormalSupportError(ValueError):
    def __init__(self,message,audit):
        super().__init__(message);self.audit=pd.DataFrame(audit)


def normal_supported_cases(runtime,examples):
    accepted=[];audit=[]
    for case in examples:
        encoder,meta=runtime.encoders[case['step']]
        names=case.get('sensor_columns',meta['sensor_columns'])
        issues=runtime.normal_profile.support(case['step'],case.get('recipe_key'),names,
            case['offset_samples'],meta['window_length']//encoder.patch_len)
        audit.append(dict(step=case['step'],split=case['split'],pair_id=case['pair_id'],supported=not issues,
            requested_targets=len(case['centers']),changed_targets=int(case['targets'][:,0].sum()),
            unsupported_sensors=len(issues),reason=issues[0]['reason'] if issues else None))
        if not issues:accepted.append(case)
    return accepted,audit


def target_catalog(examples):
    """TRAIN sensor targets, not waveform batches; all context sensors stay input."""
    positive=[];negative={};support={}
    for case_index,case in enumerate(examples):
        truth=np.asarray(case['targets'])
        if truth.shape!=(len(case['centers']),3):raise ValueError('One typed target per analysis center required')
        for center_index,(name,target) in enumerate(zip(case['centers'],truth)):
            if target[0] not in (0,1) or (target[0]==0 and np.any(target[1:]!=0)) or (target[0]==1 and np.any(target[1:]<=0)):
                raise ValueError('Inconsistent normal/changed typed target')
            key=(case['step'],name)
            row=support.setdefault(key,dict(step=case['step'],center_sensor=name,normal_original=0,normal_variation=0,unchanged=0,changed=0))
            if target[0]:
                positive.append((case_index,center_index));row['changed']+=1
            else:
                population=_negative_population(case)
                negative.setdefault((case['step'],name,population),[]).append((case_index,center_index))
                row[population]+=1
    if not positive or not negative:raise ValueError('TRAIN must contain positive and negative sensor targets')
    return positive,negative,list(support.values())


def balanced_batches(examples,*,batch_size,seed,epoch):
    """Half Yes/half No per optimizer batch; never join waveform input arrays.

    A complete epoch covers every positive target and every negative Step/sensor/
    population stratum. Original windows rotate within their stratum across epochs.
    Oversampling is explicit; it does not create independent positive examples.
    """
    if batch_size<2 or batch_size%2:raise ValueError('Even balanced target batch required')
    positive,negative,_=target_catalog(examples)
    rng=np.random.default_rng(seed+epoch*100003);keys=sorted(negative);rng.shuffle(keys)
    draws=max(len(positive),len(keys));pos=[]
    while len(pos)<draws:pos.extend(positive[i] for i in rng.permutation(len(positive)))
    pos=pos[:draws];neg=[]
    for i in range(draws):
        key=keys[i%len(keys)];values=negative[key]
        initial=int(_sha(key)[:8],16)%len(values)
        neg.append(values[(initial+epoch+i//len(keys))%len(values)])
    half=batch_size//2
    batches=[]
    for start in range(0,draws,half):
        batch=pos[start:start+half]+neg[start:start+half]
        rng.shuffle(batch);batches.append(batch)
    return batches,dict(unique_positive_targets=len(positive),negative_strata=len(keys),
        planned_positive_draws=draws,planned_negative_draws=draws,planned_batches=len(batches))


def _loss_weights(model,train,validation,device):
    labels=np.concatenate([case['targets'] for case in train]);positive=labels[labels[:,0]==1]
    weights={'BOOLEAN':None}
    for i,kind in enumerate(fa.KINDS):
        counts=np.bincount(labels[:,i],minlength=len(fa.ANSWER_CRITERIA[kind]))
        if counts[0]<1 or (counts>0).sum()<2:raise ValueError('TRAIN needs normal and changed answer classes')
        getattr(model.readout,'allowed_'+kind).copy_(torch.tensor(counts>0,device=device))
        if any(not np.isin(case['targets'][:,i],np.flatnonzero(counts)).all() for case in validation):
            raise ValueError('Validation answer absent in TRAIN; increase TRAIN case support')
        if kind=='BOOLEAN':continue  # Do not compensate twice after balanced sampling.
        conditional=np.bincount(positive[:,i],minlength=len(counts));active=np.flatnonzero(conditional[1:])+1
        values=np.zeros(len(counts),dtype=np.float32);values[0]=1.
        values[active]=len(positive)/(len(active)*conditional[active])
        weights[kind]=torch.tensor(values,device=device)
    return weights


def _target_loss(logits,targets,weights,repeats,denominator):
    """Fixed target count denominator, never a chunk's sum of class weights."""
    if denominator<1:raise ValueError('Complete optimizer target count required')
    return torch.stack([(F.cross_entropy(logits[k],targets[:,i],weight=weights[k],
        reduction='none')*repeats).sum() for i,k in enumerate(fa.KINDS)]).sum()/(3*denominator)


def _scores(truth, predictions):
    truth=np.asarray(truth);predictions=np.asarray(predictions)
    if not len(truth):raise ValueError('Nonempty validation predictions required')
    scores=[]
    for i,kind in enumerate(fa.KINDS):
        f1=[]
        for label in range(len(fa.ANSWER_CRITERIA[kind])):
            t=truth[:,i]==label;p=predictions[:,i]==label
            if t.any() or p.any():f1.append(2*int((t&p).sum())/max(1,int(t.sum()+p.sum())))
        scores.append(float(np.mean(f1)))
    y=truth[:,0];p=predictions[:,0]
    tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum());tp=int(((y==1)&(p==1)).sum())
    if not tn+fp or not tp+fn:raise ValueError('Validation needs nominal negatives and injected positives')
    recall=tp/(tp+fn);fpr=fp/(tn+fp)
    return dict(boolean_recall=recall,normal_false_positive_rate=fpr,
        boolean_precision=tp/max(1,tp+fp),boolean_positive_f1=2*tp/max(1,2*tp+fp+fn),
        balanced_accuracy=(recall+1-fpr)/2,tp=tp,fn=fn,tn=tn,fp=fp,
        positive_support=tp+fn,negative_support=tn+fp,
        macro_f1=dict(zip(fa.KINDS,scores)),joint_accuracy=float(np.all(truth==predictions,axis=1).mean()),
        always_no_joint_accuracy=float(np.all(truth==0,axis=1).mean()))


@torch.no_grad()
def _validation(model,examples,config,device):
    model.eval();truths=[];predictions=[];total_loss=0.;count=0;loss_by_boolean={0:[],1:[]}
    for case in examples:
        keys=model.current_keys(case['encoded'].to(device))
        for start in range(0,len(case['centers']),config.center_batch_size):
            centers=case['centers'][start:start+config.center_batch_size]
            target=torch.as_tensor(case['targets'][start:start+len(centers)],device=device)
            logits=model(keys,case['step'],centers,offset_samples=case['offset_samples'],sensor_columns=case.get('sensor_columns'),recipe_key=case.get('recipe_key'))
            per_target=torch.stack([F.cross_entropy(logits[k],target[:,i],reduction='none') for i,k in enumerate(fa.KINDS)]).mean(0)
            if not torch.isfinite(per_target).all():raise ValueError('Nonfinite Validation loss')
            values=per_target.cpu().tolist();total_loss+=sum(values);count+=len(values)
            for label,value in zip(target[:,0].cpu().tolist(),values):loss_by_boolean[label].append(value)
            truths.extend(target.cpu().tolist());predictions.extend(torch.stack([logits[k].argmax(-1) for k in fa.KINDS],-1).cpu().tolist())
    score=_scores(truths,predictions)
    score['balanced_validation_loss']=float(np.mean([np.mean(v) for v in loss_by_boolean.values()]))
    return score,total_loss/count


def _selection_key(score,val_loss):
    # Positive F1 uses FP and FN on the unchanged full Validation population.
    # No Test-derived threshold, weighted utility or majority-accuracy priority.
    return (score['boolean_positive_f1'],min(score['macro_f1'].values()),
            score['balanced_accuracy'],-score['normal_false_positive_rate'],-val_loss)


def _fit(runtime,train,validation,config,progress=None):
    from collections import Counter
    device=runtime.module.DEVICE;torch.manual_seed(config.seed)
    model=CurrentHead(runtime,width=config.width).to(device)
    weights=_loss_weights(model,train,validation,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    best=None;best_key=None;best_epoch=None;best_updates=0;best_coverage=None;history=[];stale=0;updates=0;started=time.perf_counter()
    positive,negative,support=target_catalog(train);seen_positive=set();seen_negative=set();seen_strata=set()
    stop_reason='epochs_completed'
    for epoch in range(config.epochs):
        if updates and time.perf_counter()-started>=config.max_seconds:stop_reason='wall_time_budget';break
        batches,plan=balanced_batches(train,batch_size=config.learning_batch_size,seed=config.seed,epoch=epoch)
        model.train();losses=[];positive_draws=0;negative_draws=0;epoch_started=time.perf_counter()
        for batch in batches:
            if losses and time.perf_counter()-started>=config.max_seconds:stop_reason='wall_time_budget';break
            optimizer.zero_grad(set_to_none=True);grouped={};batch_loss=0.
            for case_index,center_index in batch:
                grouped.setdefault(case_index,[]).append(center_index)
                case=train[case_index]
                if case['targets'][center_index,0]:
                    positive_draws+=1;seen_positive.add((case_index,center_index))
                else:
                    negative_draws+=1;seen_negative.add((case_index,center_index))
                    seen_strata.add((case['step'],case['centers'][center_index],_negative_population(case)))
            # One optimization step for the whole mixed target batch. Each waveform
            # is encoded/projected separately and keeps its full observed context.
            for case_index,indices in grouped.items():
                case=train[case_index];multiplicity=Counter(indices);ids=list(multiplicity)
                keys=model.current_keys(case['encoded'].to(device))
                chunks=math.ceil(len(ids)/config.center_batch_size)
                for chunk,start in enumerate(range(0,len(ids),config.center_batch_size)):
                    chosen=ids[start:start+config.center_batch_size]
                    centers=[case['centers'][j] for j in chosen]
                    target=torch.as_tensor(case['targets'][chosen],device=device)
                    repeats=torch.tensor([multiplicity[j] for j in chosen],dtype=torch.float32,device=device)
                    logits=model(keys,case['step'],centers,offset_samples=case['offset_samples'],sensor_columns=case.get('sensor_columns'),recipe_key=case.get('recipe_key'))
                    loss=_target_loss(logits,target,weights,repeats,len(batch))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite current-only loss; no model save')
                    loss.backward(retain_graph=chunk+1<chunks);batch_loss+=float(loss.detach())
            nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1;losses.append(batch_loss)
        if not losses:break
        optimization_seconds=time.perf_counter()-epoch_started;validation_started=time.perf_counter()
        score,val_loss=_validation(model,validation,config,device);validation_seconds=time.perf_counter()-validation_started
        key=_selection_key(score,val_loss)
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_loss=val_loss,
            optimization_seconds=optimization_seconds,validation_seconds=validation_seconds,
            text_cache_stats=dict(model.text_cache_stats),
            completed_batches=len(losses),epoch_completed=len(losses)==len(batches),
            positive_draws=positive_draws,negative_draws=negative_draws,
            cumulative_unique_positive_targets=len(seen_positive),cumulative_unique_negative_targets=len(seen_negative),
            cumulative_negative_strata=len(seen_strata),**plan,**score))
        if progress:progress(f'Current-only epoch {epoch+1}: recall={score["boolean_recall"]:.4f}, FPR={score["normal_false_positive_rate"]:.4f}, '
            f'TP/FN={score["tp"]}/{score["fn"]}, TN/FP={score["tn"]}/{score["fp"]}, '
            f'balanced accuracy={score["balanced_accuracy"]:.4f}; optimizer batches={len(losses)}/{len(batches)}',flush=True)
        if best_key is None or key>best_key:
            best_key=key;best_epoch=epoch+1;best_updates=updates
            best_coverage=dict(unique_positive_targets=len(seen_positive),unique_negative_targets=len(seen_negative),negative_strata=len(seen_strata))
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if time.perf_counter()-started>=config.max_seconds:stop_reason='wall_time_budget';break
        if stale>=config.patience:stop_reason='validation_patience';break
    if best is None or updates<1:raise RuntimeError('No trained current-only state; no random-head save')
    model.load_state_dict(best,strict=True)
    return model.requires_grad_(False).eval(),dict(optimizer_updates=best_updates,attempted_optimizer_updates=updates,selected_target_coverage=best_coverage,history=history,selected_epoch=best_epoch,
        fit_seconds=time.perf_counter()-started,stop_reason=stop_reason,training_recipe=TRAINING_RECIPE,
        train_runs=sorted({case['run_id'] for case in train}),validation_runs=sorted({case['run_id'] for case in validation}),
        target_support=support,unique_positive_targets=len(positive),negative_strata=len(negative),
        cumulative_unique_positive_targets=len(seen_positive),cumulative_negative_strata=len(seen_strata),
        selection='Validation positive BOOLEAN F1, worst family macro F1, balanced accuracy, lower normal FPR, loss',
        loss='balanced BOOLEAN target batches; unweighted Boolean CE; positive-conditional type/ordinal weights; fixed batch target denominator; three-family mean')


def payload_digest(payload):
    """Bind state and every saved scalar/contract table, including explicit loads."""
    digest=hashlib.sha256(_json({k:v for k,v in payload.items() if k not in ('state','content_sha256')}).encode())
    for name,value in sorted(payload['state'].items()):
        digest.update(name.encode());digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_payload(payload,path):
    if payload.get('schema')!=SCHEMA:raise ValueError('A paired/unknown model cannot substitute for a current-only head')
    if payload.get('content_sha256')!=payload_digest(payload):raise ValueError('Current-only state/metadata content changed')
    identity=payload['identity'];training=payload['training']
    spec=payload.get('normal_profile_spec',{});normal.validate_spec(spec)
    if normal.sha(spec)!=training.get('normal_profile_spec_sha256') or spec.get('fit_data_sha256')!=training.get('normal_profile_training',{}).get('fit_data_sha256'):
        raise ValueError('Normal model specification/fitting provenance differs from saved head')
    request=identity.get('normal_profile_request',{})
    if request.get('recipe')!=normal.RECIPE or request.get('code_sha256')!=peald.digest(normal.__file__):
        raise ValueError('Normal-context model code/recipe changed')
    if not set(spec['train_groups'])<=set(identity['preparation']['split_runs']['train']):
        raise ValueError('Normal profile contains non-TRAIN groups')
    if training.get('normal_profile_training',{}).get('optimizer_updates',0)<1:
        raise ValueError('Missing fitted normal-model evidence')
    if identity.get('training_recipe')!=TRAINING_RECIPE or training.get('training_recipe')!=TRAINING_RECIPE:
        raise ValueError('Stored head uses another optimization recipe; train the new STEP3 recipe without replacing old weights')
    if Path(path).parent.name!=_sha(identity)[:24] or identity['code_sha256']!=peald.digest(__file__):
        raise ValueError('Current-only source namespace/code differs')
    if _sha(payload['metadata'])!=_sha(identity['metadata']):raise ValueError('Current sensor order/normalization differs from training')
    if (training.get('input_mode')!='current_only' or training.get('reference_mode')!='none' or
            training.get('reference_contract')!=SCHEMA or training.get('optimizer_updates',0)<1 or
            set(training.get('trained_steps',[]))!=set(identity['selected_steps']) or
            set(payload['metadata'])!=set(identity['selected_steps'])):
        raise ValueError('Missing/inconsistent current-only training evidence')
    Config(**identity['config']).validate()
    for step,meta in payload['metadata'].items():
        names=meta['sensor_columns'];mean=np.asarray(meta['mean']);scale=np.asarray(meta['std'])
        if (not names or len(names)!=len(set(names)) or mean.shape!=(len(names),) or scale.shape!=mean.shape or
                not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any()):
            raise ValueError('Invalid saved normalization or sensor inventory')
    return identity,training


def _atomic_torch(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    import os,uuid
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    # Link publishes a complete file and refuses an existing final checkpoint.
    # Temporary is created in the same directory/volume; it is our own file only.
    os.link(temporary,path)
    temporary.unlink()


def train_or_load(root,runtime,*,steps=None,config=Config(),allow_training=False,progress=print):
    config.validate();selected=_steps(runtime,steps)
    if not runtime.prepared.identity:raise ValueError('Verified preparation identity required')
    for split in ('train','validation'):
        if any(not runtime.prepared.windows[s][split] for s in selected):raise ValueError('Each trained Step needs TRAIN and Validation originals')
    identity=dict(schema=SCHEMA,training_recipe=TRAINING_RECIPE,normal_profile_request=normal.request_identity(runtime,selected,config),domain='PEALD',config=asdict(config),preparation=runtime.prepared.identity,
        metadata={s:runtime.encoders[s][1] for s in selected},source=source_identity(runtime,selected),
        source_encoder_step=runtime.source_step,selected_steps=selected,code_sha256=peald.digest(__file__),
        injection_sha256=peald.digest(Path(fa.__file__).with_name('injection.py')),
        fusion_helper_sha256=peald.digest(fa.__file__),questions=fa.QUESTIONS,
        criteria={k:list(fa.ANSWER_CRITERIA[k]) for k in fa.KINDS})
    stamp=_sha(identity)[:24];path=Path(root)/'artifacts/gevis_current_input'/stamp/'answer_head.pt'
    if path.is_file():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        validate_payload(payload,path)
        if _sha(payload.get('identity',{}))!=_sha(identity):raise ValueError('Current-only checkpoint identity changed')
        runtime.normal_profile=normal.NormalProfile(payload['normal_profile_spec'])
        model=CurrentHead(runtime,width=config.width);model.load_state_dict(payload['state'],strict=True);training=payload['training']
        model.normal_profile.validate_fitted()
        if int(model.normal_profile.fit_updates)!=training['normal_profile_training']['optimizer_updates']:raise ValueError('Normal fitting counter differs from saved record')
    else:
        if not allow_training:raise FileNotFoundError('Current-only ALD head is not trained. Run STEP_3_train_gevis_decision with RUN_CURRENT_HEAD_TRAINING=True once; a paired head cannot substitute.')
        normal_started=time.perf_counter()
        runtime.normal_profile,normal_training=normal.fit_or_load(root,runtime,selected,config,
            identity['normal_profile_request'],_atomic_torch,allow_fit=allow_training,progress=progress)
        runtime.normal_profile_training=normal_training
        normal_prepare_seconds=time.perf_counter()-normal_started
        started=time.perf_counter()
        train_candidates=cases(runtime,'train',steps=selected,config=config)
        variation_candidates,variation_audit=normal_variation_cases(runtime,selected,train_candidates,config=config)
        validation_candidates=cases(runtime,'validation',steps=selected,config=config)
        train_cases,train_support=normal_supported_cases(runtime,train_candidates+variation_candidates)
        validation_cases,validation_support=normal_supported_cases(runtime,validation_candidates)
        if progress:
            for split,audit in (('TRAIN',train_support),('Validation',validation_support)):
                requested=sum(row['requested_targets'] for row in audit);supported=sum(row['requested_targets'] for row in audit if row['supported'])
                progress(f'Normal context support {split}: {supported}/{requested} targets; unobserved are not No',flush=True)
        runtime.normal_support_audit=train_support+validation_support
        if not train_cases or not validation_cases:raise NormalSupportError('No common normal-supported TRAIN/Validation targets; inspect support audit, never replace unobserved with No',runtime.normal_support_audit)
        train_labels=np.concatenate([c['targets'] for c in train_cases])
        validation_labels=np.concatenate([c['targets'] for c in validation_cases])
        try:positives,negatives,support=target_catalog(train_cases)
        except ValueError as error:raise NormalSupportError(str(error),runtime.normal_support_audit) from error
        if progress:
            n_positive=int(validation_labels[:,0].sum());n_negative=len(validation_labels)-n_positive
            progress(f'TRAIN unique changed targets={len(positives)}, negative Step/sensor/population strata={len(negatives)}; '
                f'balanced optimizer target batch={config.learning_batch_size}. All waveform context sensors stay input.',flush=True)
            progress(f'Validation supported population: changed={n_positive}, No={n_negative}; '
                f'always-No joint baseline={np.all(validation_labels==0,axis=1).mean():.4f}. '
                'Oversampling reuses TRAIN positives; it does not add independent examples.',flush=True)
        for i,kind in enumerate(fa.KINDS):
            support=set(train_labels[:,i].tolist())
            if len(support)<2 or not set(validation_labels[:,i].tolist())<=support:
                raise NormalSupportError('Missing TRAIN class support for '+kind+' after normal support checks',runtime.normal_support_audit)
        normal_validation=normal.describe_originals(runtime.normal_profile,runtime,validation_cases)
        cache_root=Path(root)/'.cache/current-input-features' if config.cache_frozen_features else None
        # Encoder features do not depend on the new normal model. Preserve the
        # original case-set cache key; apply supported-target filtering afterwards.
        train=_encoded(runtime,train_candidates,progress,cache_root=cache_root,source=identity['source'])
        # Additional normals have a separate cache; preserve old base-case keys.
        if variation_candidates:
            train+=_encoded(runtime,variation_candidates,progress,cache_root=cache_root,source=identity['source'])
        validation=_encoded(runtime,validation_candidates,progress,cache_root=cache_root,source=identity['source'])
        allowed_train={(case['step'],case['pair_id']) for case in train_cases}
        allowed_validation={(case['step'],case['pair_id']) for case in validation_cases}
        train=[case for case in train if (case['step'],case['pair_id']) in allowed_train]
        validation=[case for case in validation if (case['step'],case['pair_id']) in allowed_validation]
        model,training=_fit(runtime,train,validation,config,progress)
        training.update(normal_variation_audit=variation_audit,normal_profile_training=normal_training,normal_validation=normal_validation,normal_profile_spec_sha256=normal.sha(runtime.normal_profile.spec),
            normal_profile_prepare_seconds=normal_prepare_seconds,normal_support_audit=train_support+validation_support,
            encoding_and_fit_seconds=time.perf_counter()-started,input_mode='current_only',
            reference_mode='none',reference_contract=SCHEMA,target_domain='PEALD',trained_steps=selected,
            original_runs_normal=True,normality_premise='user-declared nominal original Runs',checkpoint_binding=SCHEMA)
        if source_identity(runtime,selected)!=identity['source'] or peald.digest(__file__)!=identity['code_sha256'] or peald.digest(normal.__file__)!=identity['normal_profile_request']['code_sha256']:
            raise ValueError('Frozen source/code changed while fitting; no compatible final save')
        process=getattr(runtime.prepared,'process',None)
        for relative,expected_digest in runtime.prepared.identity.get('source_files',{}).items():
            if peald.digest(Path(root)/relative)!=expected_digest:raise ValueError('Native training input changed; prepare again')
        process_columns=dict(process.source_columns) if process is not None else None
        payload=dict(schema=SCHEMA,identity=identity,normal_profile_spec=runtime.normal_profile.spec,state={k:v.detach().cpu() for k,v in model.state_dict().items()},
            training=training,metadata={s:runtime.encoders[s][1] for s in selected},
            cadence=runtime.prepared.tables['cadence'].to_dict('records'),source_columns=process_columns)
        payload['content_sha256']=payload_digest(payload)
        _atomic_torch(payload,path)
    if training.get('input_mode')!='current_only' or training.get('optimizer_updates',0)<1:raise ValueError('Not a trained current-only head')
    runtime.current_head=model.requires_grad_(False).eval().to(runtime.module.DEVICE)
    runtime.module.FUSION_ANSWER_HEAD=runtime.current_head;runtime.module.FUSION_ANSWER_TRAINING=training
    runtime.module.FUSION_ANSWER_PATH=path
    record=dict(schema=SCHEMA,checkpoint=path.relative_to(Path(root)).as_posix(),sha256=peald.digest(path))
    handoff=Path(root)/HANDOFF;handoff.parent.mkdir(parents=True,exist_ok=True)
    temporary=handoff.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2),encoding='utf-8');temporary.replace(handoff)
    return training


class Session:
    """A complete current window -> actual typed answers. No TRAIN data access."""
    def __init__(self,runtime,*,center_batch_size=8):
        training=runtime.module.FUSION_ANSWER_TRAINING or {}
        if training.get('input_mode')!='current_only' or training.get('optimizer_updates',0)<1:
            raise ValueError('A trained current-only head is required; paired heads are incompatible')
        if type(center_batch_size) is not int or not 1<=center_batch_size<=32:raise ValueError('Center batch must be 1..32')
        self.runtime=runtime;self.center_batch_size=center_batch_size
        self.metadata={s:dict(runtime.encoders[s][1]) for s in training['trained_steps']}

    @torch.no_grad()
    def process_window(self,*,current,step,run_id=None,recipe_key=None,centers=None,sensor_columns=None,offset_samples=0,standardized=False):
        from .online_inference import sync
        started=time.perf_counter();module=self.runtime.module;sync(module);started=time.perf_counter()
        stages=dict(normalization_ms=0.,reference_ms=0.,encoder_ms=0.,normal_profile_ms=0.,fusion_ms=0.,head_ms=0.)
        reply=dict(status='NOT_EVALUATED',reason='unsupported_step',answers=[],step=step,
            reference_count=0,reference_policy='none',input_mode='current_only')
        if step in self.metadata:
            meta=self.metadata[step]
            if meta.get('allowed_recipe_keys') and recipe_key not in meta['allowed_recipe_keys']:
                reply['reason']='unsupported_or_missing_recipe';stages['request_ms']=1000*(time.perf_counter()-started);reply['timings']=stages;return reply
            names=list(meta['sensor_columns'] if sensor_columns is None else sensor_columns)
            if not names or len(names)!=len(set(names)) or any(v not in meta['sensor_columns'] for v in names):raise ValueError('Known observed sensor grid required')
            chosen=names if centers is None else list(centers)
            if not chosen or len(chosen)!=len(set(chosen)) or any(s not in names for s in chosen):raise ValueError('Unknown center selection')
            if not isinstance(offset_samples,(int,np.integer)) or offset_samples<0:raise ValueError('Nonnegative known elapsed sample offset required')
            now=time.perf_counter();values=np.asarray(current,np.float32)
            if values.shape!=(meta['window_length'],len(names)) or not np.isfinite(values).all():raise ValueError('Finite complete current window required')
            if not standardized:
                ids=[meta['sensor_columns'].index(v) for v in names]
                values=((values-np.asarray(meta['mean'])[ids])/np.asarray(meta['std'])[ids]).astype(np.float32)
            issues=self.runtime.current_head.normal_profile.support(step,recipe_key,names,int(offset_samples),
                meta['window_length']//self.runtime.encoders[step][0].patch_len)
            if issues:
                reply.update(reason='unsupported_normal_context_or_elapsed_time',normal_support_issues=issues)
                stages['normalization_ms']=1000*(time.perf_counter()-now);stages['request_ms']=1000*(time.perf_counter()-started);reply['timings']=stages;return reply
            stages['normalization_ms']=1000*(time.perf_counter()-now)
            now=time.perf_counter();encoded=encode_current(self.runtime,step,values,names).to(module.DEVICE);sync(module)
            stages['encoder_ms']=1000*(time.perf_counter()-now)
            now=time.perf_counter();model=self.runtime.current_head.eval();keys=model.current_keys(encoded);sync(module)
            stages['fusion_ms']=1000*(time.perf_counter()-now)
            now=time.perf_counter()
            statistic_grid=encoded[0,:,-normal.FEATURES:].reshape(len(names),-1,normal.FEATURES)
            center_grid=torch.stack([statistic_grid[names.index(center)] for center in chosen])
            normal_values=model.normal_profile.evidence(center_grid,step,recipe_key,chosen,int(offset_samples))
            normal_evidence=dict(zip(chosen,normal_values));sync(module)
            stages['normal_profile_ms']=1000*(time.perf_counter()-now)
            answers=[]
            for start in range(0,len(chosen),self.center_batch_size):
                selected=chosen[start:start+self.center_batch_size]
                now=time.perf_counter()
                before_normal=stages['normal_profile_ms']
                features=model(keys,step,selected,offset_samples=int(offset_samples),sensor_columns=names,recipe_key=recipe_key,return_features=True,timings=stages,normal_evidence=normal_evidence);sync(module)
                stages['fusion_ms']+=max(0.,1000*(time.perf_counter()-now)-(stages['normal_profile_ms']-before_normal));now=time.perf_counter()
                logits={kind:model.readout(features[kind],kind) for kind in fa.KINDS}
                for kind in fa.KINDS:
                    probs=logits[kind].softmax(-1).cpu()
                    for center,prob in zip(selected,probs):
                        choice=int(prob.argmax());criteria=fa.ANSWER_CRITERIA[kind]
                        answers.append(dict(center_sensor=center,answer_type=kind,prediction=choice,
                            answer=criteria[choice],probabilities=dict(zip(criteria,map(float,prob)))))
                sync(module);stages['head_ms']+=1000*(time.perf_counter()-now)
            reply.update(status='OK',reason=None,answers=answers,selected_sensors=len(chosen))
        sync(module);stages['request_ms']=1000*(time.perf_counter()-started);reply['timings']=stages
        return reply


def load(root,*,checkpoint=None):
    """Restore frozen models and saved scalars only; never read TRAIN CSV/windows."""
    from . import external_test as external
    root=Path(root).resolve()
    if checkpoint is None:
        handoff=root/HANDOFF
        if not handoff.is_file():raise FileNotFoundError('No current-only ALD model. Train STEP_3_train_gevis_decision once with RUN_CURRENT_HEAD_TRAINING=True; existing paired weights remain preserved.')
        record=json.loads(handoff.read_text(encoding='utf-8'))
        if record.get('schema')!=SCHEMA:raise ValueError('This checkpoint predates normal-context learning. Train the new STEP3 once; old models remain preserved.')
        checkpoint=root/record['checkpoint']
        if peald.digest(checkpoint)!=record['sha256']:raise ValueError('Current-only checkpoint changed')
    path=Path(checkpoint);path=(path if path.is_absolute() else root/path).resolve()
    path.relative_to(root/'artifacts/gevis_current_input')
    checkpoint_before=peald.digest(path)
    payload=torch.load(path,map_location='cpu',weights_only=False);identity,training=validate_payload(payload,path)
    source=load_frozen_source(root)
    step=identity['source_encoder_step'];encoder,source_meta=source.encoders[step]
    metadata=payload['metadata'];encoders={s:(encoder,dict(m)) for s,m in metadata.items()}
    module=SimpleNamespace(**source.module.__dict__);module.STEP_ENCODERS=encoders
    runtime=SimpleNamespace(module=module,encoders=encoders,source_step=step,source_runtime=source,
        prepared=SimpleNamespace(metadata=metadata,tables={'cadence':pd.DataFrame(payload['cadence'])},
            split_runs=identity['preparation']['split_runs'],identity=identity['preparation']))
    if source_identity(runtime)!=identity['source']:raise ValueError('Frozen encoder/Fusion source weights differ')
    runtime.normal_profile=normal.NormalProfile(payload['normal_profile_spec'])
    model=CurrentHead(runtime,width=identity['config']['width']);model.load_state_dict(payload['state'],strict=True)
    model.normal_profile.validate_fitted()
    if int(model.normal_profile.fit_updates)!=training['normal_profile_training']['optimizer_updates']:raise ValueError('Normal fitting counter differs from saved record')
    runtime.current_head=model.requires_grad_(False).eval().to(module.DEVICE)
    module.FUSION_ANSWER_HEAD=runtime.current_head;module.FUSION_ANSWER_TRAINING=training;module.FUSION_ANSWER_PATH=path
    runtime.config=Config(**identity['config'])
    if peald.digest(path)!=checkpoint_before:raise ValueError('Current-only checkpoint changed while loading')
    return external.InputContract(runtime,SimpleNamespace(source_columns=payload['source_columns']))


def evaluate(runtime,*,steps=None,config=Config()):
    from . import fast_experiments as fast
    trained=runtime.module.FUSION_ANSWER_TRAINING['trained_steps']
    selected=_steps(runtime,trained if steps is None else steps)
    if not set(selected)<=set(trained):raise ValueError('Requested Step has no current-only head support')
    session=Session(runtime,center_batch_size=config.center_batch_size);rows=[];counts=[]
    examples=cases(runtime,'test',steps=selected,config=config)
    for step in selected:
        local=[case for case in examples if case['step']==step]
        if not local:counts.append(dict(step=step,status='NOT_EVALUATED',reason='no_native_Test_windows'))
        for case in local:
            reply=session.process_window(current=case['current'],step=step,centers=case['centers'],
                run_id=case['run_id'],recipe_key=case['recipe_key'],sensor_columns=case.get('sensor_columns'),offset_samples=case['offset_samples'],standardized=True)
            if reply['status']!='OK':
                counts.append(dict(step=step,pair_id=case['pair_id'],status='NOT_EVALUATED',reason=reply['reason']));continue
            for answer in reply['answers']:
                index=case['centers'].index(answer['center_sensor']);family=fa.KINDS.index(answer['answer_type'])
                label=int(case['targets'][index,family])
                rows.append(dict(answer,label=label,scored=True,step=step,run_id=case['run_id'],pair_id=case['pair_id'],
                    injected=case['injected'],center_is_perturbed=bool(case['targets'][index,0]),split='test'))
    if not rows:
        return ExperimentReport('Current-only Test NOT_EVALUATED',pd.DataFrame(),
            diagnostics=dict(unobserved_test_steps=pd.DataFrame(counts),boolean_populations=pd.DataFrame()))
    report=fast._report('Current-only nominal/injected held-out Test',rows,all_answers=rows,
        metadata=dict(input_mode='current_only',reference_policy='none',model_schema=SCHEMA,normal_model_recipe=normal.RECIPE,training_recipe=TRAINING_RECIPE,nominal_original_normality_assumed=True))
    b=report.details.loc[report.details.answer_type.eq('BOOLEAN')].copy()
    b['population']=np.where(~b.injected,'nominal_original',np.where(b.center_is_perturbed,'injected_sensor','unchanged_sensor'))
    report.diagnostics['boolean_populations']=pd.DataFrame([dict(population=name,questions=len(g),
        true_negatives=int((g.label.eq(0)&g.prediction.eq(0)).sum()),false_positives=int((g.label.eq(0)&g.prediction.eq(1)).sum()),
        false_negatives=int((g.label.eq(1)&g.prediction.eq(0)).sum()),true_positives=int((g.label.eq(1)&g.prediction.eq(1)).sum()))
        for name,g in b.groupby('population',sort=False)])
    report.diagnostics['unobserved_test_steps']=pd.DataFrame(counts)
    return report


class StreamBuffer:
    """Bounded live receiver. push() returns each completed-window model reply.

    timestamp_seconds is a numeric arrival measurement timestamp, not an inferred
    final Step position. Context switches/gaps drop pending rows explicitly.
    No previous Run waveforms, labels or unbounded sample history are retained.
    """
    def __init__(self, session, *, on_result=None):
        from collections import deque
        self.session=session;self.on_result=on_result;self._rows=[];self._indices=[]
        self._context=None;self._origin=None;self._previous=None;self._seen=0
        self._audit=deque(maxlen=256)
        table=session.runtime.prepared.tables['cadence']
        self._tolerance=dict(zip(table.step,table.tolerance_seconds))

    def _drop(self,reason):
        if self._rows:
            self._audit.append(dict(step=self._context[1],source_row_start=self._indices[0],
                source_row_end=self._indices[-1]+1,rows=len(self._rows),reason=reason))
        self._rows=[];self._indices=[]

    def drain_audit(self):
        rows=list(self._audit);self._audit.clear();return rows

    def finish(self):
        self._drop('incomplete_window_at_input_end');self._context=None;self._origin=None;self._previous=None
        return self.drain_audit()

    def push(self,*,run_id,step,timestamp_seconds,values,recipe,row_index=None):
        index=self._seen if row_index is None else int(row_index);self._seen+=1
        if not str(run_id).strip() or not isinstance(step,str) or not step.strip() or not str(recipe).strip():
            self._drop('missing_process_context');self._context=None;self._previous=None;self._origin=None
            self._audit.append(dict(step=step,source_row_start=index,source_row_end=index+1,rows=1,reason='missing_process_context'))
            return dict(status='NOT_EVALUATED',reason='missing_process_context',answers=[])
        recipe_key=peald._hash(str(recipe).strip())
        context=(str(run_id),step,recipe_key)
        if context!=self._context:
            self._drop('incomplete_window_at_context_boundary');self._context=context;self._origin=None;self._previous=None
        meta=self.session.metadata.get(step)
        if meta is None or (meta.get('allowed_recipe_keys') and recipe_key not in meta['allowed_recipe_keys']):
            self._audit.append(dict(step=step,source_row_start=index,source_row_end=index+1,rows=1,reason='unsupported_step_or_recipe'))
            return dict(status='NOT_EVALUATED',reason='unsupported_step_or_recipe',answers=[])
        current=np.array(values,dtype=np.float32,copy=True)
        timestamp=float(timestamp_seconds)
        if current.shape!=(len(meta['sensor_columns']),) or not np.isfinite(current).all() or not math.isfinite(timestamp):
            self._drop('invalid_row_breaks_window');self._previous=None
            self._audit.append(dict(step=step,source_row_start=index,source_row_end=index+1,rows=1,reason='nonfinite_sensor_or_timestamp'))
            return dict(status='NOT_EVALUATED',reason='invalid_row',answers=[])
        if self._origin is None:self._origin=timestamp
        cadence=float(meta['cadence_seconds']);tolerance=float(self._tolerance[step])
        if self._previous is not None and abs(timestamp-self._previous-cadence)>tolerance:
            self._drop('timestamp_gap_breaks_window')
            if timestamp<=self._previous:
                self._audit.append(dict(step=step,source_row_start=index,source_row_end=index+1,rows=1,reason='nonmonotonic_timestamp'))
                return dict(status='NOT_EVALUATED',reason='nonmonotonic_timestamp',answers=[])
        self._previous=timestamp;self._rows.append(current);self._indices.append(index)
        if len(self._rows)==1:self._window_offset=max(0,int(round((timestamp-self._origin)/cadence)))
        if len(self._rows)<meta['window_length']:
            return dict(status='PENDING',samples=len(self._rows),required_samples=meta['window_length'],answers=[])
        matrix=np.stack(self._rows);start=self._indices[0];end=self._indices[-1]+1
        result=self.session.process_window(current=matrix,step=step,run_id=run_id,
            recipe_key=recipe_key,offset_samples=self._window_offset)
        result.update(row_start=start,row_end=end,run_id=run_id,offset_samples=self._window_offset)
        self._rows=[];self._indices=[]
        if self.on_result is not None:self.on_result(result)
        return result


def read_frame(path,contract):
    """Require native context columns; preserve missing sensor fields as unobserved."""
    from . import peald_current_data as native
    header=set(pd.read_csv(path,nrows=0).columns)
    required=set(native.COLUMNS.values())
    if not required<=header:raise ValueError('Native PEALD context columns missing from incoming CSV')
    sensors=list(dict.fromkeys(s for _,meta in contract.runtime.encoders.values() for s in meta['sensor_columns'] if s in header))
    frame=pd.read_csv(path,usecols=sorted(required)+sensors,dtype={s:'string' for s in required},low_memory=False)
    for sensor in sensors:frame[sensor]=pd.to_numeric(frame[sensor],errors='coerce')
    if frame.empty:raise ValueError('Empty incoming CSV')
    return frame


def _replay_frame(contract,frame,boundaries,*,center_batch_size=8,progress=None,on_result=None):
    """Same native Receiver as live input; predictions precede evaluator labels."""
    from . import external_test as external,ald_delivery_evaluation as delivery,online_inference as online
    from . import peald_current_data as native
    runtime=contract.runtime;model_before=external._fingerprint(runtime)
    session=Session(runtime,center_batch_size=center_batch_size)
    started=time.perf_counter()
    stamps=pd.to_datetime(frame[native.COLUMNS['order']],format='mixed',errors='coerce')
    seconds=np.where(stamps.notna(),stamps.astype('int64').to_numpy()/1e9,np.nan)
    preparation_ms=1000*(time.perf_counter()-started)
    rows=[];requests=[];exclusions=[];coverage=[];provisional=[]
    for order,boundary in enumerate(boundaries):
        rid=boundary['run_id'];start=boundary['start'];end=boundary['end'];observed=np.zeros(end-start,bool)
        receiver=native.Receiver(session,on_result=on_result)
        local=[];local_requests=[];answered=0
        def consume(replies):
            nonlocal answered
            for result in replies:
                a=result['row_start'];b=result['row_end'];step=result['step'];request_id=f'{rid}/{step}/{a}:{b}'
                local_requests.append(dict(case_id=rid,request_id=request_id,step=step,row_start=a,row_end=b,
                    status=result['status'],reason=result['reason'],answers=len(result['answers']),
                    selected_sensors=result.get('selected_sensors',0),reference_count=0,**result['timings']))
                if result['status']!='OK':
                    exclusions.append(dict(case_id=rid,step=step,source_row_start=a,source_row_end=b,
                        rows=b-a,reason=result['reason'],unsupported_normal_sensors=len(result.get('normal_support_issues',[]))))
                if result['status']=='OK':
                    answered+=1;observed[a-start:b-start]=True
                    local.extend(dict(answer,case_id=rid,run_id=rid,source_group=result['run_id'],step=step,
                        row_start=a,row_end=b,pair_id=request_id) for answer in result['answers'])
                    if progress and (answered==1 or answered%50==0):
                        progress(f'Current-only Run {order+1}/{len(boundaries)}: {answered} windows answered',flush=True)
        part=frame.iloc[start:end]
        source_ids=part[native.COLUMNS['run']].astype('string').str.strip()
        identities=source_ids.dropna().unique()
        if source_ids.isna().any() or source_ids.eq('').any() or len(identities)!=1 or not identities[0]:
            exclusions.append(dict(case_id=rid,step=None,source_row_start=start,source_row_end=end,rows=end-start,reason='ambiguous_delivery_Run_identity'))
            coverage.append(dict(case_id=rid,run_order=order,input_start=start,input_end=end,input_rows=end-start,available_windows=0,answered_windows=0,sensor_windows=0,yes_sensor_windows=0,observed_prediction=np.nan,run_prediction=np.nan,scope_status='UNSCORED',whole_run_negative_certified=False,input_row_coverage=0.,overlap_status='not_certified_independent',preparation_ms=0.))
            continue
        for offset,values in enumerate(part.itertuples(index=False,name=None)):
            record=dict(zip(part.columns,values));record['_seconds']=float(seconds[start+offset])
            consume(receiver.push(record,row_index=start+offset))
            exclusions.extend(dict(item,case_id=rid) for item in receiver.drain_audit())
        consume(receiver.finish());exclusions.extend(dict(item,case_id=rid) for item in receiver.drain_audit())
        if receiver.invalid_run:
            provisional.extend(local);local=[];observed[:]=False;answered=0
        for request in local_requests:request['retained_after_Run_quality_check']=not receiver.invalid_run
        requests.extend(local_requests);rows.extend(local)
        boolean=[v for v in local if v['answer_type']=='BOOLEAN'];yes=sum(v['prediction']==1 for v in boolean)
        coverage.append(dict(case_id=rid,run_order=order,input_start=start,input_end=end,input_rows=end-start,
            available_windows=len(local_requests),answered_windows=answered,sensor_windows=len(boolean),yes_sensor_windows=yes,
            observed_prediction=int(yes>0) if boolean else np.nan,run_prediction=int(yes>0) if boolean else np.nan,
            scope_status='INVALID_MAIN_TIMESTAMP' if receiver.invalid_run else ('COMPLETE_SUPPORTED_ROWS' if observed.all() else ('PARTIAL_SUPPORTED_ROWS' if boolean else 'UNSCORED')),
            whole_run_negative_certified=False,input_row_coverage=float(observed.mean()),
            overlap_status='not_certified_independent_development_delivery',preparation_ms=preparation_ms if order==0 else 0.))
    if external._fingerprint(runtime)!=model_before:raise RuntimeError('Current-only weights changed during inference')
    details=pd.DataFrame(rows);latency,by_step=online.latency_tables(requests)
    return ExperimentReport('ALD current-only native sequential input',pd.DataFrame([dict(runs=len(boundaries),questions=len(rows),accuracy_available=False)]),
        details=details,diagnostics=dict(coverage=pd.DataFrame(coverage),exclusions=pd.DataFrame(exclusions),
            requests=pd.DataFrame(requests),provisional_invalidated_answers=pd.DataFrame(provisional),
            decisions=delivery.decisions(details),request_latency=latency,request_latency_by_step=by_step),
        metadata=dict(version=delivery.VERSION,input_mode='current_only',model_schema=SCHEMA,normal_model_recipe=normal.RECIPE,training_recipe=TRAINING_RECIPE,decision_system=fa.SYSTEM,
            checkpoint=str(runtime.module.FUSION_ANSWER_PATH),model_sha256=model_before,
            head_training_reference_mode='none',reference_contract=SCHEMA,original_used_for_prediction=False,
            run_rule=delivery.RUN_RULE,training=False,selection=False,automatic_export=False,
            independent_test_certified=False,real_fault_truth=False,
            timing_scope='current-only complete-window request; excludes acquisition, native-unit validation and file reading',
            limitation='Main results are provisional until Run quality is known. Complete SubCycles validate before replies. '
                       'Boundary tail windows emit only when that boundary arrives. '
                       'Any-Yes is a diagnostic, not a trained Run probability. Offline replay retains the full input/report in memory.'))


def replay_delivery(contract,package,*,expected_runs=20,center_batch_size=8,progress=print,on_result=None):
    from . import external_test as external,ald_delivery_evaluation as delivery
    package=Path(package).resolve();identity=delivery._inputs(package);started=time.perf_counter()
    frame=read_frame(package/'inputs/test.csv',contract)
    boundaries=delivery.delivery_boundaries(pd.read_csv(package/'inputs/run_boundaries.csv',dtype={'run_id':str}),len(frame))
    if expected_runs is not None and len(boundaries)!=expected_runs:raise ValueError('Unexpected delivery Run count')
    read_ms=1000*(time.perf_counter()-started)
    report=_replay_frame(contract,frame,boundaries,center_batch_size=center_batch_size,progress=progress,on_result=on_result)
    if delivery._inputs(package)!=identity:raise ValueError('Delivery inputs changed during replay')
    from . import peald_current_data as native
    reservation_match=contract.runtime.prepared.identity.get('delivery_metadata_identity')==native.delivery_metadata_identity(package)[1]
    report.metadata.update(package=str(package),input_identity=identity,input_read_ms=read_ms,
        delivery_reservation_metadata_matches_training=reservation_match)
    report.metadata['prediction_sha256']=delivery._seal(report)
    return report


def replay_csv(contract,path,*,center_batch_size=8,progress=print,on_result=None):
    from . import external_test as external,ald_delivery_evaluation as delivery
    path=Path(path).resolve();before=peald.digest(path);frame=read_frame(path,contract)
    runs=frame[contract.process.source_columns['run']].astype('string').fillna('')
    bounds=[]
    for i,(_,g) in enumerate(frame.groupby(runs.ne(runs.shift()).cumsum(),sort=False)):
        bounds.append(dict(run_id=f'INPUT-{i+1:04}',start=int(g.index[0]),end=int(g.index[-1])+1))
    report=_replay_frame(contract,frame,bounds,center_batch_size=center_batch_size,progress=progress,on_result=on_result)
    if peald.digest(path)!=before:raise ValueError('Input CSV changed during replay')
    report.metadata.update(input_identity={'csv_sha256':before},paired_truth_available=False)
    report.metadata['prediction_sha256']=delivery._seal(report)
    return report


def benchmark(runtime,*,steps=None,config=Config(),repeats=3,warmup=1,progress=print):
    from . import online_inference as online
    if type(repeats) is not int or repeats<1 or type(warmup) is not int or warmup<0:raise ValueError('Positive repeats and nonnegative warmup required')
    trained=runtime.module.FUSION_ANSWER_TRAINING['trained_steps']
    selected=trained if steps is None else list(steps)
    if not set(selected)<=set(trained):raise ValueError('Requested Step has no current-only head support')
    examples=[c for c in cases(runtime,'test',steps=selected,config=config) if not c['injected']]
    per_step={c['step']:c for c in examples};session=Session(runtime,center_batch_size=config.center_batch_size);requests=[]
    for step,case in per_step.items():
        for i in range(warmup+repeats):
            result=session.process_window(current=case['current'],step=step,run_id=case['run_id'],recipe_key=case['recipe_key'],
                offset_samples=case['offset_samples'],sensor_columns=case.get('sensor_columns'),standardized=True)
            if i>=warmup:requests.append(dict(request_id=f'{step}/{i}',step=step,status=result['status'],reason=result['reason'],
                answers=len(result['answers']),selected_sensors=len(case['centers']),**result['timings']))
        if progress:progress(f'Current-only timing: {step}',flush=True)
    latency,by_step=online.latency_tables(requests)
    return ExperimentReport('Current-only complete-window timing',pd.DataFrame(),
        diagnostics=dict(requests=pd.DataFrame(requests),request_latency=latency,request_latency_by_step=by_step),
        metadata=dict(input_mode='current_only',reference_policy='none',model_schema=SCHEMA,normal_model_recipe=normal.RECIPE,training_recipe=TRAINING_RECIPE,deployed_weights_updated=False,
            timing_scope='all supported centers and three answers per complete window; excludes collection and preprocessing'))
