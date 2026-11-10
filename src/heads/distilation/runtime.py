"""Merged Student + bit-preserved original head, independent of Teacher inference."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import time
import uuid

import numpy as np
import pandas as pd
import torch

from pecvd.experiments import ald_head_experiments as heads, ald_preprocessing as preprocessing
from pecvd.experiments import current_input as current, decision_training as d, fusion_answers as fa
from pecvd import answer_display
from .student import GevisStudent, inputs, boundary_tail, INPUT_CONTRACT
from .mapping import remap_reply, MAPPING_POLICY

SCHEMA = 'gevis.control48.original_head.merged.v1'


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def code_identity():
    # Source files only. Called explicitly on save/load, never on import.
    folder = Path(__file__).parent
    files = list(folder.glob('*.py')) + [Path(m.__file__) for m in
        (heads, fa, preprocessing, d, current, answer_display)]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def head_digest(head):
    return current._state_digest(head)


def teacher_binding(runtime):
    return dict(head=head_digest(runtime.module.FUSION_ANSWER_HEAD),
        references=d._hash(runtime.references), metadata=d._hash({s: m for s, (_, m) in runtime.encoders.items()}),
        source=current.source_identity(runtime), targets=runtime.target_sensors, domain=runtime.domain)


def validate_original_head(head):
    if getattr(getattr(head, 'choice', None), 'mode', 'softmax') != 'softmax':
        raise ValueError('Original-five Softmax head required; grouped/sigmoid heads cannot substitute')
    if head.answers['CHOICE'].out_features != 6 or head.answers['ORDINAL'].out_features != 5:
        raise ValueError('Original BOOLEAN/5-type CHOICE/4-stage ORDINAL head required')


class Session:
    """Original input validation/normalization and gate; fixed final argmax mapping.

    A loaded merged runtime has no Teacher encoders or E5 model. Student.encode
    runs once per Window and is reused for conditional positive answers.
    """
    def __init__(self, runtime, student=None, *, center_batch_size=None):
        self.runtime = runtime
        self.student = student if student is not None else getattr(runtime, 'student', None)
        self.center_batch_size = runtime.config.center_batch_size if center_batch_size is None else center_batch_size
        if type(self.center_batch_size) is not int or not 1 <= self.center_batch_size <= 32:
            raise ValueError('Center batch must be 1..32')
        validate_original_head(runtime.module.FUSION_ANSWER_HEAD)
        self.input_policy = getattr(runtime, 'distilation_input_policy',
            'domain' if isinstance(runtime.module.FUSION_ANSWER_HEAD, heads.ExperimentHead) else 'detailed')
        self.teacher_session = None
        if self.student is None:
            self.teacher_session = (heads.Session(runtime) if self.input_policy == 'domain'
                else d.Session(runtime, center_batch_size=self.center_batch_size))

    @torch.no_grad()
    def process_window(self, *, current, step, recipe_key, offset_samples, lineage_id,
                       loop_key='', centers=None, sensor_columns=None, standardized=False, run_id=None):
        request = dict(current=current, step=step, recipe_key=recipe_key, offset_samples=offset_samples,
            lineage_id=lineage_id, loop_key=loop_key, centers=centers, sensor_columns=sensor_columns,
            standardized=standardized, run_id=run_id)
        if self.teacher_session is not None:
            return remap_reply(self.teacher_session.process_window(**request))
        r = self.runtime
        def withheld(reason):
            return dict(status='NOT_EVALUATED', reason=reason, answers=[])
        if step not in r.encoders:
            return withheld('unsupported_native_context')
        meta = r.encoders[step][1]
        names = meta['sensor_columns']
        if sensor_columns is not None and list(sensor_columns) != names:
            return withheld('incomplete_observed_sensor_grid')
        if not isinstance(lineage_id, str) or not lineage_id:
            return withheld('source_lineage_required')
        if lineage_id in set(r.references['basis_lineages']):
            return withheld('reference_basis_lineage_not_independent')
        record = r.reference_bank.get(d._key(step, recipe_key, loop_key, offset_samples))
        if record is None:
            return withheld('native_normal_reference_unobserved')
        targets = r.target_sensors[step]
        chosen = list(targets) if centers is None else list(centers)
        if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(names):
            raise ValueError('Unique known centers required')
        if set(chosen) - set(targets):
            raise ValueError('Context-only sensor cannot be selected as a prediction target')
        try:
            if standardized:
                values = np.asarray(current, np.float32)
            elif self.input_policy == 'detailed':
                values = np.asarray(current, np.float32)
                if values.shape != (meta['window_length'], len(names)) or not np.isfinite(values).all():
                    raise ValueError('Finite complete native window required')
                values = ((values - np.asarray(meta['mean'])) / np.asarray(meta['std'])).astype(np.float32)
            else:
                frame = current if isinstance(current, pd.DataFrame) else pd.DataFrame(current, columns=names)
                values = preprocessing.transform_live(frame, meta)
            if values.shape != (meta['window_length'], len(names)) or not np.isfinite(values).all():
                raise ValueError('Finite complete native window required')
        except (ValueError, TypeError) as error:
            return withheld(str(error))
        head = r.module.FUSION_ANSWER_HEAD
        device = r.module.DEVICE
        state = self.student.encode(inputs(values, record).to(device))
        decoded = {}
        def infer(selected, kinds):
            for start in range(0, len(selected), self.center_batch_size):
                batch = selected[start:start + self.center_batch_size]
                ids = torch.tensor([names.index(c) for c in batch], device=device)
                tail = boundary_tail(values, record, batch, names, offset_samples, float(meta['cadence_seconds']))
                features = self.student.features(state, ids, tail, kinds)
                for i, center in enumerate(batch):
                    for j, kind in enumerate(kinds):
                        logits = head(features[i, j:j + 1], kind).clone()
                        if kind == 'CHOICE' and hasattr(head, 'choice_probabilities'):
                            probs = head.choice_probabilities(logits)[0].cpu()
                            labels = head.choice_labels
                            prediction = int(probs.argmax()) + 1
                            label = labels[prediction - 1]
                        else:
                            if kind != 'BOOLEAN':
                                logits[:, 0] = -torch.inf
                            probs = logits.softmax(-1)[0].cpu()
                            labels = fa.ANSWER_CRITERIA[kind]
                            prediction = int(probs.argmax())
                            label = labels[prediction]
                        decoded[center, kind] = dict(center_sensor=center, answer_type=kind,
                            prediction=prediction, answer=label, probabilities=dict(zip(labels, map(float, probs))),
                            evaluated=True, status='EVALUATED')
        infer(chosen, ('BOOLEAN',))
        positive = [c for c in chosen if decoded[c, 'BOOLEAN']['prediction'] == 1]
        if positive:
            infer(positive, ('CHOICE', 'ORDINAL'))
        answers = [decoded.get((c, k), dict(center_sensor=c, answer_type=k, prediction=None,
            answer=None, probabilities=None, evaluated=False, status='NOT_APPLICABLE', reason='BOOLEAN_No'))
            for c in chosen for k in fa.KINDS]
        return remap_reply(dict(status='OK', reason=None, answers=answers, run_id=run_id,
            reference_count=record['reference_runs'], selected_sensors=len(chosen),
            evaluated_answers=sum(a['evaluated'] for a in answers), conditional_answering=True))


def save_merged(root, teacher, student, *, config, training, reports=None):
    """Explicit new artifact only; never replace a Teacher checkpoint or pointer."""
    validate_original_head(teacher.module.FUSION_ANSWER_HEAD)
    head = teacher.module.FUSION_ANSWER_HEAD
    patch_lengths = {int(e.patch_len) for e, _ in teacher.encoders.values()}
    if patch_lengths != {student.patch_length} or student.width != 48 or len(student.temporal.layers) != 2:
        raise ValueError('Source patch grid and control48 architecture must match')
    head_config = dict(dimension=head.feature_mean.numel(), width=head.answers['BOOLEAN'].in_features,
        implementation='experiment' if isinstance(head, heads.ExperimentHead) else 'context',
        choice=asdict(head.choice) if isinstance(head, heads.ExperimentHead) else None)
    state = {'student.' + k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    state.update({'head.' + k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    payload = dict(schema=SCHEMA, code=code_identity(), input_contract=INPUT_CONTRACT, mapping=MAPPING_POLICY,
        config=asdict(config), teacher_config=asdict(teacher.config), binding=teacher_binding(teacher),
        input_policy=getattr(teacher, 'distilation_input_policy',
            'domain' if isinstance(head, heads.ExperimentHead) else 'detailed'),
        metadata={s: deepcopy(m) for s, (_, m) in teacher.encoders.items()}, references=deepcopy(teacher.references),
        targets=deepcopy(teacher.target_sensors), head_config=head_config, patch_length=student.patch_length,
        question_vectors=student.question_vectors.detach().cpu().tolist(),
        training=deepcopy(training), reports=reports, state=state)
    payload['content_sha256'] = current.payload_digest(payload)
    folder = Path(root).resolve() / 'artifacts/gevis_distilation' / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / 'control48.pt'
    torch.save(payload, path)
    return path


def load_merged(checkpoint, *, device='cpu'):
    """Load Student + original head without loading Teacher/E5 or reading CSVs."""
    from .objective import Config
    path = Path(checkpoint).resolve()
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA or payload.get('content_sha256') != current.payload_digest(payload):
        raise ValueError('Merged package schema/content differs')
    if payload['code'] != code_identity() or payload['input_contract'] != INPUT_CONTRACT or payload['mapping'] != MAPPING_POLICY:
        raise ValueError('Merged source/input/mapping contract differs')
    config = Config(**payload['config']); config.validate()
    shape = payload['head_config']
    if shape['implementation'] == 'experiment':
        head = heads.ExperimentHead(shape['dimension'], shape['width'], heads.ChoiceConfig(**shape['choice']))
    elif shape['implementation'] == 'context':
        head = fa.ContextReadout(shape['dimension'], shape['width'])
    else:
        raise ValueError('Unknown original head implementation')
    head.load_state_dict({k[5:]: v for k, v in payload['state'].items() if k.startswith('head.')}, strict=True)
    validate_original_head(head)
    if head_digest(head) != payload['binding']['head'] or d._hash(payload['references']) != payload['binding']['references']:
        raise ValueError('Original head/reference binding differs')
    if (d._hash(payload['metadata']) != payload['binding']['metadata']
            or payload['targets'] != payload['binding']['targets']):
        raise ValueError('Saved input/target binding differs')
    student = GevisStudent(torch.tensor(payload['question_vectors']), payload['patch_length'], 48, 2)
    student.load_state_dict({k[8:]: v for k, v in payload['state'].items() if k.startswith('student.')}, strict=True)
    device = torch.device(device)
    head.requires_grad_(False).eval().to(device)
    student.requires_grad_(False).eval().to(device)
    return SimpleNamespace(module=SimpleNamespace(DEVICE=device, FUSION_ANSWER_HEAD=head), student=student,
        encoders={s: (None, m) for s, m in payload['metadata'].items()},
        references=payload['references'], reference_bank=d._bank(payload['references']),
        target_sensors=payload['targets'], domain=payload['binding']['domain'],
        distilation_input_policy=payload['input_policy'],
        config=d.Config(**payload['teacher_config']), distillation_config=config,
        training=payload['training'], reports=payload['reports'], checkpoint=path,
        teacher_binding=payload['binding'])
