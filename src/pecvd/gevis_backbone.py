"""Question-conditioned backbone and sensor-free discovery curricula for GEVIS.

The active curriculum holds question text constant while controlled numeric
evidence moves across sensors and patches. Those targets are not equipment-fault
labels, and attention localization is not causal diagnosis.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import math
import random
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


ANSWER_TYPES = {"BOOLEAN": 0, "CHOICE": 1, "ORDINAL": 2}
FEATURE_TYPES = {"LEVEL": 0, "VARIABILITY": 1, "POINT": 2, "RESIDUAL": 3}
SEMANTIC_TYPES = {
    "statistic": {"LEVEL": 0, "VARIABILITY": 1, "SLOPE": 2, "POINT": 3, "RESIDUAL": 4},
    "direction": {"INCREASE": 0, "DECREASE": 1, "CHANGE": 2},
    "time_scope": {"WHOLE": 0, "EARLY": 1, "LATE": 2},
}
BACKBONE_SOURCE_SCHEMA = "gevis.backbone_source.v9"
SEMANTIC_CURRICULUM_VERSION = "fdc_semantic_step1_bridge_all_types_v1"
CROSS_BINDING_VERSION = "real_step1_binding_v2"
EVIDENCE_BINDING_VERSION = "real_step1_evidence_v3"
TARGET_BINDING_VERSION = "real_step1_target_binding_v4"
TARGET_FUSION_MODE = "descriptor_anchor_v4"
PRETRAINED_BINDING_VERSION = "real_step1_pretrained_binding_v5"
GENERIC_DISCOVERY_VERSION = "real_step1_generic_discovery_v10"
REFERENCE_DISCOVERY_VERSION = 'reference_aware_discovery_v11'
REFERENCE_FUSION_MODE = 'numeric_first_generic_v11'


def token_descriptions(phase,sensor_names,patch_count):
    if patch_count<1 or not sensor_names: raise ValueError('Nonempty sensor/patch grid required')
    scopes=['early interval' if p<patch_count/3 else 'late interval' if p>=2*patch_count/3 else 'middle interval' for p in range(patch_count)]
    return [f'Process phase {phase}; sensor {sensor}; relative time {scope}.' for sensor in sensor_names for scope in scopes]


def attention_inputs(numeric_keys, descriptors, text_query, typed_query, gate_layer, mode='legacy'):
    """One train/runtime fusion contract; no target IDs or oracle metadata.

    Named-target candidate anchors keys in sensor/time descriptions. Numeric
    features still contribute through a learned bounded gate; answer type enters
    the fused residual but cannot itself redirect sensor attention.
    """
    if mode not in ('legacy',TARGET_FUSION_MODE,REFERENCE_FUSION_MODE): raise ValueError('Unknown GEVIS fusion mode')
    if descriptors is None:
        if mode!= 'legacy': raise ValueError('Target binding requires token descriptions')
        return typed_query,numeric_keys
    gate=gate_layer(torch.cat([numeric_keys,descriptors],-1))
    if mode==REFERENCE_FUSION_MODE:
        return text_query,numeric_keys+gate*descriptors
    if mode==TARGET_FUSION_MODE:
        return text_query,descriptors+gate*numeric_keys
    return typed_query,numeric_keys+gate*descriptors
DEFAULT_TEXT_ENCODER = "intfloat/e5-small-v2"
DEFAULT_TEXT_ENCODER_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"


@dataclass(frozen=True)
class QuestionExample:
    text: str
    answer_type: str | None
    supported: bool


@dataclass(frozen=True)
class SemanticQuestionExample:
    text: str
    answer_type: str
    statistic: str
    direction: str
    time_scope: str
    contract: str
    step: str = ''
    sensor: str = ''
    variant: int = 0


STATISTIC_WORDS = {'LEVEL': 'mean level', 'VARIABILITY': 'variability', 'SLOPE': 'trend slope', 'POINT': 'local deviation', 'RESIDUAL': 'reconstruction residual'}
DIRECTION_WORDS = {'INCREASE': 'increase', 'DECREASE': 'decrease', 'CHANGE': 'change'}
SCOPE_WORDS = {'WHOLE': 'the full interval', 'EARLY': 'the early interval', 'LATE': 'the late interval'}
SEMANTIC_SEEDS = {'train': 104729, 'validation': 130363, 'test': 155921, 'audit': 196613}


def render_semantic_question(step, sensor, statistic, direction, time_scope, *, split='train', variant=0, answer_type='BOOLEAN'):
    """Render sensor-free wording; ``sensor`` remains audit metadata only."""
    statistic_words = {
        'LEVEL': ('mean level', 'average value', 'signal baseline'),
        'VARIABILITY': ('variability', 'signal fluctuation', 'standard deviation'),
        'SLOPE': ('trend slope', 'rate of change', 'temporal gradient'),
        'POINT': ('local deviation', 'pointwise difference', 'instantaneous departure'),
        'RESIDUAL': ('reconstruction residual', 'reconstruction error', 'unexplained residual'),
    }
    direction_words = {
        'INCREASE': ('increase', 'rise', 'upward shift'),
        'DECREASE': ('decrease', 'drop', 'downward shift'),
        'CHANGE': ('change', 'departure', 'difference'),
    }
    scope_words = {
        'WHOLE': ('the full interval', 'the entire window', 'the complete segment'),
        'EARLY': ('the early interval', 'the beginning of the window', 'the initial segment'),
        'LATE': ('the late interval', 'the end of the window', 'the final segment'),
    }
    # Separate strides prevent statistic, direction and scope wording from becoming shortcuts.
    st = statistic_words[statistic][variant % 3]
    dr = direction_words[direction][(variant // 3) % 3]
    sc = scope_words[time_scope][(variant // 9) % 3]
    templates = {
        'train': ('For {step}, search all signals for a {dr} in {st} over {sc}.',
                  'Compare every measurement with the reference during {sc}; focus on the {dr} in {st}. Process: {step}.',
                  'Search process {step}. Determine the strongest {dr} of {st} within {sc}.',
                  'Restrict the all-signal comparison to {sc}. Metric: {st}; requested direction: {dr}; phase: {step}.',
                  'Using the reference traces, inspect every signal in {sc} for a {dr} in {st}. Phase: {step}.',
                  'Does any signal show a {dr} in {st} during {sc}? Evaluate phase {step}.',
                  'Review all measurements in {step}: within {sc}, compare {st} for a {dr}.',
                  'The requested operation is {st}; direction {dr}; time scope {sc}; search all signals in {step}.',
                  'Analyze {sc} of {step} and locate the strongest reference-relative {dr} in {st}.',
                  'For the specified {step} phase, assess every signal for {st}, {dr}, and {sc}.'),
        'validation': ('The comparison metric is {st}. Did any signal show a {dr} during {sc} in {step}?',),
        'test': ('For {step}, locate the strongest reference-relative {dr} of {st} within {sc}.',),
        'audit': ('Independent review: search all signals in {sc} and assess whether {st} has a {dr} in {step}.',),
    }
    choices = templates[split]
    text = choices[(variant * 7 + variant // 3) % len(choices)].format(sc=sc, st=st, dr=dr, step=step)
    prefixes = {
        'BOOLEAN': 'Binary review question: ',
        'CHOICE': 'Change-type selection question: ',
        'ORDINAL': 'Change-magnitude rating question: ',
    }
    if answer_type not in prefixes:
        raise ValueError('Unknown semantic answer type')
    return prefixes[answer_type] + text


def make_semantic_questions(split, *, seed=None, variants=24, answer_types=None):
    """Each new identifier sees the full operation product: names predict no label."""
    rng = random.Random(SEMANTIC_SEEDS[split] if seed is None else seed)
    rows = []
    for variant in range(variants):
        wording = variant + (rng.randrange(6) if split == 'train' else 0)
        step = f'{split}_phase_{rng.randrange(10**9):09d}'
        sensor = f'{split}_sensor_{rng.randrange(10**9):09d}'
        for answer_type in (tuple(ANSWER_TYPES) if answer_types is None else answer_types):
            for statistic in SEMANTIC_TYPES['statistic']:
                for direction in SEMANTIC_TYPES['direction']:
                    for scope in SEMANTIC_TYPES['time_scope']:
                        rows.append(SemanticQuestionExample(render_semantic_question(step, sensor, statistic, direction, scope, split=split, variant=wording, answer_type=answer_type), answer_type, statistic, direction, scope, split, step, sensor, wording))
    rng.shuffle(rows)
    return tuple(rows)


SEMANTIC_QUESTION_BANK = {split: make_semantic_questions(split) for split in SEMANTIC_SEEDS}


def semantic_counterfactual(item, field):
    """Change exactly one semantic slot, retaining names, wording and answer type."""
    values = tuple(SEMANTIC_TYPES[field])
    changed = replace(item, **{field: values[(values.index(getattr(item, field))+1)%len(values)]})
    return replace(changed, text=render_semantic_question(
        changed.step, changed.sensor, changed.statistic, changed.direction,
        changed.time_scope, split=changed.contract, variant=changed.variant,
        answer_type=changed.answer_type))


def _supported(texts: Iterable[str], answer_type: str) -> list[QuestionExample]:
    return [QuestionExample(text, answer_type, True) for text in texts]


def _unsupported(texts: Iterable[str]) -> list[QuestionExample]:
    return [QuestionExample(text, None, False) for text in texts]


def _english_product(leads: tuple[str, ...], bodies: tuple[str, ...]) -> list[str]:
    return [f"{lead} {body}" for lead in leads for body in bodies]


QUESTION_BANK = {
    "train": tuple(
        _supported(_english_product(
            ("Review whether", "Determine whether", "Check whether", "Assess whether"),
            ("the current segment differs from its reference.", "the signal requires engineer review.",
             "the process trace contains a notable change.", "the current window departs from its usual pattern.",
             "the available evidence justifies additional inspection."),
        ), "BOOLEAN")
        + _supported(_english_product(
            ("Identify", "Select", "Report", "Describe"),
            ("the dominant type of change against the reference.", "the feature that differs most from the usual trace.",
             "whether level, variability, timing, or residual behavior dominates.", "the clearest form of signal departure.",
             "the main pattern difference in the current segment."),
        ), "CHOICE")
        + _supported(_english_product(
            ("Rate", "Assign", "Estimate", "Choose"),
            ("the review priority for this change.", "the magnitude category of the observed difference.",
             "the urgency level for engineer inspection.", "the appropriate response tier for this segment.",
             "the severity band supported by the trace evidence."),
        ), "ORDINAL")
        + _unsupported(_english_product(
            ("Without additional evidence,", "Using only this trace,", "Without a verified label,", "Automatically"),
            ("guarantee the next wafer yield.", "confirm the physical root cause.", "order a replacement component.",
             "change the equipment recipe.", "declare a definite equipment failure."),
        ))
    ),
    "validation": tuple(
        _supported(["Does this segment require human review?", "Is there a reference-relative change worth inspecting?", "Should an engineer examine the current trace?", "Does the evidence indicate an unusual process segment?"], "BOOLEAN")
        + _supported(["Which change type is most prominent?", "What is the primary difference from the reference?", "Select the dominant signal-change category.", "Which feature best describes the observed departure?"], "CHOICE")
        + _supported(["What review-priority level is appropriate?", "Rate the magnitude of the current difference.", "Which response tier matches this evidence?", "How urgently should this segment be inspected?"], "ORDINAL")
        + _unsupported(["Prove the unmeasured film quality.", "Shut down the equipment immediately.", "Guarantee future production output.", "Declare an unavailable root cause."])
    ),
    "test": tuple(
        _supported(["Is this interval a review candidate?", "Does the log contain a change that needs inspection?", "Is the current flow meaningfully different from the reference?", "Should this state receive additional observation?"], "BOOLEAN")
        + _supported(["How does this interval differ from normal operation?", "Which change pattern is strongest?", "What feature changed most relative to the reference?", "Which category best explains the difference?"], "CHOICE")
        + _supported(["How quickly should an engineer review this?", "What inspection grade fits the current state?", "Which response level is supported by this change?", "How should this case be prioritized?"], "ORDINAL")
        + _unsupported(["Determine the defect cause without verification.", "Purchase equipment parts automatically.", "Guarantee a future sensor value.", "Answer an unrelated general-knowledge question."])
    ),
}


FEATURE_QUESTIONS = {
    "LEVEL": ["Did the signal level change?", "Did the mean value move?", "Is the value above or below the reference?", "Locate the strongest level difference."],
    "VARIABILITY": ["Did signal variability change?", "Did the fluctuation range increase?", "Did signal stability change?", "Locate the interval with different variance."],
    "POINT": ["Where is the largest instantaneous difference?", "Locate concentrated pointwise differences.", "Is there a sudden local departure?", "Where is the largest raw-value difference?"],
    "RESIDUAL": ["Where is reconstruction error largest?", "Which interval is poorly explained by the reference pattern?", "Locate the largest reconstruction difference.", "Where is the PatchTST residual most prominent?"],
}


class ByteTextEncoder(nn.Module):
    """Local FDC text baseline with byte coverage and a learned summary token."""

    def __init__(self, d_model: int = 192, max_bytes: int = 512, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.max_bytes = max_bytes
        self.token = nn.Embedding(258, d_model, padding_idx=0)
        self.position = nn.Embedding(max_bytes + 1, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, 4, 4 * d_model, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        encoded = [list(text.encode("utf-8")[: self.max_bytes]) for text in texts]
        width = max(1, max(map(len, encoded))) + 1
        ids = torch.zeros((len(texts), width), dtype=torch.long, device=device)
        ids[:, 0] = 257
        for index, item in enumerate(encoded):
            if item:
                ids[index, 1 : len(item) + 1] = torch.tensor(item, device=device) + 1
        positions = torch.arange(width, device=device)[None]
        valid = ids.ne(0)
        value = self.encoder(
            self.token(ids) + self.position(positions), src_key_padding_mask=~valid
        )
        return self.norm(value[:, 0])


class E5TextEncoder(nn.Module):
    """Frozen English E5 encoder with a small trainable projection into GEVIS space."""

    def __init__(
        self,
        d_model: int,
        model_name: str = DEFAULT_TEXT_ENCODER,
        *,
        revision: str = DEFAULT_TEXT_ENCODER_REVISION,
        max_length: int = 128,
        tokenizer=None,
        model=None,
    ):
        super().__init__()
        if tokenizer is None or model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "E5 text encoding requires transformers; install config/requirements-a.txt"
                ) from error
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModel.from_pretrained(model_name, revision=revision)
        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()
        hidden_size = int(self.model.config.hidden_size)
        self.projection = nn.Linear(hidden_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def train(self, mode: bool = True):
        super().train(mode)
        # The pretrained encoder stays deterministic; only projection/norm train.
        self.model.eval()
        return self

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        if any(any('\uac00' <= char <= '\ud7a3' or char == '\ufffd' for char in text) for text in texts):
            raise ValueError('English E5 input required: use English questions, options and metadata descriptors; damaged text is not accepted')
        prefixed = [text if text.startswith("query: ") else f"query: {text}" for text in texts]
        encoded = self.tokenizer(
            prefixed,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = F.normalize(pooled, p=2, dim=-1)
        return self.norm(self.projection(pooled))


class GEVISBackbone(nn.Module):
    def __init__(
        self,
        numeric_dim: int = 264,
        d_model: int = 128,
        dropout: float = 0.1,
        text_encoder_name: str = "byte",
        text_encoder_revision: str = DEFAULT_TEXT_ENCODER_REVISION,
    ):
        super().__init__()
        self.text_encoder_name = text_encoder_name
        self.text = (
            ByteTextEncoder(d_model, dropout=dropout)
            if text_encoder_name == "byte"
            else E5TextEncoder(d_model, model_name=text_encoder_name, revision=text_encoder_revision)
        )
        self.numeric_input_norm = nn.LayerNorm(numeric_dim)
        self.numeric = nn.Linear(numeric_dim, d_model)
        from .temporal_bridge import install
        install(self,numeric_dim,d_model)
        self.numeric_key_norm = nn.LayerNorm(d_model)
        self.descriptor_norm = nn.LayerNorm(d_model)
        self.fusion_gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.type_emb = nn.Embedding(3, d_model)
        self.cross = nn.MultiheadAttention(d_model, 4, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        numeric_tokens: torch.Tensor,
        questions: list[str],
        answer_types: list[str],
        token_descriptions: list[list[str]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = numeric_tokens.device
        type_ids = torch.tensor([ANSWER_TYPES[item] for item in answer_types], device=device)
        text_query = self.text(questions, device)
        query = text_query + self.type_emb(type_ids)
        numeric_keys = self.numeric_key_norm(self.numeric(self.numeric_input_norm(numeric_tokens)))
        descriptors = None
        if token_descriptions is not None:
            if len(token_descriptions) != len(numeric_tokens):
                raise ValueError("token_descriptions batch does not match numeric tokens")
            if any(len(items) != numeric_tokens.shape[1] for items in token_descriptions):
                raise ValueError("token_descriptions length does not match numeric token count")
            flat = [description for items in token_descriptions for description in items]
            unique = list(dict.fromkeys(flat)); lookup = {text: index for index, text in enumerate(unique)}
            indices = torch.tensor([lookup[text] for text in flat], device=device)
            descriptors = self.text(unique, device).index_select(0, indices).reshape_as(numeric_keys)
            descriptors = self.descriptor_norm(descriptors)
        attention_query,keys=attention_inputs(numeric_keys,descriptors,text_query,query,self.fusion_gate,getattr(self,'fusion_mode','legacy'))
        fused, attention = self.cross(attention_query[:, None], keys, keys, need_weights=True)
        return self.norm(fused[:, 0] + query), attention[:, 0]


class BackbonePretrainHeads(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.type_head = nn.Linear(d_model, len(ANSWER_TYPES))
        self.support_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.feature_head = nn.Linear(d_model, len(FEATURE_TYPES))
        self.semantic_heads = nn.ModuleDict(
            {name: nn.Linear(d_model, len(values)) for name, values in SEMANTIC_TYPES.items()}
        )
        self.grounding_head = nn.Linear(d_model, 2)


def _semantic_loss(backbone, heads, examples, device):
    embeddings = backbone.text([item.text for item in examples], device)
    renamed = [item.text.replace(item.step, 'replacement_phase').replace(item.sensor, 'replacement_sensor') for item in examples]
    renamed_embeddings = backbone.text(renamed, device)
    fields = tuple(SEMANTIC_TYPES)
    mutated_fields = [fields[index % len(fields)] for index in range(len(examples))]
    contrasts = [semantic_counterfactual(item, field) for item, field in zip(examples, mutated_fields)]
    contrast_embeddings = backbone.text([item.text for item in contrasts], device)
    losses = {}
    for field, values in SEMANTIC_TYPES.items():
        targets = torch.tensor([values[getattr(item, field)] for item in examples], device=device)
        logits = heads.semantic_heads[field](embeddings)
        renamed_logits = heads.semantic_heads[field](renamed_embeddings)
        contrast_logits = heads.semantic_heads[field](contrast_embeddings)
        contrast_targets = torch.tensor([values[getattr(item, field)] for item in contrasts], device=device)
        changed = torch.tensor([name == field for name in mutated_fields], device=device)
        margin = logits.sum()*0.
        if changed.any():
            # A changed slot must change its prediction, not just the embedding.
            old = targets[changed, None]; new = contrast_targets[changed, None]
            before = logits[changed].gather(1,old)-logits[changed].gather(1,new)
            after = contrast_logits[changed].gather(1,new)-contrast_logits[changed].gather(1,old)
            margin = .5*(F.relu(.5-before).mean()+F.relu(.5-after).mean())
        invariant = logits.sum()*0.
        if (~changed).any():
            invariant = F.mse_loss(logits[~changed].softmax(-1),contrast_logits[~changed].softmax(-1))
        losses[field] = (F.cross_entropy(logits, targets) + F.cross_entropy(renamed_logits, targets) + F.cross_entropy(contrast_logits, contrast_targets))/3 + .1 * F.mse_loss(logits.softmax(-1), renamed_logits.softmax(-1)) + .1*invariant + .2*margin
    type_targets = torch.tensor([ANSWER_TYPES[item.answer_type] for item in examples],device=device)
    type_loss = F.cross_entropy(heads.type_head(embeddings),type_targets)
    loss = torch.stack(list(losses.values())).mean() + .2*type_loss
    return loss, {f"semantic_{name}_loss": float(value.detach()) for name, value in losses.items()}


@torch.no_grad()
def evaluate_semantics(backbone, heads, examples, split, device):
    backbone.eval(); heads.eval()
    embeddings = torch.cat([backbone.text([item.text for item in examples[start:start+32]], device) for start in range(0, len(examples), 32)])
    result = {"split": split}
    for field, values in SEMANTIC_TYPES.items():
        targets = torch.tensor([values[getattr(item, field)] for item in examples], device=device)
        predictions = heads.semantic_heads[field](embeddings).argmax(-1)
        result[f"{field}_accuracy"] = float(predictions.eq(targets).float().mean())
    result["within_type_semantic_accuracy"] = sum(
        result[f"{field}_accuracy"] for field in SEMANTIC_TYPES
    ) / len(SEMANTIC_TYPES)
    result['joint_semantic_accuracy'] = float(torch.stack([heads.semantic_heads[field](embeddings).argmax(-1).eq(torch.tensor([values[getattr(item, field)] for item in examples], device=device)) for field, values in SEMANTIC_TYPES.items()]).all(0).float().mean())
    renamed = [item.text.replace(item.step, 'unseen_phase').replace(item.sensor, 'unseen_sensor') for item in examples]
    renamed_embeddings = torch.cat([backbone.text(renamed[start:start+32], device) for start in range(0, len(renamed), 32)])
    result['identifier_prediction_agreement'] = float(torch.stack([heads.semantic_heads[field](embeddings).argmax(-1).eq(heads.semantic_heads[field](renamed_embeddings).argmax(-1)) for field in SEMANTIC_TYPES]).all(0).float().mean())
    return result


@torch.no_grad()
def semantic_diagnostics(backbone, heads, examples, split, device):
    """Per-type slot confusion and one-slot paired success; no numeric labels used."""
    backbone.eval(); heads.eval()
    def predict(rows):
        embeddings = torch.cat([backbone.text([item.text for item in rows[start:start+32]], device) for start in range(0,len(rows),32)])
        return {field: heads.semantic_heads[field](embeddings).argmax(-1).cpu() for field in SEMANTIC_TYPES}
    original = predict(examples); confusion=[]; pairs=[]
    for field, values in SEMANTIC_TYPES.items():
        targets = torch.tensor([values[getattr(item,field)] for item in examples])
        for kind in ANSWER_TYPES:
            selected = torch.tensor([item.answer_type==kind for item in examples])
            for true_name,true_id in values.items():
                for pred_name,pred_id in values.items():
                    count=int((selected & (targets==true_id) & (original[field]==pred_id)).sum())
                    confusion.append({'split':split,'answer_type':kind,'slot':field,'true_class':true_name,'predicted_class':pred_name,'count':count})
        contrasts=[semantic_counterfactual(item,field) for item in examples]
        mutated=predict(contrasts)
        new_targets=torch.tensor([values[getattr(item,field)] for item in contrasts])
        success=(original[field]==targets)&(mutated[field]==new_targets)
        unchanged=torch.stack([original[other]==mutated[other] for other in SEMANTIC_TYPES if other!=field]).all(0)
        for kind in ANSWER_TYPES:
            selected=torch.tensor([item.answer_type==kind for item in examples])
            if selected.any():
                pairs.append({'split':split,'answer_type':kind,'changed_slot':field,'pairs':int(selected.sum()),'changed_slot_pair_accuracy':float(success[selected].float().mean()),'unchanged_slots_agreement':float(unchanged[selected].float().mean())})
    return {'confusion':confusion,'contrast_pairs':pairs}


def _text_loss(
    backbone: GEVISBackbone,
    heads: BackbonePretrainHeads,
    examples: list[QuestionExample],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    embeddings = backbone.text([item.text for item in examples], device)
    supported = torch.tensor([item.supported for item in examples], dtype=torch.float32, device=device)
    support_logits = heads.support_head(embeddings).squeeze(-1)
    positive = F.binary_cross_entropy_with_logits(support_logits[supported.bool()], supported[supported.bool()])
    negative = F.binary_cross_entropy_with_logits(support_logits[~supported.bool()], supported[~supported.bool()])
    support_loss = 0.5 * (positive + negative)
    supported_index = supported.bool()
    targets = torch.tensor(
        [ANSWER_TYPES[item.answer_type] for item in examples if item.supported], device=device
    )
    type_loss = F.cross_entropy(heads.type_head(embeddings[supported_index]), targets)
    loss = type_loss + support_loss
    return loss, {"type_loss": float(type_loss.detach()), "support_loss": float(support_loss.detach())}


@torch.no_grad()
def evaluate_text(
    backbone: GEVISBackbone,
    heads: BackbonePretrainHeads,
    examples: list[QuestionExample],
    split: str,
    device: torch.device,
) -> dict[str, float | str]:
    backbone.eval(); heads.eval()
    embeddings = backbone.text([item.text for item in examples], device)
    supported = torch.tensor([item.supported for item in examples], dtype=torch.bool, device=device)
    support_prediction = heads.support_head(embeddings).squeeze(-1).sigmoid().ge(0.5)
    type_targets = torch.tensor(
        [ANSWER_TYPES[item.answer_type] for item in examples if item.supported], device=device
    )
    type_prediction = heads.type_head(embeddings[supported]).argmax(-1)
    return {
        "split": split,
        "question_type_accuracy": float(type_prediction.eq(type_targets).float().mean()),
        "support_accuracy": float(support_prediction.eq(supported).float().mean()),
        "supported_recall": float(support_prediction[supported].float().mean()),
        "unsupported_recall": float((~support_prediction[~supported]).float().mean()),
    }


def train_text_encoder(
    backbone: GEVISBackbone,
    heads: BackbonePretrainHeads,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    steps_per_epoch: int = 12,
    progress=print,
) -> tuple[list[dict[str, float]], int]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in backbone.text.parameters() if parameter.requires_grad]
        + list(heads.type_head.parameters())
        + list(heads.support_head.parameters())
        + list(heads.semantic_heads.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    train_supported = [item for item in QUESTION_BANK["train"] if item.supported]
    train_unsupported = [item for item in QUESTION_BANK["train"] if not item.supported]
    best_state = None
    best_score = float("-inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        backbone.text.train(); heads.train(); rng.shuffle(train_supported); rng.shuffle(train_unsupported)
        supported_per_batch = max(3, int(batch_size * 0.75)); unsupported_per_batch = max(1, batch_size - supported_per_batch)
        epoch_losses: list[float] = []; epoch_type_losses: list[float] = []; epoch_support_losses: list[float] = []
        semantic_rows = list(make_semantic_questions('train', seed=seed+epoch, variants=48)); rng.shuffle(semantic_rows)
        if steps_per_epoch < 1: raise ValueError('Positive semantic steps_per_epoch required')
        for offset in range(0, min(len(semantic_rows),batch_size*steps_per_epoch), batch_size):
            typed_offset = offset % len(train_supported)
            supported_batch = [train_supported[(typed_offset+i) % len(train_supported)] for i in range(supported_per_batch)]
            unsupported_batch = [train_unsupported[(offset + index) % len(train_unsupported)] for index in range(unsupported_per_batch)]
            batch = supported_batch + unsupported_batch; rng.shuffle(batch)
            loss, parts = _text_loss(backbone, heads, batch, device)
            semantic_batch = semantic_rows[offset:offset+batch_size]
            semantic_loss, semantic_parts = _semantic_loss(
                backbone, heads, semantic_batch[: min(batch_size, len(semantic_batch))], device
            )
            loss = loss + semantic_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_losses.append(float(loss.detach())); epoch_type_losses.append(parts["type_loss"]); epoch_support_losses.append(parts["support_loss"])
        validation = evaluate_text(backbone, heads, QUESTION_BANK["validation"], "validation", device)
        semantic_validation = evaluate_semantics(
            backbone, heads, list(SEMANTIC_QUESTION_BANK["validation"]), "validation", device
        )
        score = (
            float(validation["question_type_accuracy"])
            + float(validation["support_accuracy"])
            + 2*float(semantic_validation["within_type_semantic_accuracy"])
            + 2*float(semantic_validation["joint_semantic_accuracy"])
        ) / 6.0
        scheduler.step(score)
        history.append({"epoch": epoch, "loss": sum(epoch_losses)/len(epoch_losses), "validation_score": score, "type_loss":sum(epoch_type_losses)/len(epoch_type_losses), "support_loss":sum(epoch_support_losses)/len(epoch_support_losses), **semantic_validation})
        progress({'phase':'text_semantics','epoch':epoch,'loss':history[-1]['loss'],'validation_score':score,'validation_joint_semantic_accuracy':semantic_validation['joint_semantic_accuracy'],'updates':len(epoch_losses)})
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {"backbone": deepcopy(backbone.state_dict()), "heads": deepcopy(heads.state_dict())}
        else:
            stale += 1
        if epoch >= 10 and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Text pretraining did not produce a checkpoint")
    backbone.load_state_dict(best_state["backbone"])
    heads.load_state_dict(best_state["heads"])
    return history, best_epoch


def controlled_numeric_batch(
    batch_size: int,
    token_count: int,
    numeric_dim: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, list[str], list[str], list[list[str]], torch.Tensor, dict[str, torch.Tensor | int]]:
    """Build counterfactual sensor-by-time grounding examples.

    Adjacent rows contain the same numeric state but ask about different sensors.
    Every sensor has comparable evidence, so attention cannot use one globally
    salient token or a target-only numeric amplitude shortcut. Tokens are
    sensor-major with EARLY/MIDDLE/LATE patches, matching STEP 3.
    """
    patch_count = 3
    if token_count < patch_count * 2 or token_count % patch_count:
        raise ValueError("token_count must contain at least two sensors and be divisible by three")
    if numeric_dim < 8:
        raise ValueError("numeric_dim must expose the eight trailing FDC feature channels")
    generator = torch.Generator(device=device).manual_seed(seed)
    pair_count = (batch_size + 1) // 2
    base_numeric = 0.05 * torch.randn((pair_count, token_count, numeric_dim), generator=generator, device=device)
    base_features = torch.randint(len(FEATURE_TYPES), (pair_count,), generator=generator, device=device)
    sensor_count = token_count // patch_count
    first_sensors = torch.randint(sensor_count, (pair_count,), generator=generator, device=device)
    offsets = torch.randint(1, sensor_count, (pair_count,), generator=generator, device=device)
    second_sensors = (first_sensors + offsets) % sensor_count
    scopes = torch.randint(3, (pair_count,), generator=generator, device=device)
    feature_targets = base_features.repeat_interleave(2)[:batch_size]
    sensor_targets = torch.stack((first_sensors, second_sensors), dim=1).reshape(-1)[:batch_size]
    scope_targets = scopes.repeat_interleave(2)[:batch_size]
    channel_groups = ((numeric_dim - 8, numeric_dim - 7), (numeric_dim - 5, numeric_dim - 4),
                      (numeric_dim - 3,), (numeric_dim - 2, numeric_dim - 1))
    for pair in range(pair_count):
        feature = int(base_features[pair])
        for sensor in range(sensor_count):
            strength = 1.5 + 0.5 * torch.rand((), generator=generator, device=device)
            for channel in channel_groups[feature]:
                for patch in range(patch_count):
                    base_numeric[pair, sensor * patch_count + patch, channel] += strength
    numeric = base_numeric.repeat_interleave(2, 0)[:batch_size]
    feature_names = list(FEATURE_TYPES)
    questions: list[str] = []
    token_descriptions: list[list[str]] = []
    for index in range(batch_size):
        feature = int(feature_targets[index])
        pair = index // 2
        phase = f"training phase {seed}_{pair}"
        sensors = [f"sensor {seed}_{pair}_{sensor}" for sensor in range(sensor_count)]
        descriptions = [
            f"Process phase {phase}; sensor {sensor}; relative time {time_word}."
            for sensor in sensors for time_word in ("early interval", "middle interval", "late interval")
        ]
        token_descriptions.append(descriptions)
        questions.append(render_semantic_question(
            phase, sensors[int(sensor_targets[index])], feature_names[feature], "CHANGE",
            ("WHOLE", "EARLY", "LATE")[int(scope_targets[index])],
            split="train", variant=seed + index,
        ))
    token_targets = sensor_targets * patch_count + torch.where(scope_targets == 2, patch_count - 1, 0)
    targets: dict[str, torch.Tensor | int] = {
        "sensor": sensor_targets, "scope": scope_targets, "token": token_targets,
        "patch_count": patch_count,
    }
    return numeric, questions, ["CHOICE"] * batch_size, token_descriptions, feature_targets, targets


def cross_grounding_loss(attention: torch.Tensor, targets: dict[str, torch.Tensor | int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Supervise sensor identity and relative time without forcing WHOLE to one patch."""
    patch_count = int(targets["patch_count"])
    sensor_target = targets["sensor"]
    scope_target = targets["scope"]
    if not isinstance(sensor_target, torch.Tensor) or not isinstance(scope_target, torch.Tensor):
        raise TypeError("grounding targets must be tensors")
    grid = attention.reshape(attention.shape[0], -1, patch_count)
    sensor_mass = grid.sum(-1).gather(1, sensor_target[:, None]).squeeze(1)
    sensor_loss = -sensor_mass.clamp_min(1e-8).log().mean()
    target_patch_attention = grid[torch.arange(attention.shape[0], device=attention.device), sensor_target]
    scope_mask = torch.ones_like(target_patch_attention, dtype=torch.bool)
    scoped = scope_target != 0
    scope_mask[scoped] = False
    early_end = max(1, math.ceil(patch_count / 3))
    late_start = min(patch_count - 1, math.floor(2 * patch_count / 3))
    scope_mask[scope_target == 1, :early_end] = True
    scope_mask[scope_target == 2, late_start:] = True
    scope_mass = (target_patch_attention * scope_mask).sum(-1)
    scope_loss = -scope_mass.clamp_min(1e-8).log().mean()
    return 0.5 * (sensor_loss + scope_loss), sensor_loss, scope_loss


def real_step1_cross_batch(
    numeric: torch.Tensor,
    phase: str,
    sensor_names: Sequence[str],
    *,
    seed: int,
    split: str = "train",
    binding_refinement: bool = False,
    runtime_wording: bool = False,
    permute_sensors: bool = False,
    rename_sensors: bool = False,
) -> tuple[torch.Tensor, list[str], list[str], list[list[str]], torch.Tensor, dict[str, torch.Tensor | int]]:
    """Bind questions to frozen STEP 1 states without inventing fault labels.

    Each STEP 1 state is repeated with two different sensor questions and then
    expanded evenly across BOOLEAN, CHOICE and ORDINAL. Sensor and time targets
    come only from the declared question structure.
    """
    batch, token_count, _ = numeric.shape
    sensor_count = len(sensor_names)
    if sensor_count < 2 or token_count % sensor_count:
        raise ValueError("STEP 1 tokens must be sensor-major with at least two sensors")
    patch_count = token_count // sensor_count
    generator = torch.Generator(device=numeric.device).manual_seed(seed)
    if permute_sensors:
        permutation=torch.randperm(sensor_count,generator=generator,device=numeric.device)
        numeric=numeric.reshape(batch,sensor_count,patch_count,-1).index_select(1,permutation).reshape_as(numeric)
        sensor_names=[sensor_names[i] for i in permutation.cpu().tolist()]
    if rename_sensors:
        # Coherent identifier substitution, no anomalous/normal labels. Applies
        # to both question and token descriptor, never to evaluation labels only.
        sensor_names=[f'channel_{seed}_{i:03d}_signal' for i in range(sensor_count)]
    first = torch.randint(sensor_count, (batch,), generator=generator, device=numeric.device)
    offset = torch.randint(1, sensor_count, (batch,), generator=generator, device=numeric.device)
    second = (first + offset) % sensor_count
    if binding_refinement:
        # Half hard (lexically confusable), half random distractors. No numeric
        # anomaly labels, amplitude shortcuts or fixed sensor index targets.
        nearest = [max((j for j in range(sensor_count) if j != i),
                       key=lambda j: SequenceMatcher(None, str(sensor_names[i]), str(sensor_names[j])).ratio())
                   for i in range(sensor_count)]
        hard = torch.tensor(nearest, device=numeric.device)[first]
        second = torch.where(torch.arange(batch, device=numeric.device) % 2 == 0, hard, second)
    base_sensor_target = torch.stack((first, second), 1).reshape(-1)
    scope_base = torch.randint(3, (batch,), generator=generator, device=numeric.device)
    base_scope_target = scope_base.repeat_interleave(2)
    feature_base = torch.randint(len(FEATURE_TYPES), (batch,), generator=generator, device=numeric.device)
    base_feature_target = feature_base.repeat_interleave(2)
    direction_base = torch.randint(3, (batch,), generator=generator, device=numeric.device)
    base_direction_target = direction_base.repeat_interleave(2)
    base_numeric = numeric.repeat_interleave(2, 0)
    group_size = 2
    if binding_refinement:
        # Per state: [A whole, B whole, A early, B early, A late, B late].
        # Sensor comparisons differ only in sensor; scope comparisons only in time.
        group_size = 6
        base_sensor_target = base_sensor_target.reshape(batch, 2).repeat(1, 3).reshape(-1)
        base_scope_target = torch.arange(3, device=numeric.device).repeat_interleave(2).repeat(batch)
        base_feature_target = feature_base.repeat_interleave(group_size)
        base_direction_target = direction_base.repeat_interleave(group_size)
        base_numeric = numeric.repeat_interleave(group_size, 0)
    answer_types = [kind for kind in ANSWER_TYPES for _ in range(group_size * batch)]
    repeated = base_numeric.repeat((len(ANSWER_TYPES), 1, 1))
    sensor_target = base_sensor_target.repeat(len(ANSWER_TYPES))
    scope_target = base_scope_target.repeat(len(ANSWER_TYPES))
    feature_target = base_feature_target.repeat(len(ANSWER_TYPES))
    direction_target = base_direction_target.repeat(len(ANSWER_TYPES))
    scope_words = [
        "early interval" if patch < patch_count / 3 else
        "late interval" if patch >= 2 * patch_count / 3 else "middle interval"
        for patch in range(patch_count)
    ]
    one_description = token_descriptions(phase,sensor_names,patch_count)
    descriptions = [one_description for _ in range(len(answer_types))]
    feature_names = list(FEATURE_TYPES)
    questions = [render_semantic_question(
        phase, sensor_names[int(sensor_target[row])], feature_names[int(feature_target[row])],
        ("INCREASE", "DECREASE", "CHANGE")[int(direction_target[row])],
        ("WHOLE", "EARLY", "LATE")[int(scope_target[row])],
        split=split, variant=seed + row // group_size if binding_refinement else seed + row,
        answer_type=answer_types[row],
    ) for row in range(len(answer_types))]
    if runtime_wording:
        from .experiments.questions import make_question
        stat_map={'LEVEL':'level','VARIABILITY':'variability','POINT':'local_change','RESIDUAL':'residual'}
        questions=[make_question(phase,sensor_names[int(sensor_target[row])],answer_types[row],
                    stat_map[feature_names[int(feature_target[row])]],
                    ('increase','decrease','either')[int(direction_target[row])],
                    ('whole','early','late')[int(scope_target[row])],split=split,variant=seed+row//group_size)
                   for row in range(len(answer_types))]
    early_token = torch.zeros_like(scope_target)
    late_token = torch.full_like(scope_target, max(0, math.floor(2 * patch_count / 3)))
    patch_target = torch.where(scope_target == 2, late_token, early_token)
    targets: dict[str, torch.Tensor | int] = {
        "sensor": sensor_target,
        "scope": scope_target,
        "direction": direction_target,
        "token": sensor_target * patch_count + patch_target,
        "patch_count": patch_count,
        "group_size": group_size,
    }
    return repeated, questions, answer_types, descriptions, feature_target, targets


def refined_cross_grounding_loss(attention, targets, *, margin=0.5):
    """Train binding only; no answer/fault targets and no architecture changes.

    Scope KL is conditional on the declared sensor, separating identity and time
    objectives. WHOLE spreads over all patches instead of rewarding one spike.
    A log-mass margin separates the target from the strongest competing sensor.
    """
    if margin < 0:
        raise ValueError('margin must be nonnegative')
    patch_count = int(targets['patch_count'])
    grid = attention.reshape(len(attention), -1, patch_count).clamp_min(1e-8)
    grid = grid / grid.sum((1, 2), keepdim=True)
    sensor = targets['sensor']; scope = targets['scope']
    scores = grid.sum(-1)
    rows = torch.arange(len(grid), device=grid.device)
    target = scores[rows, sensor]
    competitor = scores.masked_fill(F.one_hot(sensor, scores.shape[1]).bool(), 0).max(-1).values
    selected = grid[rows, sensor] / target[:, None]
    mask = torch.ones_like(selected)
    early_end = max(1, math.ceil(patch_count / 3))
    late_start = min(patch_count - 1, math.floor(2 * patch_count / 3))
    mask[scope == 1, early_end:] = 0
    mask[scope == 2, :late_start] = 0
    distribution = mask / mask.sum(-1, keepdim=True)
    losses = {
        'sensor_loss': -target.log().mean(),
        'scope_loss': (distribution * (distribution.clamp_min(1e-8).log() - selected.log())).sum(-1).mean(),
        'sensor_margin_loss': F.relu(margin - target.log() + competitor.clamp_min(1e-8).log()).mean(),
    }
    # Compare early/late masses explicitly on same-state, same-sensor contrasts.
    if int(targets.get('group_size', 2)) == 6:
        masses = selected.reshape(-1, 3, 2, patch_count)
        early = masses[:, 1]; late = masses[:, 2]
        early_right = early[..., :early_end].sum(-1).clamp_min(1e-8)
        early_wrong = early[..., late_start:].sum(-1).clamp_min(1e-8)
        late_right = late[..., late_start:].sum(-1).clamp_min(1e-8)
        late_wrong = late[..., :early_end].sum(-1).clamp_min(1e-8)
        losses['scope_contrast_loss'] = .5 * (
            F.relu(margin - early_right.log() + early_wrong.log()).mean()
            + F.relu(margin - late_right.log() + late_wrong.log()).mean())
    else:
        losses['scope_contrast_loss'] = attention.sum() * 0
    return losses


def joint_binding_loss(attention,targets):
    """Direct sensor AND allowed-time mass; no conditional-only success."""
    patches=int(targets['patch_count']); grid=attention.reshape(len(attention),-1,patches)
    grid=grid/grid.sum((1,2),keepdim=True).clamp_min(1e-8)
    selected=grid[torch.arange(len(grid),device=grid.device),targets['sensor']]
    mask=torch.ones_like(selected)
    mask[targets['scope']==1,max(1,math.ceil(patches/3)):]=0
    mask[targets['scope']==2,:min(patches-1,math.floor(2*patches/3))]=0
    return -(selected.mul(mask).sum(-1).clamp_min(1e-8)).log().mean()


def binding_paraphrases(phase,descriptions,feature_targets,targets,answer_types,*,variant=1):
    from .experiments.questions import make_question
    names=('level','variability','local_change','residual')
    result=[]; patches=int(targets['patch_count'])
    for i,kind in enumerate(answer_types):
        sensor=int(targets['sensor'][i]); descriptor=descriptions[i][sensor*patches]
        sensor_name=descriptor.split('; sensor ',1)[1].split('; relative time ',1)[0]
        # Names and target IDs appear only in TRAIN supervision, never routing.
        # Runtime-style BOOLEAN needs a direction; hold it identical to the
        # original generated question rather than add an arbitrary new direction.
        direction=int(targets['direction'][i])
        result.append(make_question(phase,sensor_name,kind,names[int(feature_targets[i])],
            ('increase','decrease','either')[direction],('whole','early','late')[int(targets['scope'][i])],
            split='train',variant=variant))
    return result


def attention_consistency_loss(first,second):
    a=first.clamp_min(1e-8); b=second.clamp_min(1e-8)
    a=a/a.sum(-1,keepdim=True); b=b/b.sum(-1,keepdim=True)
    return .5*((a*(a.log()-b.log())).sum(-1)+(b*(b.log()-a.log())).sum(-1)).mean()


def evidence_questions(phase, count):
    """Shared TRAIN/runtime generic localization prompt, no injected IDs."""
    from .experiments.questions import localization_questions
    return localization_questions(phase, count)


def numeric_evidence_pairs(current, patch_length, *, seed):
    """Controlled numeric contrasts on frozen STEP 1 inputs, not fault labels.

    Same question and same source waveform; a localized change moves between
    sensors and patches. The source waveform itself is the comparison baseline,
    so pre-existing cross-run differences cannot corrupt positional truth.
    Inputs are standardized using TRAIN constants before this function.
    """
    if current.ndim!=3 or current.shape[-1]<2 or patch_length<1 or current.shape[1]%patch_length:
        raise ValueError('Time/sensor windows with at least two sensors and full patches required')
    batch,length,sensors=current.shape; patches=length//patch_length
    if patches<2: raise ValueError('At least two patches required for evidence contrasts')
    g=torch.Generator(device=current.device).manual_seed(seed)
    first=torch.randint(sensors,(batch,),generator=g,device=current.device)
    second=(first+torch.randint(1,sensors,(batch,),generator=g,device=current.device))%sensors
    sensor_ids=torch.stack((first,second),1).flatten()
    patch_ids=torch.randint(patches,(2*batch,),generator=g,device=current.device)
    changed=current.detach().repeat_interleave(2,0).clone()
    reference=current.detach().repeat_interleave(2,0).clone()
    for row in range(2*batch):
        start=int(patch_ids[row])*patch_length; end=start+patch_length
        amplitude=2.+2.*float(torch.rand((),generator=g,device=current.device))
        kind=(seed+row//2)%4
        if kind==0: changed[row,start:end,sensor_ids[row]]+=amplitude
        elif kind==1: changed[row,start+patch_length//2,sensor_ids[row]]+=amplitude
        elif kind==2: changed[row,start:end,sensor_ids[row]]+=amplitude*torch.randn(patch_length,generator=g,device=current.device)
        else: changed[row,start:end,sensor_ids[row]]+=torch.linspace(0,amplitude,patch_length,device=current.device)
    return changed,reference,{'sensor':sensor_ids,'patch':patch_ids,'patch_count':patches}


def numeric_evidence_loss(attention, targets):
    """Attention must move with numeric evidence, not a named sensor shortcut."""
    patches=int(targets['patch_count'])
    grid=attention.reshape(len(attention),-1,patches).clamp_min(1e-8)
    grid=grid/grid.sum((1,2),keepdim=True)
    rows=torch.arange(len(grid),device=grid.device)
    sensor_mass=grid.sum(-1)
    selected=sensor_mass[rows,targets['sensor']]
    if 'patch_mask' in targets:
        mask=targets['patch_mask'].to(grid.dtype)
        target=(grid[rows,targets['sensor']]*mask).sum(-1)
    else:
        target=grid[rows,targets['sensor'],targets['patch']]
    loss=-.5*(selected.log()+target.log()).mean()
    sensor_ok=sensor_mass.argmax(-1).eq(targets['sensor'])
    predicted_sensor=sensor_mass.argmax(-1)
    predicted_patch=grid[rows,predicted_sensor].argmax(-1)
    token_ok=(sensor_ok & targets['patch_mask'][rows,predicted_patch]) if 'patch_mask' in targets else grid.flatten(1).argmax(-1).eq(targets['sensor']*patches+targets['patch'])
    return loss,{'evidence_sensor_accuracy':float(sensor_ok.float().mean()),
                 'evidence_token_accuracy':float(token_ok.float().mean())}


def reference_evidence_pairs(current, reference, patch_length, *, seed, background='mixed', temporal=False):
    """Paired injected sensors on real cross-run backgrounds plus self controls.

    Half the source pairs use the actual reference; half use the source itself.
    Truth marks the injected change, never declares background runs fault-free.
    Paired interventions share kind, strength and interval and differ in sensor.
    """
    if current.shape!=reference.shape or current.ndim!=3:
        raise ValueError('Aligned current/reference windows required')
    if background not in ('mixed','cross','self'):
        raise ValueError('Unknown reference background')
    batch,length,sensors=current.shape
    if sensors<2 or patch_length<1 or length%patch_length or length//patch_length<2:
        raise ValueError('At least two sensors and two complete patches required')
    if not torch.isfinite(current).all() or not torch.isfinite(reference).all():
        raise ValueError('Finite standardized inputs required')
    g=torch.Generator(device=current.device).manual_seed(seed)
    patches=length//patch_length
    first=torch.randint(sensors,(batch,),generator=g,device=current.device)
    second=(first+torch.randint(1,sensors,(batch,),generator=g,device=current.device))%sensors
    sensor_ids=torch.stack((first,second),1).flatten()
    changed=current.detach().repeat_interleave(2,0).clone()
    source=reference.detach().repeat_interleave(2,0).clone()
    mask=torch.zeros(2*batch,patches,dtype=torch.bool,device=current.device)
    cross_run=torch.zeros(2*batch,dtype=torch.bool,device=current.device)
    for pair in range(batch):
        use_cross=background=='cross' or (background=='mixed' and pair%2==0)
        start=int(torch.randint(length,(1,),generator=g,device=current.device))
        duration=int(torch.randint(1,length-start+1,(1,),generator=g,device=current.device))
        end=start+duration
        amplitude=float(torch.exp(torch.empty((),device=current.device).uniform_(math.log(.5),math.log(4.),generator=g)))
        sign=1 if int(torch.randint(2,(1,),generator=g,device=current.device)) else -1
        kind=int(torch.randint(6 if temporal else 4,(1,),generator=g,device=current.device))
        delta=torch.zeros(length,device=current.device,dtype=current.dtype)
        if kind==0:delta[start:end]=sign*amplitude
        elif kind==1:delta[start+duration//2]=sign*amplitude
        elif kind==2:delta[start:end]=amplitude*torch.randn(duration,generator=g,device=current.device)
        else:delta[start:end]=sign*amplitude*torch.linspace(1/duration,1,duration,device=current.device)
        temporal_deltas=[]
        if kind>=4:
            lag=int(torch.randint(1,min(4,length-1)+1,(1,),generator=g,device=current.device))
            for row in (2*pair,2*pair+1):
                waveform=current[pair,:,sensor_ids[row]]
                replacement=(waveform[(torch.arange(start,end,device=current.device)-lag).clamp_min(0)]
                             if kind==4 else waveform[start].expand(duration))
                local=torch.zeros_like(waveform);local[start:end]=replacement-waveform[start:end]
                temporal_deltas.append(local)
            # A flat signal cannot supply delay/stuck localization truth.
            # Fall back for BOTH members, preserving a matched effective pair.
            if any(not bool(d.ne(0).any()) for d in temporal_deltas):
                temporal_deltas=[]
        for row in (2*pair,2*pair+1):
            actual_delta=temporal_deltas[row%2] if temporal_deltas else delta
            changed[row,:,sensor_ids[row]]+=actual_delta
            if not use_cross:source[row]=current[pair]
            cross_run[row]=use_cross
            mask[row]=actual_delta.reshape(patches,patch_length).ne(0).any(-1)
    return changed,source,{'sensor':sensor_ids,'patch':mask.long().argmax(-1),
        'patch_count':patches,'patch_mask':mask,'cross_run':cross_run}


def cross_attention_batch_metrics(
    attention: torch.Tensor,
    targets: dict[str, torch.Tensor | int],
) -> dict[str, float]:
    patch_count = int(targets["patch_count"])
    sensor_target = targets["sensor"]
    scope_target = targets["scope"]
    if not isinstance(sensor_target, torch.Tensor) or not isinstance(scope_target, torch.Tensor):
        raise TypeError("grounding targets must be tensors")
    grid = attention.reshape(attention.shape[0], -1, patch_count)
    sensor_scores = grid.sum(-1)
    predicted_sensor = sensor_scores.argmax(-1)
    target_patch = grid[torch.arange(len(grid), device=grid.device), sensor_target].argmax(-1)
    early_end = max(1, math.ceil(patch_count / 3))
    late_start = min(patch_count - 1, math.floor(2 * patch_count / 3))
    scope_ok = (scope_target == 0) | ((scope_target == 1) & (target_patch < early_end)) | ((scope_target == 2) & (target_patch >= late_start))
    target_score = sensor_scores.gather(1, sensor_target[:, None]).squeeze(1)
    competitor = sensor_scores.masked_fill(F.one_hot(sensor_target, sensor_scores.shape[1]).bool(), -1).max(-1).values
    pairs = len(grid) // 2
    paired_prediction = predicted_sensor[:2 * pairs].reshape(-1, 2)
    paired_target = sensor_target[:2 * pairs].reshape(-1, 2)
    result = {
        "sensor_top1_accuracy": float(predicted_sensor.eq(sensor_target).float().mean()),
        "scope_top1_accuracy": float(scope_ok.float().mean()),
        "counterfactual_sensor_shift_accuracy": float(paired_prediction.eq(paired_target).all(-1).float().mean()) if pairs else 0.0,
        "mean_sensor_attention_margin": float((target_score - competitor).mean()),
    }
    joint = predicted_sensor.eq(sensor_target) & scope_ok
    scoped = scope_target != 0
    result['joint_sensor_scope_accuracy'] = float(joint.float().mean())
    result['scoped_joint_accuracy'] = float(joint[scoped].float().mean()) if scoped.any() else 0.0
    if int(targets.get('group_size', 2)) == 6:
        result['counterfactual_scope_shift_accuracy'] = float(joint.reshape(-1, 3, 2)[:, 1:].all(1).float().mean())
    else:
        result['counterfactual_scope_shift_accuracy'] = 0.0
    return result


def train_cross_attention(
    backbone: GEVISBackbone,
    heads: BackbonePretrainHeads,
    device: torch.device,
    *,
    steps: int,
    batch_size: int,
    token_count: int,
    numeric_dim: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    text_requires_grad = [parameter.requires_grad for parameter in backbone.text.parameters()]
    for parameter in backbone.text.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(backbone.numeric.parameters())
        + list(backbone.type_emb.parameters())
        + list(backbone.cross.parameters())
        + list(backbone.norm.parameters())
        + list(heads.feature_head.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    backbone.train(); backbone.text.eval(); heads.train(); history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        numeric, questions, types, token_descriptions, feature_target, targets = controlled_numeric_batch(
            batch_size, token_count, numeric_dim, device, seed + step
        )
        fused, attention = backbone(numeric, questions, types, token_descriptions)
        feature_loss = F.cross_entropy(heads.feature_head(fused), feature_target)
        localization_loss, sensor_loss, scope_loss = cross_grounding_loss(attention, targets)
        loss = feature_loss + localization_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step == 1 or step % 20 == 0 or step == steps:
            history.append({"step": step, "loss": float(loss.detach()), "feature_loss": float(feature_loss.detach()), "localization_loss": float(localization_loss.detach()), "sensor_loss": float(sensor_loss.detach()), "scope_loss": float(scope_loss.detach())})
    for parameter, requires_grad in zip(backbone.text.parameters(), text_requires_grad):
        parameter.requires_grad_(requires_grad)
    return history


@torch.no_grad()
def evaluate_cross_attention(
    backbone: GEVISBackbone,
    heads: BackbonePretrainHeads,
    device: torch.device,
    *,
    batch_size: int,
    token_count: int,
    numeric_dim: int,
    seed: int,
) -> dict[str, float | list[int] | bool]:
    backbone.eval(); heads.eval()
    numeric, questions, types, token_descriptions, feature_target, targets = controlled_numeric_batch(
        batch_size, token_count, numeric_dim, device, seed
    )
    fused, attention = backbone(numeric, questions, types, token_descriptions)
    token_target = targets["token"]
    sensor_target = targets["sensor"]
    scope_target = targets["scope"]
    patch_count = int(targets["patch_count"])
    assert isinstance(token_target, torch.Tensor) and isinstance(sensor_target, torch.Tensor) and isinstance(scope_target, torch.Tensor)
    top_k = min(5, token_count)
    attention_grid = attention.reshape(batch_size, -1, patch_count)
    sensor_scores = attention_grid.sum(-1)
    predicted_sensor = sensor_scores.argmax(-1)
    selected_patch = attention_grid[torch.arange(batch_size, device=device), sensor_target].argmax(-1)
    predicted_scope = torch.where(selected_patch == 0, 1, torch.where(selected_patch == patch_count - 1, 2, 0))
    expected_patch = torch.where(scope_target == 2, patch_count - 1, 0)
    allowed_top1 = predicted_sensor.eq(sensor_target) & ((scope_target == 0) | selected_patch.eq(expected_patch))
    pair_count = batch_size // 2
    counterfactual = predicted_sensor[:pair_count * 2].reshape(-1, 2)
    counterfactual_target = sensor_target[:pair_count * 2].reshape(-1, 2)
    target_score = sensor_scores.gather(1, sensor_target[:, None]).squeeze(1)
    competing_score = sensor_scores.masked_fill(F.one_hot(sensor_target, sensor_scores.shape[1]).bool(), -1).max(-1).values
    return {
        "feature_accuracy": float(heads.feature_head(fused).argmax(-1).eq(feature_target).float().mean()),
        "token_top1_accuracy": float(attention.argmax(-1).eq(token_target).float().mean()),
        "token_top5_accuracy": float(attention.topk(top_k, dim=-1).indices.eq(token_target[:, None]).any(-1).float().mean()),
        "sensor_top1_accuracy": float(predicted_sensor.eq(sensor_target).float().mean()),
        "scope_top1_accuracy": float(((scope_target == 0) | predicted_scope.eq(scope_target)).float().mean()),
        "allowed_token_top1_accuracy": float(allowed_top1.float().mean()),
        "counterfactual_sensor_shift_accuracy": float(counterfactual.eq(counterfactual_target).all(-1).float().mean()) if pair_count else 0.0,
        "mean_sensor_attention_margin": float((target_score - competing_score).mean()),
        "fused_shape": list(fused.shape),
        "attention_shape": list(attention.shape),
        "finite": bool(torch.isfinite(fused).all() and torch.isfinite(attention).all()),
    }


def grounded_numeric_batch(pairs, device, seed, *, split='train', numeric_dim=264):
    """Synthetic FDC channels with competing sensors; opposite queries share state.

    LEVEL/VARIABILITY/RESIDUAL use signed current-reference operations. This
    separate numeric curriculum does not assert slope/point grounding quality.
    """
    if numeric_dim != 264 or pairs < 1:
        raise ValueError('Grounding curriculum requires positive pairs and 264-channel contract')
    rng = random.Random(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    numeric = torch.zeros(pairs, 6, numeric_dim, device=device)
    deltas = (torch.rand((pairs, 2, 3, 3), generator=generator, device=device) * 2 - 1) * 1.8
    numeric[..., 256] = deltas[..., 0].reshape(pairs, 6)
    numeric[..., 258] = numeric[..., 256].abs()
    numeric[..., 259] = 2 + deltas[..., 1].reshape(pairs, 6)
    numeric[..., 260] = 2
    numeric[..., 261] = numeric[..., 258]
    numeric[..., 262] = 2 + deltas[..., 2].reshape(pairs, 6)
    numeric[..., 263] = 2
    questions, descriptions, labels, masks = [], [], [], []
    stats = ('LEVEL', 'VARIABILITY', 'RESIDUAL')
    for row in range(pairs):
        phase = f'{split}_phase_{rng.randrange(10**9):09d}'
        sensors = [f'{split}_sensor_{rng.randrange(10**9):09d}' for _ in range(2)]
        sensor, stat, scope = row % 2, (row // 2) % 3, (row // 6) % 3
        scopes = (0, 1, 2) if scope == 0 else ((0,) if scope == 1 else (2,))
        value = deltas[row, sensor, list(scopes), stat].mean()
        mask = torch.zeros(6, device=device); mask[[sensor*3+p for p in scopes]] = 1
        token_descriptions = [f'Process phase {phase}; measurement {name}; relative interval {scope_word}' for name in sensors for scope_word in ('early', 'middle', 'late')]
        for selected_sensor in (sensor, 1-sensor):
            value = deltas[row, selected_sensor, list(scopes), stat].mean()
            mask = torch.zeros(6, device=device); mask[[selected_sensor*3+p for p in scopes]] = 1
            for direction, sign in (('INCREASE', 1.), ('DECREASE', -1.)):
                questions.append(render_semantic_question(phase, sensors[selected_sensor], stats[stat], direction, ('WHOLE','EARLY','LATE')[scope], split=split, variant=row))
                labels.append(int(float(value) * sign > .25))
                descriptions.append(token_descriptions)
                masks.append(mask)
    return numeric.repeat_interleave(4, 0), questions, ['BOOLEAN'] * (4*pairs), descriptions, torch.tensor(labels, device=device), torch.stack(masks)


@torch.no_grad()
def evaluate_grounding(backbone, heads, device, *, split='validation', seed=9091, pairs=36):
    backbone.eval(); heads.eval()
    numeric, questions, types, descriptions, labels, masks = grounded_numeric_batch(pairs, device, seed, split=split)
    fused, attention = backbone(numeric, questions, types, descriptions)
    prediction = heads.grounding_head(fused).argmax(-1)
    pair_labels = labels.reshape(-1, 2); pair_pred = prediction.reshape(-1, 2)
    changed = pair_labels[:, 0] != pair_labels[:, 1]
    return {'split': split, 'numeric_answer_accuracy': float(prediction.eq(labels).float().mean()),
            'numeric_balanced_accuracy': float(torch.stack([prediction[labels==cls].eq(cls).float().mean() for cls in (0,1) if (labels==cls).any()]).mean()),
            'changed_question_pair_accuracy': float(pair_pred[changed].eq(pair_labels[changed]).all(-1).float().mean()) if changed.any() else None,
            'target_scope_attention_mass': float((attention*masks).sum(-1).mean()),
            'pairs_with_changed_answer': int(changed.sum()),
            'operations': ['LEVEL', 'VARIABILITY', 'RESIDUAL']}


def train_grounding(backbone, heads, device, *, steps=120, pairs=12, learning_rate=1e-4, seed=8001):
    """Offline fusion training; select with validation only, never test/audit."""
    flags = [p.requires_grad for p in backbone.text.parameters()]
    backbone.text.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in backbone.parameters() if p.requires_grad] + list(heads.grounding_head.parameters()), lr=learning_rate)
    best_score, best_state, history = float('-inf'), None, []
    try:
        for step in range(1, steps+1):
            backbone.train(); backbone.text.eval(); heads.train()
            numeric, questions, types, descriptions, labels, masks = grounded_numeric_batch(pairs, device, seed+step)
            fused, attention = backbone(numeric, questions, types, descriptions)
            loss = F.cross_entropy(heads.grounding_head(fused), labels) - .2 * (attention*masks).sum(-1).clamp_min(1e-8).log().mean()
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(backbone.parameters(), 1.); optimizer.step()
            if step == 1 or step % 20 == 0 or step == steps:
                metrics = evaluate_grounding(backbone, heads, device)
                score = metrics['numeric_answer_accuracy'] + metrics['target_scope_attention_mass']
                history.append({'step': step, 'loss': float(loss.detach()), **metrics})
                if score > best_score:
                    best_score = score
                    best_state = (deepcopy(backbone.state_dict()), deepcopy(heads.grounding_head.state_dict()))
        if best_state is None: raise ValueError('Grounding steps must be positive')
        backbone.load_state_dict(best_state[0]); heads.grounding_head.load_state_dict(best_state[1])
    finally:
        for p, flag in zip(backbone.text.parameters(), flags): p.requires_grad_(flag)
    return history
