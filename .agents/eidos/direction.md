---
eidos_version: 3
document_type: direction
project_id: project:pecvd
revision: D0011
updated_at: 2026-10-21
---

# PECVD Direction

## Purpose

### Active objective: fixed questions, selected sensor, learned Fusion answers

This PECVD-test workspace restores the neural answer objective. A user selects a
process/run, one Step occurrence, and an analysis-center sensor. Keep every
context sensor and the current/reference numeric inputs. The existing English
BOOLEAN/CHOICE/ORDINAL question strings stay fixed and contain no sensor or Step
identifier. Run/Step select the data; selected-center role belongs to numeric
context and an explicit selection mask, never to question wording or answer labels.
The process_key identifies a configured process occurrence; without an explicit
wafer ID, do not certify its equivalence to a physical wafer identity.

Current/reference traces -> STEP1 tokens -> question-conditioned STEP2 Fusion ->
selected-center conditioned learned answer head -> logits/softmax/answer. Final
answers must come from the trained head. q99, GMM, hand-calculated operation
thresholds and rule-generated display probabilities are baselines only. Missing
or incompatible neural weights withhold answers; never fall back to a numeric
rule or use a random untrained head. Preserve existing encoders and Fusion.

STEP1 retains reconstruction pretraining. STEP2 retains question/numeric Fusion
pretraining. STEP3 fits the contextual answer readout using TRAIN only and selects
trained weights on independent Validation. STEP3.1 loads and predicts only. STEP4
measures actual head answers on held-out Test. Different selected centers on the
same waveform must receive independently derived targets. Fixed questions do not
mean one target vector per answer family. Include off-center changes and paired
center swaps; report their joint correctness with per-question confusion and
majority baselines. Test cannot fit weights, thresholds or select a default model.

Controlled paired perturbations establish a synthetic comparison task, not known
equipment fault truth. Self-reference controls have exact known equality; original
cross-run comparisons remain unlabelled unless independent truth is supplied.
Report this domain gap explicitly. Ordinal synthetic perturbation strength is
not physical severity. Attention alone is not correctness or causal evidence.

Evaluate the selected Step's complete timeline by explicit model windows; never
silently truncate to its last window. Per-window answers remain learned outputs;
do not introduce an untrained hand-written whole-Step alarm aggregation rule.
Never mix runs or separate occurrences of the same Step in one model window.

This objective supersedes D0010's numeric-first deployment and CE-head deferral.
The preserved numerical implementation is a comparator, not this model's answer.
Existing artifacts and the source CSVs remain unchanged. Use a new source-bound
checkpoint namespace. No private quality improvement is claimed without the
user's local measurements. The wider purpose remains process-independent typed
FDC review, not a PECVD name/class shortcut or certified fault diagnosis.

## Boundaries

- Included scope: preserved process CSVs, reproducible run-level comparisons,
  question/choice sets, transparent baselines, A/B/C model experiments, and
  engineer-facing evidence plots and latency measurements. The agent may run
  bounded, read-only local analysis of confidential project data while keeping
  raw values and derived artifacts out of retained records and external services.
- Excluded scope: declaring confirmed equipment faults from unlabelled sensor
  differences, treating rule-derived labels as proof of novel anomaly detection,
  assuming an unseen question is answered correctly without evaluation, and
  encoding PECVD Step names or sensor identities as universal answer semantics.
  Raw or sampled rows and unapproved data-derived outputs must not be copied
  into chat, Work, Failure, documentation, or an external service.
- Scope requiring separate approval: rewriting or deleting source CSVs,
  publishing data or derived artifacts externally, changing a shared
  data/output contract, or retaining a data-derived conclusion in project records.
- GEVIS answer quality means selecting a supplied typed criterion from grounded
  numeric evidence, returning evidence and calibrated coverage, and abstaining
  when the requested operation or evidence is unavailable. It does not mean
  unconstrained free-text generation.

## Baseline

The repository contains process CSVs, a combined CSV, and a manifest. Local
read-only inspection is permitted under the confidential-data boundary, but
does not authorize mutation, publication, or retained raw-data evidence. Process semantics, verified anomaly
cases, production labels, and downstream consumers are not established.
Harness/Eidos installation and scoped Work are active; no private-data product
test suite exists. The predictive TCN attempt was withdrawn after it did not
provide a useful baseline.

O's local experiments now include step-specific PatchTST-style masked
reconstruction (Step A), a small typed FDC model over frozen A representations
(Step B), and checkpoint/inference assembly (Step C). A synthetic Streamlit UI
demonstrates the intended question flow. These experiments show that the code
paths can train and run, but do not establish calibrated real-process decision
quality, reliable evidence localization, or reduced engineer review time. Step
C is not presently a separately trained larger language model. The bounded
design and experiment history are in `_blueprint/experiment-plan.md`.

The current Step B text objective primarily measures answer-type and support
classification. That is insufficient evidence that GEVIS distinguishes two
questions of the same answer type, binds a question to the requested numeric
operation, or transfers to an unseen process family. The next baseline must add
process-name-independent semantic slots, within-type hard negatives, grounded
question/numeric alignment, and fail-closed evaluation while retaining the
current tiny text encoder as the comparison baseline.

## Stages

### S01 — Establish the operating baseline

- Status: done
- Outcome: Harness/Eidos are installed, a human owner is registered, the
  preserved data inventory is recorded, and scoped Works are completed.
- Depends on: none

### S02 — Define the experimental protocol

- Status: active
- Outcome: Define comparable questions and choices, run-level splits,
  reference baselines, review labels or proxies, and measures of quality/cost.
  Data audits run locally and retain only an O-approved safe summary.
- Depends on: S01

### S03 — Establish the numeric state encoder

- Status: active
- Outcome: Establish the reproducible step-aware PatchTST representation and a
  transparent statistical reference on held-out runs. Define the current versus
  reference state interface without treating reconstruction loss as a fault.
- Depends on: S02

### S04 — Test typed FDC decisions

- Status: active
- Outcome: Combine numeric state and question text to support `NOUL`, `CHOICE`,
  and ordinal `SCORE` without treating process Step or sensor names as answer
  classes. Return FDC evidence, reference provenance, and coverage; test
  within-type meaning changes, hard negatives, unseen identifiers, held-out
  process-family contracts, calibration, abstention, utility, and runtime.
- Depends on: S03

### S05 — Test C only if justified

- Status: planned
- Outcome: Assess a Kev-style Qwen/LoRA choice scorer against B after the
  numeric time-series interface, labels, resources, and expected benefit are clear.
- Depends on: S04

### S06 — Decide practical use

- Status: planned
- Outcome: Decide which method, if any, shortens correct human review enough
  to justify false alerts, missed differences, maintenance, and compute cost.
- Depends on: S04, S05

## Gates

### G01 — Operating baseline recorded

- Stage: S01
- Criteria: Kit validation passes, member:o is active, and the data inventory
  and at least one scoped Work have retained evidence.
- Evidence: Installed-tool validation, `_evidence/adoption-baseline.md`, and
  completed Work records under `.agents/eidos/work/`.

### G02 — Protocol ready

- Stage: S02
- Criteria: Question/choice definitions, supported input context, run-level
  split, label provenance, and metrics are explicit before model comparison;
  confidential inputs remain local and outputs follow the disclosure boundary.
- Evidence: Reviewed protocol and held-out evaluation plan in Work; raw data
  audit evidence stays local, with only an O-approved safe summary recorded.

### G03 — Baselines comparable

- Stage: S03
- Criteria: Statistical checks and the PatchTST state encoder use identical
  held-out runs; the representation, reference interface, latency, and observed
  limitations are reproducible without unsupported decision claims.
- Evidence: Reproducible code/configuration and synthetic tests in Work; actual
  outputs and plots remain local to O unless explicitly cleared for sharing.

### G04 — B adds measurable value

- Stage: S04
- Criteria: B supports the three typed primitives; distinguishes target,
  statistic, comparison, direction, time scope, wording, and unseen
  combinations within the same answer type; does not rely on fixed process Step
  or sensor names; and reports calibrated decisions plus grounded FDC evidence
  and coverage limitations. A leave-one-process-contract-out synthetic test and
  fail-closed unsupported-operation test are required before any transfer claim.
- Evidence: Local paired evaluation and error analysis; Work stores only
  approved non-sensitive conclusions and synthetic-test evidence.

### G05 — C has a fair comparison

- Stage: S05
- Criteria: C receives the same supported context and questions as B; any
  benefit is weighed against memory, latency, training, and maintenance cost.
- Evidence: Locally reproducible trial or a recorded decision to skip C, with
  no raw data or unapproved outputs copied into Work.

### G06 — Use decision supported

- Stage: S06
- Criteria: Claims are limited to demonstrated review assistance and name
  unresolved label, fault-causality, and generalization limits.
- Evidence: O-reviewed comparative report and explicit decision; only
  O-approved non-sensitive findings enter shared project records.

## Dependencies

Expert review or verified fault cases, the meaning of each process step, intended
questions and choices, and the available training budget remain open inputs.
Data-processing code and results remain local; the agent may inspect them
read-only but must not retain raw values or unapproved derived conclusions.
Cross-run questions require multiple runs as input. The typed decision layer
requires reference answers or explicitly identified proxies; labels from rules
or controlled perturbations prove rule reproduction, not real fault detection.
S03 depends on the S02 protocol; C depends on evidence that B leaves a useful gap. Stage and sensor
names alone do not establish comparable recipes or causal fault diagnoses.
