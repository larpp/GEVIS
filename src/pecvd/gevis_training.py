"""Notebook-independent entry points for domain head training and evaluation."""
from types import SimpleNamespace

from .experiments.ald_preprocessing import NativeConfig, PreprocessingConfig, schema_inventory, transform_live, sensor_roles, canonicalize_run_lineages
from .experiments.ald_preprocessing import prepare as _prepare_ald
from .experiments.decision_training import Config as TrainingConfig
from .experiments.ald_head_experiments import ChoiceConfig, Session, load_head, run_comparison, PreparationSupportError


def prepare_ald(root, gevis, *, encoder='DEP.', preprocessing=PreprocessingConfig(),
                native=NativeConfig(), max_memory_mb=4096, progress=print):
    return _prepare_ald(root, config=native, patch_length=gevis.encoders[encoder][0].patch_len,
        policy=preprocessing,
        max_memory_mb=max_memory_mb, progress=progress)


def prepare_cvd(root, gevis):
    """Keep the existing CVD preprocessing and saved Run split unchanged."""
    from .experiments.pecvd_current_data import prepare
    source = SimpleNamespace(module=gevis.module, encoders=gevis.encoders)
    return prepare(root, source, splits=('train', 'validation', 'test'))


def train_ald(root, gevis, prepared, *, encoder='DEP.', preprocessing=PreprocessingConfig(),
              native=NativeConfig(), training=TrainingConfig(), choices=None, steps=None, progress=print):
    return run_comparison(root, gevis, encoder_step=encoder, prepared=prepared,
        preparation_policy=preprocessing, native_config=native, config=training,
        choices=choices, steps=steps, domain='PEALD', progress=progress)


def train_cvd(root, gevis, prepared, *, training=TrainingConfig(), steps=None, progress=print):
    return run_comparison(root, gevis, prepared=prepared, config=training,
        choices=(ChoiceConfig(mode='softmax'),), steps=steps, domain='PECVD', progress=progress)
