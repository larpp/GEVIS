"""Frozen numeric representation sensitivity on controlled validation changes."""
import torch


@torch.no_grad()
def audit_encoder(encoder, windows, *, seed=42):
    from pecvd.gevis_backbone import numeric_evidence_pairs
    encoder.eval()
    changed,reference,truth=numeric_evidence_pairs(windows,encoder.patch_len,seed=seed)
    baseline=encoder.tokens(reference); perturbed=encoder.tokens(changed)
    patch_values,_=encoder.patchify(changed)
    baseline_values,_=encoder.patchify(reference)
    current_error=(encoder(changed)-patch_values).square().mean(-1)
    baseline_error=(encoder(reference)-baseline_values).square().mean(-1)
    token_change=(perturbed-baseline).square().mean(-1)
    targets=truth['sensor']*truth['patch_count']+truth['patch']
    rows=torch.arange(len(changed),device=changed.device)
    return {'controlled_cases':len(changed),
            'latent_change_at_injected_patch':float(token_change[rows,targets].mean()),
            'largest_latent_change_patch_accuracy':float(token_change.argmax(-1).eq(targets).float().mean()),
            'residual_difference_at_injected_patch':float((current_error-baseline_error)[rows,targets].mean()),
            'meaning':'controlled sensitivity on validation; not anomaly accuracy'}
