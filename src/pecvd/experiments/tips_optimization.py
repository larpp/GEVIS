"""Bounded caching of frozen trace encodings; no cached trainable activations."""
from collections import OrderedDict
import hashlib
import math
import time

import torch


def trace_key(values):
    """Hash the actual float32 model input, independent of labels and case IDs."""
    return hashlib.sha256(str(values.shape).encode() + values.tobytes()).hexdigest()


class FrozenTraceCache:
    def __init__(self, max_mb=256):
        if max_mb < 0:
            raise ValueError('Trace cache memory must be nonnegative')
        self.limit = int(max_mb * 1024 * 1024)
        self.entries = OrderedDict(); self.bytes = 0
        self.hits = self.misses = self.encoded_traces = 0

    @torch.no_grad()
    def encode_pair(self, encoder, current, reference, trace_keys=None):
        if encoder.training or any(p.requires_grad for p in encoder.parameters()):
            raise ValueError('Trace caching requires a frozen eval-mode encoder')
        joined = torch.cat([current, reference], 0)
        keys = (list(trace_keys['current']) + list(trace_keys['reference'])) if trace_keys else [
            trace_key(x.detach().cpu().numpy()) for x in joined]
        unique = list(dict.fromkeys(keys)); batch_values = {}; missing = []
        for key in unique:
            if key in self.entries:
                self.hits += 1; self.entries.move_to_end(key)
                batch_values[key] = tuple(x.to(current.device) for x in self.entries[key])
            else:
                missing.append(key); self.misses += 1
        if missing:
            first = {key: keys.index(key) for key in missing}
            indices = torch.tensor([first[key] for key in missing], device=current.device)
            values = joined.index_select(0, indices)
            tokens = encoder.tokens(values); prediction = encoder.blind_prediction(values)
            self.encoded_traces += len(missing)
            for key, token, pred in zip(missing, tokens, prediction):
                batch_values[key] = (token, pred)
                size = sum(x.numel() * x.element_size() for x in (token, pred))
                if size <= self.limit:
                    while self.bytes + size > self.limit:
                        _, evicted = self.entries.popitem(last=False)
                        self.bytes -= sum(x.numel() * x.element_size() for x in evicted)
                    # Independent storage: a CPU view would retain the whole miss batch.
                    self.entries[key] = tuple(x.detach().to('cpu', copy=True) for x in (token, pred))
                    self.bytes += size
        tokens = torch.stack([batch_values[key][0] for key in keys])
        predictions = torch.stack([batch_values[key][1] for key in keys])
        return (*tokens.chunk(2, 0), *predictions.chunk(2, 0))

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, encoded_traces=self.encoded_traces,
                    host_cache_mb=self.bytes / 1024 ** 2, limit_mb=self.limit / 1024 ** 2)


def numeric_state_from_encoded(encoder, current, reference, encoded):
    """Same Multiscale bridge algebra as temporal_bridge.numeric_state."""
    from pecvd.relational_features import relational_features
    from pecvd.temporal_bridge import MULTISCALE_DIM
    a, b, cr, rr = encoded
    cp, p = encoder.patchify(current); rp, _ = encoder.patchify(reference)
    cm, rm = cp.mean(-1), rp.mean(-1)
    cs, rs = cp.std(-1, unbiased=False), rp.std(-1, unbiased=False)
    cd, rd = cp.diff(dim=-1), rp.diff(dim=-1)
    def progression(v):
        v = v.reshape(len(current), current.shape[-1], p)
        return torch.cat([torch.zeros_like(v[..., :1]), v.diff(dim=-1)], -1).reshape_as(cm)
    cprog, rprog = progression(cm), progression(rm)
    delta = cp - rp; centered_c, centered_r = cp - cm[..., None], rp - rm[..., None]
    corr = (centered_c * centered_r).sum(-1) / (centered_c.square().sum(-1) * centered_r.square().sum(-1)).clamp_min(1e-8).sqrt()
    extra = torch.stack([cm-rm, cs-rs, cd.mean(-1), rd.mean(-1), (cd-rd).mean(-1),
        delta.abs().amax(-1), delta.square().mean(-1).sqrt(), (cd-rd).std(-1, unbiased=False),
        cd.std(-1, unbiased=False), rd.std(-1, unbiased=False), cprog, rprog, cprog-rprog, corr,
        (cr-rr).mean(-1), (cr-rr).abs().mean(-1)], -1)
    old = torch.stack([cm, rm, (cm-rm).abs(), cs, rs, delta.abs().mean(-1),
        (cr-cp).square().mean(-1), (rr-rp).square().mean(-1)], -1)
    extra = torch.cat([extra, relational_features(current, reference, encoder.patch_len)], -1)
    state = torch.cat([a, b, a-b, (a-b).abs(), extra, old], -1)
    if state.shape[-1] != MULTISCALE_DIM:
        raise ValueError('Multiscale bridge dimension mismatch')
    return state


class PairedBatchSampler:
    """Keep center/counterfactual twins adjacent, retaining every original example."""
    def __init__(self, dataset, batch_size, shuffle=False, seed=42):
        if len(dataset) % 2 or batch_size < 1:
            raise ValueError('Paired examples and a positive batch size required')
        self.size, self.batch_size = len(dataset), batch_size
        self.shuffle = shuffle; self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __iter__(self):
        pairs = self.size // 2
        order = torch.randperm(pairs, generator=self.generator).tolist() if self.shuffle else range(pairs)
        batch = []
        for pair in order:
            for index in [2 * pair, 2 * pair + 1]:
                batch.append(index)
                if len(batch) == self.batch_size:
                    yield batch; batch = []
        if batch:
            yield batch


class BatchProgress:
    def __init__(self, label, total, *, every=50, seconds=10., callback=None):
        if every < 1 or seconds <= 0:
            raise ValueError('Positive progress intervals required')
        self.label, self.total = label, total
        self.every, self.seconds, self.callback = every, seconds, callback
        self.start = self.last = time.perf_counter()
        self.update(0)

    def update(self, batch, loss=None):
        now = time.perf_counter()
        if batch not in (0, self.total) and batch % self.every and now - self.last < self.seconds:
            return
        self.last = now; elapsed = now - self.start
        eta = elapsed * (self.total - batch) / batch if batch else None
        payload = dict(stage=self.label, batch=batch, total_batches=self.total,
            percent=100 * batch / max(1, self.total), elapsed_s=elapsed, eta_s=eta, loss=loss)
        detail = f' loss={loss:.5f}' if loss is not None else ''
        remaining = f'{eta / 60:.1f} min' if eta is not None else 'estimating'
        print(f'[{self.label}] batch {batch:,}/{self.total:,} ({payload["percent"]:.1f}%)'
              f'{detail} elapsed={elapsed / 60:.1f} min ETA={remaining}', flush=True)
        if self.callback:
            self.callback(payload)
