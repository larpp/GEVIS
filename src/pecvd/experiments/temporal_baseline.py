"""Bounded, locally fitted LSTM reconstruction baseline; not a Merlion model."""
import time
import numpy as np
import torch
from torch import nn


class WindowLSTMAutoencoder(nn.Module):
    def __init__(self, channels, hidden=16):
        super().__init__()
        self.encoder=nn.LSTM(channels,hidden,batch_first=True)
        self.decoder=nn.LSTM(hidden,hidden,batch_first=True)
        self.output=nn.Linear(hidden,channels)

    def forward(self,x):
        _,(hidden,_)=self.encoder(x)
        repeated=hidden[-1][:,None].expand(-1,x.shape[1],-1)
        decoded,_=self.decoder(repeated)
        return self.output(decoded)


def temporal_scores(train_windows, test_windows, *, seed=42, epochs=10, hidden=16):
    """Each window is independent: never learn artificial joins between runs.

    All input windows use the existing TRAIN normalization. Score = largest
    per-time mean-channel squared reconstruction error; not fault probability.
    No validation/test fitting or checkpoint selection.
    """
    train=np.asarray(train_windows,dtype=np.float32); test=np.asarray(test_windows,dtype=np.float32)
    if train.ndim!=3 or test.ndim!=3 or train.shape[1:]!=test.shape[1:] or min(len(train),len(test))<1:
        raise ValueError('Matching independent time/sensor windows required')
    if not np.isfinite(train).all() or not np.isfinite(test).all() or not 1<=epochs<=100:
        raise ValueError('Finite windows and bounded epochs required')
    started=time.perf_counter()
    # CPU, bounded capacity, isolated RNG; caller's CUDA/torch seed stays intact.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=WindowLSTMAutoencoder(train.shape[-1],hidden)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        x=torch.from_numpy(train)
        for _ in range(epochs):
            for batch in x.split(8):
                loss=(model(batch)-batch).square().mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite temporal baseline training')
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        fit_ms=(time.perf_counter()-started)*1000
        model.eval(); score_started=time.perf_counter()
        with torch.no_grad():
            scores=torch.cat([(model(batch)-batch).square().mean(-1).max(-1).values
                for batch in torch.from_numpy(test).split(8)]).numpy()
        score_ms=(time.perf_counter()-score_started)*1000
    return scores,{'fit_ms':fit_ms,'scoring_ms':score_ms,'scored_windows':len(test),
                  'epochs':epochs,'hidden_size':hidden,'device':'cpu','fit_split':'train',
                  'parameter_count':sum(p.numel() for p in model.parameters())}
