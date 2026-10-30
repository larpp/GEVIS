"""Shared non-LLM paired temporal encoder and bounded training utilities.

The optional accepted center decision uses these same tested components; legacy
Trial names remain for compatibility with the standalone optimization runner.

No sensor IDs, targets, kinds or intervals enter forward. The frozen text input
is a sensor-free generic question; this pilot does not establish arbitrary
question comprehension. Patch features preserve TRAIN units, not window z-scores.
"""
import numpy as np
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


def paired_patch_features(current, reference, patches=16):
    a, b = np.asarray(current, float), np.asarray(reference, float)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) < 1 or patches < 1:
        raise ValueError('Matching nonempty time/sensor inputs required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite inputs required')
    # Fixed fractional bins; no truth-selected segment or per-window rescaling.
    out = []
    for ids in np.array_split(np.arange(len(a)), min(patches, len(a))):
        x, y = a[ids], b[ids]; d = x-y
        dx = np.diff(x, axis=0, prepend=x[:1]); dy = np.diff(y, axis=0, prepend=y[:1])
        xc, yc = x-x.mean(0), y-y.mean(0)
        corr = (xc*yc).mean(0)/(np.sqrt((xc*xc).mean(0)*(yc*yc).mean(0))+1e-6)
        out.append(np.stack([x.mean(0), y.mean(0), d.mean(0), x.std(0), y.std(0), d.std(0),
            np.abs(d).max(0), np.abs(d).mean(0), x.min(0), x.max(0), y.min(0), y.max(0),
            dx.std(0), dy.std(0), (dx-dy).std(0), np.abs(dx).max(0), np.abs(dy).max(0),
            corr, x[-1]-x[0], y[-1]-y[0]], -1))
    out = np.stack(out, axis=1)
    if out.shape[1] != patches:
        out = out[:, np.floor(np.linspace(0, out.shape[1]-1, patches)).astype(int)]
    return (np.sign(out)*np.log1p(np.abs(out))).astype(np.float32)


def split_runs(examples, fraction, seed):
    """Global run split across all Steps; repeated windows never become subjects."""
    if not 0 < fraction < 1: raise ValueError('Interior split fraction required')
    ids = sorted({e['run_id'] for e in examples})
    if len(ids) < 4: raise ValueError('At least four independent runs required')
    np.random.default_rng(seed).shuffle(ids)
    cut = max(1, min(len(ids)-1, int(len(ids)*fraction)))
    left = set(ids[:cut])
    return ([e for e in examples if e['run_id'] in left],
            [e for e in examples if e['run_id'] not in left])


def continuous_difference_features(current,reference,base_features):
    """Preserve boundary differences without changing amplitude/statistic columns.

    Inputs contain no intervention coordinates. The native first sample has zero
    difference; subsequent patch boundaries retain their preceding native sample.
    """
    a,b=np.asarray(current,float),np.asarray(reference,float)
    base=np.asarray(base_features,np.float32)
    if a.ndim!=2 or a.shape!=b.shape or not len(a) or base.ndim!=3 or base.shape[0]!=a.shape[1] or base.shape[2]!=20:
        raise ValueError('Matching native arrays and 20-column patch features required')
    if not all(np.isfinite(x).all() for x in (a,b,base)):raise ValueError('Finite inputs required')
    da=np.diff(a,axis=0,prepend=a[:1]);db=np.diff(b,axis=0,prepend=b[:1])
    patches=base.shape[1];values=[]
    for ids in np.array_split(np.arange(len(a)),min(patches,len(a))):
        x,y=da[ids],db[ids]
        values.append(np.stack([x.std(0),y.std(0),(x-y).std(0),np.abs(x).max(0),np.abs(y).max(0)],-1))
    values=np.stack(values,axis=1)
    if values.shape[1]!=patches:
        values=values[:,np.floor(np.linspace(0,values.shape[1]-1,patches)).astype(int)]
    out=base.copy();out[:,:,12:17]=np.log1p(values)
    return out




def local_waveform_statistics(values,windows=(9,33)):
    """Mean/std/range before reference aggregation, preserving member variation."""
    a=np.asarray(values,float)
    if a.ndim!=2 or not len(a) or not np.isfinite(a).all():raise ValueError('Finite time/sensor array required')
    if not windows or any(not isinstance(w,int) or w<3 or w%2!=1 for w in windows):
        raise ValueError('Odd local widths at least three required')
    stats=[]
    for width in windows:
        v=np.lib.stride_tricks.sliding_window_view(np.pad(a,((width//2,width//2),(0,0)),mode='edge'),width,axis=0)
        stats.extend([v.mean(-1),v.std(-1),np.ptp(v,axis=-1)])
    return np.stack(stats,-1)


def local_context_features(current,reference,base_features,windows=(9,33),reference_summary=None):
    """Append native local level/variability evidence; never align away true delays.

    Centered offline windows, edge padding, fixed native row widths. Raw paired
    features remain unchanged. No target, label, Step or sensor ID enters here.
    """
    a,b=np.asarray(current,float),np.asarray(reference,float)
    base=np.asarray(base_features,np.float32)
    if a.ndim!=2 or a.shape!=b.shape or not len(a) or base.ndim!=3 or base.shape[0]!=a.shape[1]:
        raise ValueError('Matching native arrays and sensor features required')
    if not all(np.isfinite(v).all() for v in (a,b,base)):raise ValueError('Finite inputs required')
    if not windows or any(not isinstance(w,int) or w<3 or w%2!=1 for w in windows):
        raise ValueError('Odd local widths at least three required')
    current_stats=local_waveform_statistics(a,windows)
    reference_stats=local_waveform_statistics(b,windows) if reference_summary is None else np.asarray(reference_summary)
    if reference_stats.shape!=current_stats.shape or not np.isfinite(reference_stats).all():raise ValueError('Aligned finite reference statistics required')
    values=[]
    for j in range(len(windows)):
        mx,sx,rx=[current_stats[:,:,j*3+i] for i in range(3)]
        my,sy,ry=[reference_stats[:,:,j*3+i] for i in range(3)]
        values.append(np.stack([mx,my,mx-my,sx,sy,sx-sy,rx-ry],-1))
    values=np.concatenate(values,-1)
    bins=np.array_split(np.arange(len(a)),min(base.shape[1],len(a)))
    pooled=np.stack([values[ids].mean(0) for ids in bins],axis=1)
    if len(bins)!=base.shape[1]:pooled=pooled[:,np.floor(np.linspace(0,len(bins)-1,base.shape[1])).astype(int)]
    return np.concatenate([base,(np.sign(pooled)*np.log1p(np.abs(pooled))).astype(np.float32)],-1)


class PairedPatchDetector(nn.Module):
    def __init__(self, features, query, method, width=48, temporal_dilation=1):
        super().__init__()
        if method not in ('paired_conv', 'paired_cross_attention'):
            raise ValueError('Unknown paired architecture')
        if not isinstance(temporal_dilation,int) or temporal_dilation<1:raise ValueError('Positive integer dilation required')
        self.method = method
        self.register_buffer('query', torch.as_tensor(query, dtype=torch.float32).flatten())
        self.project = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, width))
        self.temporal = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.GELU(),
                                      nn.Conv1d(width, width, 3, padding=temporal_dilation,dilation=temporal_dilation))
        if method == 'paired_cross_attention':
            self.query_project = nn.Linear(self.query.numel(), width)
            self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(width*3, width), nn.GELU(), nn.Linear(width, 1))

    def encode(self, features,query=None):
        batch, sensors, patches, _ = features.shape
        h = self.project(features).reshape(batch*sensors, patches, -1)
        h = h + self.temporal(h.transpose(1, 2)).transpose(1, 2)
        if self.method == 'paired_cross_attention':
            supplied=self.query if query is None else torch.as_tensor(query,dtype=features.dtype,device=features.device)
            if supplied.shape!=self.query.shape or not torch.isfinite(supplied).all():
                raise ValueError('One finite text embedding with the fitted width required')
            projected = self.query_project(supplied)[None, None].expand(batch*sensors, 1, -1)
            pooled, _ = self.attention(projected, h, h, need_weights=False)
            pooled = pooled[:, 0]
        else:
            pooled = h.mean(1)
        context=torch.cat([pooled,h.mean(1),h.amax(1)],-1).reshape(batch,sensors,-1)
        return h.reshape(batch,sensors,patches,-1),context

    def forward(self,features,query=None):
        _,context=self.encode(features,query=query)
        return self.head(context).squeeze(-1)


class PairedEncoderTrial:
    def __init__(self, method, query, *, seed=81001, width=48, device='cpu', control_centered=False,temporal_dilation=1):
        self.method, self.query, self.seed, self.width, self.device = method, query, seed, width, device
        self.control_centered = control_centered
        self.temporal_dilation=temporal_dilation

    def transform(self, features, step=None):
        features = np.asarray(features, np.float32)
        if not self.control_centered: return features
        if step not in self.control_scales: raise ValueError('Known TRAIN Step scaling required')
        center, scale = self.control_scales[step]
        return np.arcsinh((features-center)/scale).astype(np.float32)

    def fit(self, train, selection, *, epochs=35, batch_size=32, learning_rate=.001, balanced_selection=False):
        if epochs < 1 or batch_size < 1 or learning_rate <= 0: raise ValueError('Positive training budget required')
        if not train or not selection or any(e['split'] != 'train' for e in train+selection):
            raise ValueError('Only separate Train runs may fit/select weights')
        if {e['run_id'] for e in train} & {e['run_id'] for e in selection}:
            raise ValueError('Fitting/selection run leakage')
        self.control_scales = {}
        if self.control_centered:
            for step in {e['step'] for e in train}:
                controls = [e for e in train if e['step']==step and e['target'] is None]
                if len({e['run_id'] for e in controls}) < 4: raise ValueError('Insufficient TRAIN control runs')
                values = np.stack([e['features'] for e in controls])
                center = np.median(values, axis=0)
                # Fixed floor in signed-log feature units; no epsilon division.
                spread = np.maximum(np.quantile(np.abs(values-center), .9, axis=0), .1)
                self.control_scales[step] = (center, spread)
        x = np.stack([self.transform(e['features'], e.get('step')) for e in train])
        fingerprint=hashlib.sha256(x.tobytes())
        fingerprint.update(np.asarray([e['target'] if e['target'] is not None else -1 for e in train],dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(self.query).tobytes())
        fingerprint.update(np.stack([e['features'] for e in selection]).tobytes())
        fingerprint.update(np.asarray([e['target'] if e['target'] is not None else -1 for e in selection],dtype=np.int64).tobytes())
        # Private fingerprint stays in memory; report only equality for replay.
        self.input_fingerprint=fingerprint.hexdigest()
        self.mean = x.mean(axis=(0,1,2)); self.scale = np.maximum(x.std(axis=(0,1,2)), .05)
        tx = torch.tensor((x-self.mean)/self.scale, device=self.device)
        ty = torch.tensor([e['target'] if e['target'] is not None else -1 for e in train], device=self.device)
        vx = torch.tensor((np.stack([self.transform(e['features'], e.get('step')) for e in selection])-self.mean)/self.scale, device=self.device)
        vy = torch.tensor([e['target'] if e['target'] is not None else -1 for e in selection], device=self.device)
        if not (ty >= 0).any() or not (ty < 0).any(): raise ValueError('Changed and control examples required')
        cuda_devices = [torch.device(self.device).index or 0] if str(self.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            self.model = PairedPatchDetector(x.shape[-1], self.query, self.method, self.width,self.temporal_dilation).to(self.device)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=.01)
            best, best_state = float('inf'), None
            for epoch in range(epochs):
                self.model.train()
                # Equal chance of an untouched control or changed case, independently of sensor.
                weights = torch.where(ty < 0, 1/(ty < 0).sum(), 1/(ty >= 0).sum()).float()
                indices = torch.multinomial(weights, len(ty), replacement=True)
                for chunk in indices.split(batch_size):
                    logits = self.model(tx[chunk]); y = ty[chunk]
                    loss = self.loss(logits, y)
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    optimizer.step()
                self.model.eval()
                with torch.no_grad():
                    logits = torch.cat([self.model(batch) for batch in vx.split(batch_size)])
                    if balanced_selection:
                        if not (vy<0).any() or not (vy>=0).any():raise ValueError('Selection requires controls and changes')
                        value = float(.5*self.loss(logits[vy<0],vy[vy<0])+.5*self.loss(logits[vy>=0],vy[vy>=0]))
                    else:
                        value = float(self.loss(logits, vy))
                if value < best:
                    best = value; self.best_epoch = epoch+1
                    best_state = {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
            self.model.load_state_dict(best_state); self.model.eval()
        return self

    @staticmethod
    def loss(logits, target):
        changed = target >= 0
        labels = torch.zeros_like(logits)
        if changed.any(): labels[changed, target[changed]] = 1
        sensor_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=logits.new_tensor(logits.shape[1]-1))
        window_loss = F.binary_cross_entropy_with_logits(logits.amax(1), changed.float())
        ranking = F.cross_entropy(logits[changed], target[changed]) if changed.any() else logits.sum()*0
        return sensor_loss + window_loss + .5*ranking

    @torch.no_grad()
    def scores(self, features, step=None,query=None):
        x = self.transform(features, step)
        single = x.ndim == 3
        if single: x = x[None]
        out = self.model(torch.tensor((x-self.mean)/self.scale, device=self.device),query=query).cpu().numpy()
        return out[0] if single else out
