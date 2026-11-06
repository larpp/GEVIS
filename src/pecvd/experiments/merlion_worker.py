"""Standalone local Merlion process. Binary arrays use pipes, never CSV or disk."""
from contextlib import redirect_stdout, redirect_stderr
from importlib.metadata import version
from io import BytesIO, StringIO
import json
import os
import sys
import warnings


def fit_score(train,test,seed):
    import numpy as np
    import pandas as pd
    from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig
    from merlion.transform.base import Identity
    from merlion.utils import TimeSeries
    if train.ndim!=2 or test.ndim!=2 or train.shape[1]!=test.shape[1] or train.shape[1]<1 or min(len(train),len(test))<1:
        raise ValueError('Invalid array contract')
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError('Nonfinite arrays')
    def series(x,start):
        index=pd.date_range('2000-01-01',periods=len(x),freq='s')+pd.Timedelta(seconds=start)
        return TimeSeries.from_pd(pd.DataFrame(x,index=index))
    # Preserve point/window boundaries and use raw scores with validation-only
    # thresholds in the caller. No Merlion label/threshold fitting on test data.
    model=IsolationForest(IsolationForestConfig(transform=Identity(),n_jobs=1,
        enable_calibrator=False,enable_threshold=False))
    model.model.set_params(random_state=int(seed))
    model.train(series(train,0))
    scores=model.get_anomaly_score(series(test,len(train))).to_pd().to_numpy().reshape(-1)
    if len(scores)!=len(test) or not np.isfinite(scores).all(): raise ValueError('Invalid returned score alignment')
    return scores


def main():
    # Jupyter exports its inline backend to subprocesses. This standalone worker
    # deliberately has no ipykernel/matplotlib_inline dependency.
    os.environ['MPLBACKEND']='Agg'
    try:
        # Some dependencies log during import/fit. Never mix text with the binary
        # protocol or let warning messages interpolate private arrays into logs.
        with redirect_stdout(StringIO()),redirect_stderr(StringIO()),warnings.catch_warnings(record=True):
            import numpy as np
            from merlion.models.anomaly.isolation_forest import IsolationForest
            from merlion.transform.base import Identity
            if '--probe' in sys.argv:
                payload={'ready':True,'merlion_version':version('salesforce-merlion'),
                         'numpy_version':np.__version__,'python_version':sys.version.split()[0]}
                output=json.dumps(payload).encode('utf-8')
            else:
                with np.load(BytesIO(sys.stdin.buffer.read()),allow_pickle=False) as arrays:
                    scores=fit_score(arrays['train'],arrays['test'],int(arrays['seed'].item()))
                data=BytesIO(); np.savez(data,scores=scores)
                output=data.getvalue()
        sys.stdout.buffer.write(output)
        return 0
    except Exception as error:
        # Class only: exception messages can contain private values or filenames.
        sys.stderr.write('MERLION_ERROR:'+type(error).__name__)
        return 1


if __name__=='__main__': raise SystemExit(main())
