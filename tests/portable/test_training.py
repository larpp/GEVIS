"""Tiny invented-data training and checkpoint restoration, not a real experiment."""
import numpy as np
import torch
from GEVIS_Model import create_encoder, load_numeric_encoder
from GEVIS_Model.training.encoder import fit_encoder, EncoderTrainingConfig


def test_numeric_ssl_train_and_restore(tmp_path):
    torch.manual_seed(42)
    model=create_encoder(n_sensors=2,layers=1)
    data=np.random.default_rng(42).normal(size=(2,16,2)).astype(np.float32)
    before=model.encoder.input.weight.detach().clone()
    encoder,history=fit_encoder(model.encoder,data,data*.9,
        config=EncoderTrainingConfig(epochs=1,batch_size=2))
    assert not torch.equal(before,encoder.input.weight)
    assert len(history)==1 and np.isfinite(history[0]['validation_blind_mse'])
    file=tmp_path/'numeric.pt'
    torch.save(dict(schema='gevis.numeric_ssl.portable.v1',n_sensors=2,patch_length=4,
        model_config=dict(d_model=96,nhead=8,layers=1,dropout=0.),model_state=encoder.state_dict()),file)
    restored,_=load_numeric_encoder(file)
    with torch.no_grad():
        torch.testing.assert_close(encoder.tokens(torch.from_numpy(data)),restored.tokens(torch.from_numpy(data)),rtol=0,atol=0)
