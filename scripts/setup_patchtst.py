"""Download public weights only. Does not open or upload project inputs."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pecvd.pretrained_patchtst import MODEL_ID,REVISION,checkpoint_hash

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir',type=Path,default=ROOT/'.cache/pretrained_patchtst')
    args=parser.parse_args()
    from huggingface_hub import snapshot_download
    path=snapshot_download(MODEL_ID,revision=REVISION,local_dir=args.local_dir,
        allow_patterns=['config.json','model.safetensors','README.md','LICENSE*'])
    print({'public_model':MODEL_ID,'revision':REVISION,'files_sha256':checkpoint_hash(path)})

if __name__=='__main__': main()
