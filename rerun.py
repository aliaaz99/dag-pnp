"""Run DAG-PnP experiments and save results, settings and a console log."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('table', type=int, choices=range(1,5))
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--torch-seed', type=int, default=0, help='Random seed for optimization and denoising.')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--checkpoint', choices=['SF','ER'], help='Checkpoint (default: SF for synthetic experiments, ER for Sachs).')
    p.add_argument('--output', type=Path, help='New output directory; refuses to overwrite.')
    args = p.parse_args()
    if min(args.seeds, args.iterations, args.threads) < 1:
        p.error('seeds, iterations and threads must be positive')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[name] = str(args.threads)
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import torch
    import dag_config
    from dag_pnp import GDSSDiffusionDenoiser
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.torch_seed)
    np.random.seed(args.torch_seed)
    random.seed(args.torch_seed)
    dag_config.NUM_ITERS = args.iterations
    names = {1:'experiment1_denoiser_ablation', 2:'experiment2_barrier_step',
             3:'experiment4_sample_size', 4:'experiment4_sachs'}
    import importlib
    experiment = importlib.import_module(names[args.table])
    family = args.checkpoint or ('ER' if args.table == 4 else 'SF')
    checkpoint = ROOT / 'checkpoints' / (family + '.pth')
    experiment.get_denoiser = lambda: GDSSDiffusionDenoiser.from_checkpoint(str(checkpoint), device='cpu')
    output = args.output or ROOT / 'outputs' / ('table{}_{}'.format(args.table, time.strftime('%Y%m%d_%H%M%S')))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(table=args.table, seeds=args.seeds, iterations=args.iterations,
                    torch_seed=args.torch_seed, threads=args.threads, checkpoint=family,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    python=sys.version, numpy=np.__version__, torch=torch.__version__,
                    profile=json.loads((ROOT / 'configs' / ('table{}.json'.format(args.table))).read_text()))
    manifest['code_sha256'] = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in (ROOT/'src').rglob('*.py')}
    (output / 'run.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Output: ' + str(output), flush=True)
    class Tee:
        def __init__(self, log): self.log = log
        def write(self, s):
            sys.__stdout__.write(s)
            self.log.write(s)
        def flush(self):
            sys.__stdout__.flush()
            self.log.flush()
    with (output / 'run.log').open('w') as log, contextlib.redirect_stdout(Tee(log)):
        if args.table == 1:
            experiment.NUM_SEEDS = args.seeds
            denoiser = experiment.get_denoiser()
            for family in ['ER','SF']:
                for d in [20,50]:
                    experiment.run_one_setting(d,d,family,denoiser,str(output),normalise=True)
        elif args.table == 2:
            experiment.NUM_SEEDS = args.seeds
            experiment.run_experiment2(str(output),normalise=True)
        elif args.table == 3:
            experiment.NUM_SEEDS = args.seeds
            experiment.run_experiment4(str(output))
        else:
            experiment.run_experiment4(str(output),str(ROOT/'data'),num_runs=args.seeds,normalise=False)
    print('Finished: ' + str(output))


if __name__ == '__main__':
    main()
