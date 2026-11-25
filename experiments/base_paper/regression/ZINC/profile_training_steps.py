"""Profile ShareGNN training steps on ZINC-full (spec 25 follow-up).

Runs the regular experiment setup, then for the first training epoch:
1. times WARMUP + TIMED plain steps with a CUDA synchronize per step
   (true per-step wall time), then
2. records PROFILED steps with torch.profiler and prints the top ops by GPU
   and by CPU time, plus the GPU-busy share of the wall time.

The run stops afterwards; results go to a temporary directory so existing
results are neither skipped nor overwritten.

Usage (repo root):
    python experiments/base_paper/regression/ZINC/profile_training_steps.py
    # CPU smoke test without the 32.6 GiB row cache:
    python experiments/base_paper/regression/ZINC/profile_training_steps.py \
        --set device=cpu --set share_gnn_forward.precompute_rows=false
"""
import tempfile
import time
from pathlib import Path

import click
import torch
import yaml

from simplegnn.framework.core import FrameworkMain
from simplegnn.framework.model_configuration import ModelConfiguration

MAIN_CONFIG = Path('experiments/base_paper/regression/ZINC/configs/main_config_ZINC_full.yml')


class StopProfiling(Exception):
    pass


def profiled_train_graph_task(warmup, timed, profiled):
    original = ModelConfiguration.train_graph_task

    def train_graph_task(self, epoch, values, train_batches, timer):
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        state = {'step': 0, 'last': None, 'times': [], 'prof': None}
        total = warmup + timed + profiled

        def after_step(optimizer, args, kwargs):
            sync()
            now = time.perf_counter()
            step = state['step']
            if warmup <= step < warmup + timed and state['last'] is not None:
                state['times'].append(now - state['last'])
            state['last'] = now
            if step == warmup + timed - 1:
                times = sorted(state['times'])
                print(f"\nPROFILE plain step time: median {1e3 * times[len(times) // 2]:.2f} ms, "
                      f"mean {1e3 * sum(times) / len(times):.2f} ms over {len(times)} steps "
                      f"(batch size {len(train_batches[0])}, {len(train_batches)} steps/epoch)", flush=True)
                state['prof'] = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                state['prof'].__enter__()
                state['prof_start'] = time.perf_counter()
            state['step'] += 1
            if state['step'] == total:
                prof = state['prof']
                prof.__exit__(None, None, None)
                wall = time.perf_counter() - state['prof_start']
                averages = prof.key_averages()
                gpu_total = sum(e.self_device_time_total for e in averages) / 1e6
                print(f"\nPROFILE {profiled} profiled steps: wall {wall:.2f}s, "
                      f"GPU kernel time {gpu_total:.2f}s ({100 * gpu_total / wall:.0f}% busy)", flush=True)
                print(averages.table(sort_by='self_device_time_total', row_limit=25), flush=True)
                print(averages.table(sort_by='self_cpu_time_total', row_limit=25), flush=True)
                raise StopProfiling

        self.optimizer.register_step_post_hook(after_step)
        return original(self, epoch, values, train_batches, timer)

    return train_graph_task


@click.command()
@click.option('--warmup', default=20, help='Untimed warmup steps')
@click.option('--timed', default=100, help='Steps timed without the profiler')
@click.option('--profiled', default=30, help='Steps recorded with torch.profiler')
@click.option('--set', 'overrides', multiple=True,
              help='Override a parameters entry, e.g. --set device=cpu '
                   '--set share_gnn_forward.precompute_rows=false')
def main(warmup, timed, profiled, overrides):
    ModelConfiguration.train_graph_task = profiled_train_graph_task(warmup, timed, profiled)
    with tempfile.TemporaryDirectory() as results:
        config = yaml.safe_load(MAIN_CONFIG.read_text())
        for dataset in config['datasets']:
            dataset['paths']['results'] = f'{results}/'
            if overrides:
                parameters_path = Path(dataset['paths']['hyperparameters'])
                parameters = yaml.safe_load(parameters_path.read_text())
                for override in overrides:
                    key, value = override.split('=', 1)
                    *parents, leaf = key.split('.')
                    target = parameters
                    for parent in parents:
                        target = target.setdefault(parent, {})
                    target[leaf] = yaml.safe_load(value)
                dataset['paths']['hyperparameters'] = f'{results}/{parameters_path.name}'
                Path(dataset['paths']['hyperparameters']).write_text(yaml.safe_dump(parameters))
        config_path = Path(results) / MAIN_CONFIG.name
        config_path.write_text(yaml.safe_dump(config))
        experiment = FrameworkMain(config_path)
        experiment.preprocessing(num_threads=1)
        try:
            experiment.run_configurations(num_threads=1)
        except StopProfiling:
            pass


if __name__ == '__main__':
    main()
