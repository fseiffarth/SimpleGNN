"""BREC expressiveness benchmark — RPC (Reliable Paired Comparisons) runner.

BREC (Wang & Zhang, 2023; arXiv:2304.07702) measures *realized* GNN
expressiveness: for each of 400 non-isomorphic graph pairs, a siamese model is
trained to push the two graphs' embeddings apart, and the pair counts as
distinguished only if a Hotelling T-squared statistic over 32 relabelings
exceeds a threshold while a known-isomorphic control pair stays below it.

That protocol does not fit ``FrameworkMain.run_configurations()`` (400
independent models, no labels, a statistical decision rule instead of a
test-set metric — see specs/21-brec-expressiveness-benchmark.md), so this
script drives the loop itself while reusing the framework's data pipeline:
``FrameworkMain.preprocessing()`` builds the dataset plus the invariant labels
and properties, and every model is a plain ``GraphModel`` built from
``models_brec.yml``.

The constants below are transcribed from the reference implementation
(``GraphPKU/BREC@Release:DropGNN/test_BREC.py``); ``--epochs`` and the pair
selection flags exist for cheap smoke runs and change the protocol, so any
number reported as a BREC score must come from a default full run.

Usage
-----
    # smoke run: 3 pairs of the Basic category on the 4-relabeling variant
    python experiments/brec/main_brec.py --dataset BREC-r4 --parts Basic --pairs 3

    # official protocol (400 pairs, 32 relabelings, 50 epochs each)
    python experiments/brec/main_brec.py --dataset BREC
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import torch
from torch.nn import CosineEmbeddingLoss

from simplegnn.datasets.graph_dataset import CustomBatchLoader
from simplegnn.datasets.graph_dataset_preprocessing import BREC_NUM_PAIRS, BREC_PARTS, parse_brec_name
from simplegnn.framework.core import FrameworkMain, preprocess_graph_data
from simplegnn.framework.run_configuration import get_run_configs
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.framework.utils.preprocessing import load_preprocessed_data_and_parameters
from simplegnn.models.model import GraphModel

# --- reference RPC constants (DropGNN/test_BREC.py) ---------------------------
EPOCH = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MARGIN = 0.0
THRESHOLD = 5.0
EPSILON_CMP = 1e-6
# The reference's optional ridge on the covariance, needed when a model maps all
# relabelings of a graph to *exactly* the same embedding (see t2_statistic and
# --epsilon-matrix). Off by default, like the reference.
EPSILON_MATRIX = 1e-7
LOSS_THRESHOLD = 0.0
SEED = 2023
# -----------------------------------------------------------------------------

CONFIG_PATH = Path('experiments/brec/configs/main_config_brec.yml')


def t2_statistic(embeddings: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    """
    Hotelling T-squared statistic of a pair's relabeling embeddings.

    Parameters
    ----------
    embeddings : torch.Tensor
        Model outputs for one pair's ``2 * num_relabel`` graphs, in dataset
        order (``A0 B0 A1 B1 ...``), shape ``(2 * num_relabel, output_dim)``.
    epsilon : float, optional
        Ridge added to the diagonal of the covariance before inverting
        (default: 0.0, the reference's behavior). See the note below.

    Returns
    -------
    torch.Tensor
        Scalar (0-dim) statistic ``D_mean^T · pinv(cov(D)) · D_mean`` for
        ``D = X - Y``, with ``X``/``Y`` the ``output_dim × num_relabel``
        matrices of the first/second graph's embeddings. Matches
        ``T2_calculation`` in the reference implementation.

    Notes
    -----
    The pseudo-inverse makes a *perfectly consistent* separation collapse to
    zero: if every relabeling yields the same difference vector, ``cov(D)`` is
    the zero matrix, ``pinv`` of it is zero too, and the statistic is 0 no
    matter how far apart the two graphs' embeddings are. The reference
    implementation notes the same trap and ships the fix commented out — add a
    small ridge (``epsilon``) to the covariance. It relies on floating-point
    noise across permutations to avoid the degenerate case, which holds for the
    large graphs and float32 models it was written for but not necessarily for
    a deterministic double-precision model on a 10-node graph. Use
    ``--epsilon-matrix`` when :func:`is_degenerate` reports pairs.
    """
    X = embeddings[0::2].T
    Y = embeddings[1::2].T
    D = X - Y
    D_mean = torch.mean(D, dim=1).reshape(-1, 1)
    S = torch.cov(D)
    if epsilon:
        S = S + epsilon * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
    return torch.mm(torch.mm(D_mean.T, torch.linalg.pinv(S)), D_mean).reshape(())


def is_degenerate(embeddings: torch.Tensor, tolerance: float = 1e-12) -> bool:
    """
    True if a pair's embeddings separate the graphs but with zero variance.

    That is the case the pseudo-inverse turns into a T-squared of 0 (see
    :func:`t2_statistic`): the two graphs *are* mapped to different embeddings,
    identically so for every relabeling. Reported per run so a score of 0 that
    is really a numerical artifact cannot be mistaken for a genuine failure to
    distinguish.

    Parameters
    ----------
    embeddings : torch.Tensor
        Model outputs for one pair, shape ``(2 * num_relabel, output_dim)``.
    tolerance : float, optional
        Threshold below which the covariance counts as zero (default: 1e-12).

    Returns
    -------
    bool
    """
    D = embeddings[0::2].T - embeddings[1::2].T
    return bool(torch.cov(D).abs().max() < tolerance and D.mean(dim=1).abs().max() > tolerance)


def decide(t2_traintest: torch.Tensor, t2_reliability: torch.Tensor,
           threshold: float = THRESHOLD) -> tuple[bool, bool]:
    """
    Apply the RPC decision rule to one pair's two statistics.

    Parameters
    ----------
    t2_traintest : torch.Tensor
        T-squared statistic of the evaluated (non-isomorphic) pair.
    t2_reliability : torch.Tensor
        T-squared statistic of the corresponding isomorphic control pair.
    threshold : float, optional
        Decision threshold (default: the reference's 5.0).

    Returns
    -------
    tuple of bool
        ``(distinguished, reliable)``. ``distinguished`` requires the pair's
        statistic to clear the threshold *and* to differ from the control's,
        so a model that simply produces a large statistic for everything
        (including isomorphic graphs) scores zero. ``reliable`` records
        whether the control itself stayed below the threshold.
    """
    distinguished = bool(t2_traintest > threshold
                         and not torch.isclose(t2_traintest, t2_reliability, atol=EPSILON_CMP))
    reliable = bool(t2_reliability < threshold)
    return distinguished, reliable


class RPCEvaluator:
    """
    Per-pair siamese training and T-squared evaluation over one BREC dataset.

    Parameters
    ----------
    graph_data : GraphDataset
        The preprocessed BREC dataset (all ``800 * 2 * num_relabel`` graphs).
    para : Parameters
        Parameters for one hyperparameter configuration, as produced by
        ``load_preprocessed_data_and_parameters``. Supplies the layer
        specification, device and precision.
    num_relabel : int
        Relabelings per graph in ``graph_data``.
    epochs : int, optional
        Training epochs per pair (default: the reference's 50).
    epsilon : float, optional
        Covariance ridge handed to :func:`t2_statistic` (default: 0.0).
    """

    def __init__(self, graph_data, para: Parameters, num_relabel: int, epochs: int = EPOCH,
                 epsilon: float = 0.0):
        self.graph_data = graph_data
        self.para = para
        self.num_relabel = num_relabel
        self.epochs = epochs
        self.epsilon = epsilon
        # embedding width, discovered on the first forward pass (the model config
        # decides it); used to flag a rank-deficient covariance
        self.output_dim = None
        config = para.run_config.config
        self.device = torch.device(config['device'] if config.get('device') and torch.cuda.is_available() else 'cpu')
        self.dtype = torch.double if config.get('precision', 'float') == 'double' else torch.float
        self.with_invariant_layers = config.get('with_invariant_layers', True)
        self.loss_function = CosineEmbeddingLoss(margin=MARGIN)

    def graph_ids(self, pair_id: int) -> list[int]:
        """Dataset indices of one pair id's graphs, in ``A0 B0 A1 B1 ...`` order."""
        block = 2 * self.num_relabel
        return list(range(pair_id * block, (pair_id + 1) * block))

    def _batches(self, graph_ids: list[int]) -> list[list[int]]:
        """Split a pair's graphs into batches, keeping the A/B interleaving."""
        return [graph_ids[i:i + BATCH_SIZE] for i in range(0, len(graph_ids), BATCH_SIZE)]

    def _prepare(self, graph_ids: list[int]) -> list[tuple]:
        """
        Precollate one pair's batched inputs, reused across all epochs.

        A pair's graphs never change, so the node-feature concatenation (ShareGNN)
        or the PyG collation (classical) is done once instead of once per epoch.

        Returns
        -------
        list of tuple
            ``(batch_data, positions)`` pairs ready for :meth:`_forward`;
            ``positions`` is None on the classical path.
        """
        prepared = []
        for batch_ids in self._batches(graph_ids):
            if self.with_invariant_layers:
                # same batch assembly as ModelConfiguration._assemble_share_gnn_batch
                slices = self.graph_data.slices['x']
                x = torch.cat([self.graph_data.x[int(slices[g]):int(slices[g + 1])] for g in batch_ids])
                prepared.append((SimpleNamespace(x=x.to(self.device)), batch_ids))
            else:
                batch = next(iter(CustomBatchLoader(self.graph_data, [batch_ids])))
                prepared.append((batch.to(self.device), None))
        return prepared

    @staticmethod
    def _forward(net: GraphModel, batch_data, positions) -> torch.Tensor:
        """One forward pass over a prepared batch (ShareGNN or classical)."""
        if positions is None:
            return net(batch_data)
        return net(batch_data, pos=positions)

    def _new_model(self, pair_id: int) -> GraphModel:
        """A freshly initialized model for one pair (seed varies per pair, reproducibly)."""
        return GraphModel(graph_data=self.graph_data, para=self.para,
                          seed=SEED + pair_id, device=self.device).to(self.device)

    def _embeddings(self, net: GraphModel, prepared: list[tuple]) -> torch.Tensor:
        """Concatenated model outputs for a pair's graphs, in dataset order."""
        net.eval()
        with torch.no_grad():
            return torch.cat([self._forward(net, data, positions).detach() for data, positions in prepared])

    def evaluate_pair(self, pair_id: int) -> dict:
        """
        Train on one pair and apply the RPC decision rule.

        Parameters
        ----------
        pair_id : int
            Evaluation pair id in ``[0, 400)``. Its reliability control is
            pair id ``pair_id + 400``.

        Returns
        -------
        dict
            ``pair_id``, ``t2_traintest``, ``t2_reliability``, ``loss``,
            ``distinguished``, ``reliable``, ``degenerate`` and ``seconds``.
        """
        start = time.perf_counter()
        traintest = self._prepare(self.graph_ids(pair_id))
        reliability = self._prepare(self.graph_ids(pair_id + BREC_NUM_PAIRS))

        net = self._new_model(pair_id)
        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        net.train()
        loss_all = float('nan')
        for _ in range(self.epochs):
            loss_all = 0.0
            for batch_data, positions in traintest:
                optimizer.zero_grad(set_to_none=True)
                prediction = self._forward(net, batch_data, positions)
                # target -1: push the two graphs of the pair apart
                target = torch.full((len(prediction) // 2,), -1.0,
                                    dtype=prediction.dtype, device=prediction.device)
                loss = self.loss_function(prediction[0::2], prediction[1::2], target)
                loss.backward()
                optimizer.step()
                loss_all += len(prediction) / 2 * loss.item()
            loss_all /= self.num_relabel
            if loss_all < LOSS_THRESHOLD:
                break
            scheduler.step(loss_all)

        traintest_embeddings = self._embeddings(net, traintest)
        self.output_dim = traintest_embeddings.shape[1]
        t2_traintest = t2_statistic(traintest_embeddings, self.epsilon)
        t2_reliability = t2_statistic(self._embeddings(net, reliability), self.epsilon)
        distinguished, reliable = decide(t2_traintest, t2_reliability)
        return {'pair_id': pair_id,
                't2_traintest': float(t2_traintest),
                't2_reliability': float(t2_reliability),
                'loss': float(loss_all),
                'distinguished': distinguished,
                'reliable': reliable,
                'degenerate': is_degenerate(traintest_embeddings),
                'seconds': round(time.perf_counter() - start, 2)}


def selected_pairs(parts: str | None, pairs_per_part: int | None) -> list[tuple[str, list[int]]]:
    """
    Resolve the ``--parts`` / ``--pairs`` selection into pair ids per category.

    Parameters
    ----------
    parts : str or None
        Comma-separated category names (``Basic``, ``Regular``, ``Extension``,
        ``CFI``, ``4-Vertex_Condition``, ``Distance_Regular``), or None for all.
    pairs_per_part : int or None
        Keep only the first *n* pairs of each selected category, or None for all.

    Returns
    -------
    list of tuple
        ``(category_name, pair_ids)`` in BREC's canonical category order.
    """
    if parts is None:
        wanted = [name for name, _, _ in BREC_PARTS]
    else:
        wanted = [p.strip() for p in parts.split(',') if p.strip()]
        known = {name for name, _, _ in BREC_PARTS}
        unknown = [p for p in wanted if p not in known]
        if unknown:
            raise click.BadParameter(f'Unknown BREC part(s) {unknown}. Choose from {sorted(known)}.')
    selection = []
    for name, start, end in BREC_PARTS:
        if name not in wanted:
            continue
        ids = list(range(start, end))
        if pairs_per_part is not None:
            ids = ids[:pairs_per_part]
        selection.append((name, ids))
    return selection


def run_brec(dataset: str = 'BREC',
             config_path: Path = CONFIG_PATH,
             config_id: int = 0,
             parts: str | None = None,
             pairs_per_part: int | None = None,
             epochs: int = EPOCH,
             epsilon: float = 0.0,
             num_threads: int = 1,
             skip_preprocessing: bool = False,
             results_file: Path | None = None) -> dict:
    """
    Run the RPC evaluation and print the per-category summary.

    Parameters
    ----------
    dataset : str, optional
        BREC variant: ``'BREC'`` (official, 32 relabelings) or ``'BREC-r<k>'``.
        Must match the ``name`` in the main config.
    config_path : Path, optional
        Main config file (default: ``experiments/brec/configs/main_config_brec.yml``).
    config_id : int, optional
        Index into the hyperparameter grid of the config (default: 0).
    parts, pairs_per_part
        Pair selection, see :func:`selected_pairs`.
    epochs : int, optional
        Training epochs per pair (default: 50, the reference protocol).
    epsilon : float, optional
        Covariance ridge for the T-squared statistic (default: 0.0, the
        reference's behavior). Set it (e.g. to ``EPSILON_MATRIX``) when the run
        reports degenerate pairs — see :func:`t2_statistic`.
    num_threads : int, optional
        Threads for the dataset preprocessing step (default: 1).
    skip_preprocessing : bool, optional
        Skip ``FrameworkMain.preprocessing()``; only valid once the dataset,
        labels and properties are already on disk (default: False).
    results_file : Path or None, optional
        CSV destination for the per-pair records. Defaults to
        ``<results_path>/<dataset>/brec_rpc_<config_id>.csv``.

    Returns
    -------
    dict
        ``{'total': (distinguished, num_pairs), 'parts': {name: (distinguished,
        num_pairs)}, 'fail_in_reliability': int, 'records': [...]}``.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    framework = FrameworkMain(config_path)
    if not skip_preprocessing:
        framework.preprocessing(num_threads=num_threads)

    configurations = [c for c in framework.get_configuration_list() if c['name'] == dataset]
    if not configurations:
        available = sorted({c['name'] for c in framework.get_configuration_list()})
        raise click.BadParameter(f"No dataset '{dataset}' in {config_path}. Available: {available}.")
    configuration = configurations[0]
    num_relabel = parse_brec_name(dataset)

    graph_data = preprocess_graph_data(configuration)
    run_configs = get_run_configs(configuration)
    if not 0 <= config_id < len(run_configs):
        raise click.BadParameter(f'config_id {config_id} out of range (0-{len(run_configs) - 1}).')
    para = Parameters()
    load_preprocessed_data_and_parameters(config_id=config_id, run_id=0, validation_id=0,
                                         validation_folds=1, graph_data=graph_data,
                                         run_config=run_configs[config_id], para=para)

    evaluator = RPCEvaluator(graph_data, para, num_relabel=num_relabel, epochs=epochs, epsilon=epsilon)
    selection = selected_pairs(parts, pairs_per_part)
    print(f'BREC RPC evaluation on {dataset} ({num_relabel} relabelings/graph, '
          f'{sum(len(ids) for _, ids in selection)} pairs, {epochs} epochs/pair, '
          f'device {evaluator.device}, threshold {THRESHOLD})')
    if epochs != EPOCH or pairs_per_part is not None or parts is not None or num_relabel != 32:
        print('NOTE: this is a reduced run — not a citable BREC score '
              '(the protocol is 400 pairs, 32 relabelings, 50 epochs).')

    records: list[dict] = []
    part_scores: dict[str, tuple[int, int]] = {}
    fail_in_reliability = 0
    degenerate = 0
    for name, pair_ids in selection:
        part_correct = 0
        for pair_id in pair_ids:
            record = evaluator.evaluate_pair(pair_id)
            record['part'] = name
            records.append(record)
            part_correct += record['distinguished']
            fail_in_reliability += not record['reliable']
            degenerate += record['degenerate']
            print(f"  [{name}] pair {pair_id}: T2={record['t2_traintest']:.3f} "
                  f"(control {record['t2_reliability']:.3f}) -> "
                  f"{'distinguished' if record['distinguished'] else 'not distinguished'}"
                  f"{'' if record['reliable'] else ' [RELIABILITY FAIL]'}"
                  f"{' [DEGENERATE]' if record['degenerate'] else ''} "
                  f"[{record['seconds']}s]")
        part_scores[name] = (part_correct, len(pair_ids))
        print(f'{name}: {part_correct} / {len(pair_ids)}')

    total_correct = sum(correct for correct, _ in part_scores.values())
    total_pairs = sum(count for _, count in part_scores.values())
    if evaluator.output_dim is not None and num_relabel <= evaluator.output_dim:
        # cov(D) is a (output_dim x output_dim) matrix estimated from num_relabel
        # samples: with num_relabel <= output_dim it is singular by construction,
        # and pinv turns the near-null directions into enormous statistics. The
        # protocol's 32 relabelings against 16 dimensions avoid this.
        print(f'WARNING: {num_relabel} relabelings for a {evaluator.output_dim}-dimensional embedding — '
              f'the T2 covariance is rank-deficient by construction and the statistic is inflated. '
              f'Scores from this run are not comparable to published BREC numbers; '
              f'use a dataset with more than {evaluator.output_dim} relabelings.')
    print('--- BREC summary ---')
    for name, (correct, count) in part_scores.items():
        print(f'{name:20s} {correct:4d} / {count}')
    print(f'{"Total":20s} {total_correct:4d} / {total_pairs}')
    print(f'Failed reliability checks: {fail_in_reliability} / {total_pairs}')
    if degenerate:
        print(f'WARNING: {degenerate} / {total_pairs} pairs are degenerate — the model separates the '
              f'two graphs identically for every relabeling, so cov(D) is zero and the '
              f'pseudo-inverse forces T2 to 0 (a false negative). Re-run with '
              f'--epsilon-matrix {EPSILON_MATRIX} to score those pairs.')

    if results_file is None:
        # the selection goes into the file name so that sharded runs (pairs are
        # independent, so splitting --parts across processes is the natural way
        # to parallelize a full evaluation) do not overwrite each other
        suffix = ''
        if parts is not None:
            suffix += '_' + '-'.join(name for name, _ in selection)
        if pairs_per_part is not None:
            suffix += f'_first{pairs_per_part}'
        results_file = Path(configuration['paths']['results']) / dataset / f'brec_rpc_{config_id}{suffix}.csv'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['part', 'pair_id', 't2_traintest', 't2_reliability',
                                               'loss', 'distinguished', 'reliable', 'degenerate', 'seconds'])
        writer.writeheader()
        writer.writerows({key: record[key] for key in writer.fieldnames} for record in records)
    print(f'Per-pair results written to {results_file}')

    return {'total': (total_correct, total_pairs), 'parts': part_scores,
            'fail_in_reliability': fail_in_reliability, 'degenerate': degenerate, 'records': records}


@click.command()
@click.option('--dataset', default='BREC', help="BREC variant: 'BREC' (official) or 'BREC-r<k>'")
@click.option('--config', 'config_path', type=Path, default=CONFIG_PATH, help='Main config file')
@click.option('--config_id', '--config-id', 'config_id', default=0, help='Hyperparameter configuration index')
@click.option('--parts', default=None, help='Comma-separated categories to evaluate (default: all)')
@click.option('--pairs', 'pairs_per_part', default=None, type=int, help='Evaluate only the first N pairs per category')
@click.option('--epochs', default=EPOCH, help=f'Training epochs per pair (protocol: {EPOCH})')
@click.option('--epsilon_matrix', '--epsilon-matrix', 'epsilon', default=0.0, type=float,
              help=f'Ridge on the T-squared covariance (reference default 0.0; use {EPSILON_MATRIX} '
                   f'when the run reports degenerate pairs)')
@click.option('--num_threads', '--num-threads', 'num_threads', default=1, help='Preprocessing threads')
@click.option('--skip_preprocessing', '--skip-preprocessing', 'skip_preprocessing', is_flag=True,
              help='Assume dataset, labels and properties are already on disk')
def main(dataset, config_path, config_id, parts, pairs_per_part, epochs, epsilon, num_threads, skip_preprocessing):
    run_brec(dataset=dataset, config_path=config_path, config_id=config_id, parts=parts,
             pairs_per_part=pairs_per_part, epochs=epochs, epsilon=epsilon, num_threads=num_threads,
             skip_preprocessing=skip_preprocessing)


if __name__ == '__main__':
    main()
