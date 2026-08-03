"""Shared helpers for the paper-figure scripts.

These scripts render LaTeX/pgf figures and previously duplicated the same
rcParams/pgf setup and a byte-identical ``CustomColorMap`` in several places.
The rcParams setup now lives in :func:`setup_pgf`, the savefig boilerplate in
:func:`save_latex_figure`, and the colormap is imported from the canonical
(tested) copy in ``simplegnn.datasets.utils.graph_drawing``.

base_paper is not a package, so sibling scripts import this module by having
``experiments/base_paper/src`` on ``sys.path`` (the directory of the running
script, or added explicitly by scripts living elsewhere).

Output locations are defined here as well (:data:`FIGURE_DIR`,
:data:`TABLE_DIR`, :data:`POSITION_DIR`): every generated figure and table of
the paper goes into one ``results/base_paper`` tree instead of being scattered
over per-experiment subfolders. Paths are relative to the repository root, so
run the scripts from there.
"""
from pathlib import Path

import matplotlib.pyplot as plt

#: Root of all generated paper artifacts (gitignored via ``results/``).
RESULTS_DIR = Path('results/base_paper')
#: Figures (pdf/png) of every experiment group.
FIGURE_DIR = RESULTS_DIR / 'figures'
#: Cached graph layouts, shared by all figure scripts. Keeping these pins the
#: Kamada-Kawai positions so regenerated figures stay comparable across runs.
POSITION_DIR = FIGURE_DIR / 'positions'
#: LaTeX tables and the json caches feeding them.
TABLE_DIR = RESULTS_DIR / 'tables'


def figure_path(name):
    """Path of figure ``name`` inside :data:`FIGURE_DIR`."""
    return FIGURE_DIR / name


def table_path(name):
    """Path of table/cache file ``name`` inside :data:`TABLE_DIR`."""
    return TABLE_DIR / name


def position_path(name):
    """Path of layout-cache file ``name``, creating :data:`POSITION_DIR`."""
    POSITION_DIR.mkdir(parents=True, exist_ok=True)
    return POSITION_DIR / name


def setup_pgf(font_size=30):
    """Configure matplotlib for serif LaTeX/pgf figure output.

    Parameters
    ----------
    font_size : int, optional
        Base font size passed to ``font.size`` (default: 30, matching the
        molecule figures). The ablation/visualization figures pass smaller
        sizes (12–18).
    """
    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "font.size": font_size,
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "pgf.texsystem": "lualatex",
        "pgf.preamble": "\n".join([
            r"\usepackage{url}",  # load additional packages
            r"\usepackage{unicode-math}",  # unicode math setup
            r"\setmainfont{DejaVu Serif}",  # serif font via preamble
        ])
    })


def save_latex_figure(fig, path):
    """Save ``fig`` to ``path`` with the shared pgf/tight-bbox settings.

    Creates the parent directory if needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or pathlib.Path
        Destination file path (``.pdf`` for pgf output).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', backend='pgf')
    plt.close(fig)


def save_raster_figure(fig, path):
    """Save ``fig`` to ``path`` with the default backend (raster output).

    Used for the scatter plots with ~10^5 points, where the pgf backend of
    :func:`save_latex_figure` is not worth the compile cost; text is still
    rendered by LaTeX via the ``text.usetex`` rcParam of :func:`setup_pgf`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


_experiment_cache = {}
_model_cache = {}


def get_experiment(config_path):
    """Memoized ``FrameworkMain`` per main-config path.

    Constructing ``FrameworkMain`` parses every dataset config referenced by
    the main config; the figure scripts request the same config repeatedly.
    """
    from simplegnn.framework.core import FrameworkMain
    key = str(config_path)
    if key not in _experiment_cache:
        _experiment_cache[key] = FrameworkMain(Path(config_path))
    return _experiment_cache[key]


def get_model(config_path, db_name, config_id=0, run_id=0, validation_id=0,
              best=True, experiment_db_id=0):
    """Memoized trained-model loader.

    ``FrameworkMain.load_ordinary_model`` re-runs the full dataset
    preprocessing (dataset + ShareGNN labels/properties from disk) on every
    call, which dominates figure-script runtime when the same
    ``(config, dataset)`` pair is drawn several times. Use
    :func:`clear_model_cache` between dataset groups to bound memory.
    """
    key = (str(config_path), db_name, config_id, run_id, validation_id, best,
           experiment_db_id)
    if key not in _model_cache:
        experiment = get_experiment(config_path)
        _model_cache[key] = experiment.load_ordinary_model(
            db_name=db_name, config_id=config_id, run_id=run_id,
            validation_id=validation_id, best=best,
            experiment_db_id=experiment_db_id)
    return _model_cache[key]


def clear_model_cache():
    """Drop cached experiments and models (each holds a full dataset)."""
    _experiment_cache.clear()
    _model_cache.clear()
