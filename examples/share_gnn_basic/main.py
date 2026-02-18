from pathlib import Path

from simplegnn.framework.core import FrameworkMain


def main():
    experiment = FrameworkMain(Path('examples/share_gnn_basic/main.yml'))
    experiment.preprocessing(1)
    experiment.run_configurations(-1)
    experiment.evaluate_results()
    experiment.run_best_configuration(-1)
    experiment.evaluate_results(evaluate_best_model=True)

if __name__ == '__main__':
    main()