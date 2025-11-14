from os import listdir
from os.path import join

import cpmpy as cp

from benchmarks.jobshop import generate_unsat_jobshop_model
from benchmarks.sudoku import generate_unsat_sudoku_model
from proof2seq.pipeline import compute_sequence
from proof2seq.utils import get_sequence_statistics
import time
import pandas as pd

def run_experiment(models, **kwargs):

    statistics = []
    for model in models:
        model.to_file("sudoku_model.pickle")
        start = time.time()
        # set do_sanity_check to False for proper timing results
        seq = compute_sequence(model, verbose=0, do_sanity_check=False, **kwargs)
        end = time.time()
        statistics.append(dict(runtime=end-start, **get_sequence_statistics(seq), **kwargs))

    return pd.DataFrame(statistics)


if __name__ == "__main__":


    configs = [
        dict(minimization_phase1="trim", minimization_phase2="proof"),
        dict(minimization_phase1="trim", minimization_phase2="local"),
        dict(minimization_phase1="trim", minimization_phase2="global"),
        dict(minimization_phase1="local", minimization_phase2="proof"),
        dict(minimization_phase1="local", minimization_phase2="local"),
        dict(minimization_phase1="global", minimization_phase2="proof"),
        dict(minimization_phase1="global", minimization_phase2="local"),
    ]

    models = []

    # Uncomment to run the sudoku experiments
    # models = [generate_unsat_sudoku_model("benchmarks/expert_sudokus.csv", seed=i) for i in range(100)]

    # Uncomment to run the jobshop experiments
    # models = [generate_unsat_jobshop_model(n_machines=5, n_jobs=5, horizon=50, factor=0.999999, seed=i) for i in range(100)]

    # Uncomment to run the modeling examples experiments
    # model_dir = "benchmarks/modeling_examples"
    # models = [cp.Model.from_file(join(model_dir,fname)) for fname in sorted(listdir(model_dir))]

    dfs = []
    for config in configs:
        print(config)
        stats = run_experiment(models, **config)
        dfs.append(stats)

    df = pd.concat(dfs, ignore_index=True)

    grouped = df.groupby(["minimization_phase1", "minimization_phase2"])[['runtime','length', 'max_cons']].agg(['count', 'mean', 'std'])
    print(grouped.to_markdown())
