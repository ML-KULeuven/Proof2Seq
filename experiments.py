import multiprocessing
from multiprocessing import Pool
from os import listdir
from os.path import join

import cpmpy as cp

from benchmarks.jobshop import generate_unsat_jobshop_model
from benchmarks.sudoku import generate_unsat_sudoku_model
from proof2seq.pipeline import compute_sequence
from proof2seq.utils import get_sequence_statistics
import time
import pandas as pd
import tempfile

NUM_WORKERS = multiprocessing.cpu_count() - 1 # leave one thread available for safety


def run_one_experiment(model, **kwargs):

    file = tempfile.NamedTemporaryFile(delete=False).name

    start = time.time()
    # set verbosity and do_sanity_check to false for proper timing results
    seq = compute_sequence(model, verbose=0, do_sanity_check=False, proof_name=file, **kwargs)
    end = time.time()

    return dict(runtime=end-start,
                **get_sequence_statistics(seq),
                **kwargs)

def run_experiments(models, configs):

    experiments = [dict(model=model, **config) for model in models for config in configs]

    num_workers = NUM_WORKERS
    with Pool(num_workers) as p:
        results = p.starmap(_wrap_func, [(run_one_experiment, exp) for exp in experiments])
        df = pd.DataFrame(results)
    return df


def _wrap_func(function, dict_with_arguments):
    return function(**dict_with_arguments)

def plot_runtime(df):
    import seaborn as sns

    df['method'] = df['minimization_phase1'].astype(str) + "+" + df['minimization_phase2'].astype(str)

    sns.ecdfplot(
        df,
        x = "runtime",
        hue = "method",
        stat="count"
    )



if __name__ == "__main__":


    configs = [
        # TODO: replace "proof" with "trim" after bug in Pumpkin is fixed.
        dict(minimization_phase1="proof", minimization_phase2="proof"),
        # dict(minimization_phase1="proof", minimization_phase2="local"),
        dict(minimization_phase1="proof", minimization_phase2="global"),
        # dict(minimization_phase1="local", minimization_phase2="proof"),
        # dict(minimization_phase1="local", minimization_phase2="local"),
        dict(minimization_phase1="global", minimization_phase2="proof"),
        # dict(minimization_phase1="global", minimization_phase2="local"),
    ]

    models = []

    benchmark = "sudoku"

    if benchmark == "sudoku":
        models = [generate_unsat_sudoku_model("benchmarks/expert_sudokus.csv", seed=i) for i in range(5)]

    if benchmark == "jobshop":
        models = [generate_unsat_jobshop_model(n_machines=5, n_jobs=5, horizon=50, factor=0.999999, seed=i) for i in range(100)]

    if benchmark == "modeling_examples":
        model_dir = "benchmarks/modeling_examples"
        models = [cp.Model.from_file(join(model_dir,fname)) for fname in sorted(listdir(model_dir))]

    experiment_result = run_experiments(models, configs)
    experiment_result.to_pickle(f"{benchmark}_experiments.df")

    plot_runtime(experiment_result)


