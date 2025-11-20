import multiprocessing
from multiprocessing import Pool
from os import listdir
from os.path import join

import cpmpy as cp

from benchmarks.jobshop import generate_unsat_jobshop_model
from benchmarks.sudoku import generate_unsat_sudoku_model
from proof2seq.parsing import PumpkinProofParser
from proof2seq.pipeline import compute_sequence
from proof2seq.utils import get_sequence_statistics
import time
import pandas as pd
import tempfile

NUM_WORKERS = multiprocessing.cpu_count() - 1 # leave one thread available for safety
NUM_WORKERS = 1

def run_configs_on_model(model, configs):

    file = tempfile.NamedTemporaryFile(delete=False).name

    start = time.time()
    solver = PumpkinProofParser(model)
    assert solver.solve(proof=file) is False, "Only support proofs of unsatisfiability for now"
    solve_time = time.time() - start

    results = []
    for kwargs in configs:
        start = time.time()
        # set verbosity and do_sanity_check to false for proper timing results
        seq = compute_sequence(model, verbose=0, do_sanity_check=True, pumpkin_solver=solver,**kwargs)
        end = time.time()

        results.append(dict(
            runtime=solve_time + (end - start),
            **get_sequence_statistics(seq),
            **kwargs
        ))

    return results

def run_experiments(models, configs):

    num_workers = NUM_WORKERS
    with Pool(num_workers) as p:
        results = p.starmap(_wrap_func, [(run_configs_on_model, dict(model=model, configs=configs)) for model in models])
        results = [x for lst in results for x in lst]# each function returns a list of results
    df = pd.DataFrame(results)
    return df


def _wrap_func(function, dict_with_arguments):
    return function(**dict_with_arguments)

def plot_runtime(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df['method'] = df['minimization_phase1'].astype(str) + "+" + df['minimization_phase2'].astype(str)

    fig = sns.ecdfplot(
        df,
        x = "runtime",
        hue = "method",
        stat="count"
    )

    plt.show()


if __name__ == "__main__":


    configs = [
        # TODO: replace "proof" with "trim" after bug in Pumpkin is fixed.
        dict(minimization_phase1="trim", minimization_phase2="trim"),
        dict(minimization_phase1="trim", minimization_phase2="local"),
        dict(minimization_phase1="trim", minimization_phase2="global"),
        dict(minimization_phase1="local", minimization_phase2="proof"),
        dict(minimization_phase1="local", minimization_phase2="local"),
        dict(minimization_phase1="global", minimization_phase2="proof"),
        dict(minimization_phase1="global", minimization_phase2="local"),
    ]

    models = []

    benchmark = "jobshop"
    num_experiments = 5
    only_plot = False

    if benchmark == "sudoku":
        models = [generate_unsat_sudoku_model("benchmarks/expert_sudokus.csv", seed=i) for i in range(num_experiments)]

    if benchmark == "jobshop":
        models = [generate_unsat_jobshop_model(n_machines=5, n_jobs=5, horizon=50, factor=0.999999, seed=i) for i in range(num_experiments)]

    if benchmark == "modeling_examples":
        model_dir = "benchmarks/modeling_examples"
        models = [cp.Model.from_file(join(model_dir,fname)) for fname in sorted(listdir(model_dir))[:num_experiments]]

    if only_plot is False:
        experiment_result = run_experiments(models, configs)
        experiment_result.to_pickle(f"{benchmark}_experiments.df")
    else:
        experiment_result = pd.read_pickle(f"{benchmark}_experiments.df")

    plot_runtime(experiment_result)


