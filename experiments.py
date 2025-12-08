import multiprocessing
from multiprocessing import Pool
from os import listdir
from os.path import join

import cpmpy as cp

from SimplifySeq.algorithms import construct_greedy, UNSAT, filter_sequence, relax_sequence

import proof2seq
from benchmarks.jobshop import generate_unsat_jobshop_model
from benchmarks.sudoku import generate_unsat_sudoku_model
from proof2seq.parsing import PumpkinProofParser
from proof2seq.pipeline import compute_sequence
from proof2seq.utils import get_sequence_statistics, sanity_check_sequence
import time
import pandas as pd
import tempfile

NUM_WORKERS = multiprocessing.cpu_count() - 1 # leave one thread available for safety
TIMEOUT = 3600
NUM_WORKERS = 1

def run_configs_on_model(model, configs, proof_name=None):

    if proof_name is None:
        file = tempfile.NamedTemporaryFile(delete=False).name
    else:
        file = proof_name

    start = time.time()
    solver = PumpkinProofParser(model)
    assert solver.solve(proof=file) is False, "Only support proofs of unsatisfiability for now"
    solve_time = time.time() - start

    results = []
    for kwargs in configs:
        type = kwargs['type']
        del kwargs['type']
        start = time.time()
        try:
            proof2seq.START_TIME = start
            if type == 'proof':
                # set verbosity and do_sanity_check to false for proper timing results
                seq = compute_sequence(model, verbose=1, do_sanity_check=False,
                                       pumpkin_solver=solver,time_limit=TIMEOUT,
                                       **kwargs)
                end = time.time()
            elif type == 'stepwise':
                start = time.time()
                seq = construct_greedy(model.constraints, UNSAT, time_limit=TIMEOUT, seed=0)
                seq = filter_sequence(seq, UNSAT, time_limit=TIMEOUT-(time.time()-start))
                seq = relax_sequence(seq, mus_solver="exact", time_limit=TIMEOUT-(time.time()-start))

                end = time.time()

                seq = [dict(input_lits=list(step['input']),
                            constraints=list(step['constraints']),
                            output_lits=list(step['output'])) for step in seq]

            else:
                raise ValueError("Unexpected type:", kwargs['type'])

            results.append(dict(
                runtime=solve_time + (end - start),
                timeout=False,
                **get_sequence_statistics(seq),
                **kwargs,
            ))
        except TimeoutError as e:
            results.append(dict(timeout=True,
                                runtime = solve_time + (time.time() - start),
                                timeout_reason = str(e),
                                **kwargs))

    return results

def run_experiments(models, configs):

    num_workers = NUM_WORKERS
    if NUM_WORKERS == 1:
        results = [run_configs_on_model(model, configs, proof_name=f"proof_{i}.drcp.gz") for i, model in enumerate(models)]
    else:
        with Pool(num_workers) as p:
            results = p.starmap(_wrap_func, [(run_configs_on_model, dict(model=model, configs=configs, proof_name=f"proof_{i}.drcp.gz")) for i,model in enumerate(models)])

    results = [x for lst in results for x in lst]# each function returns a list of results
    df = pd.DataFrame(results)
    return df


def _wrap_func(function, dict_with_arguments):
    return function(**dict_with_arguments)

def plot_runtime(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df['method'] = df['minimization_phase1'].astype(str) + "+" + df['minimization_phase2'].astype(str)

    df  = df[df['timeout'] == False]
    fig = sns.ecdfplot(
        df,
        x = "runtime",
        hue = "method",
        stat="count"
    )

    plt.show()


if __name__ == "__main__":


    configs = [
        dict(type="proof", minimization_phase1="trim", minimization_phase2="trim"),
        dict(type="proof", minimization_phase1="trim", minimization_phase2="local"),
        dict(type="proof", minimization_phase1="proof", minimization_phase2="global"),
        dict(type="proof", minimization_phase1="local", minimization_phase2="proof"),
        dict(type="proof", minimization_phase1="local", minimization_phase2="local"),
        dict(type="proof", minimization_phase1="global", minimization_phase2="proof"),
        dict(type="proof", minimization_phase1="global", minimization_phase2="local"),
        dict(type="stepwise")
    ]

    models = []

    benchmark = "sudoku"
    num_experiments = 100
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

    # plot_runtime(experiment_result)


