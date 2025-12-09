import multiprocessing
import os
import pickle
from multiprocessing import Pool
from os import listdir
from os.path import join

import cpmpy as cp

# from SimplifySeq.algorithms import construct_greedy, UNSAT, filter_sequence, relax_sequence

import proof2seq
from benchmarks.jobshop import generate_unsat_jobshop_model
from benchmarks.sudoku import generate_unsat_sudoku_model
from proof2seq.parsing import PumpkinProofParser
from proof2seq.pipeline import compute_sequence
from proof2seq.utils import get_sequence_statistics, sanity_check_sequence
import time
import pandas as pd
import tempfile

import runexp

NUM_WORKERS = multiprocessing.cpu_count() - 1 # leave one thread available for safety
TIMEOUT = 3600
NUM_WORKERS = 1

#### Generating the Pumpkin proofs, and parsting them to a CPMpy format #####

def get_cpmpy_proof(model, proof_name=None, proof_prefix="."):
    if proof_name is None:
        proof_name = tempfile.NamedTemporaryFile(delete=False).name


    file = join(proof_prefix, proof_name)
    start = time.time()
    solver = PumpkinProofParser(model)
    assert solver.solve(proof=file) is False, "Only support proofs of unsatisfiability for now"
    t1 = time.time()
    proof = solver.read_proof_tree()
    t2 = time.time()

    return dict(
        model_and_proof=(model, proof),
        solver_cons_to_user_cons=solver.user_cons,
        timings=dict(solve_time=t1-start, parse_time=t2-t1)
    )


class ProofGenerator(runexp.Runner):

    def make_kwargs(self, config):

        model_func = eval(config['model_func'])
        model = model_func(**config['instance'])

        return dict(
            model=model,
            proof_name=config.get("proof_name", None),
            proof_prefix=config.get("proof_prefix", ".")
        )

    def description(self, config):
        return f"Computing proof for" + ",".join(f"{key}={val}" for key,val in config['instance'].items())


#### Computing a step-wise explanation starting from the Pumpkin proof #####

def get_explanation_sequence(model, cpm_proof, solver_cons_to_user_cons, timings, model_info, **kwargs):

    start = time.time()
    seq = compute_sequence(model,
                           solver_cons_to_user_cons=solver_cons_to_user_cons,
                           cpm_proof=cpm_proof,
                           verbose=0,
                           do_sanity_check=False, # set do_sanity_check to false for proper timing results
                           **kwargs)
    end = time.time()

    return dict(
        timings = dict(coversion_time=end-start, **timings),
        **get_sequence_statistics(seq),
        model_info=model_info,
    )


class ExplanationGenerator(runexp.Runner):
    """
        The purpose of this runner is to read the already generated proof, and compute the explanation sequence
        We do it this way, to ensure each method is run on the same proof
    """

    def make_kwargs(self, config):

        proof_dir = config['proof_dir'] # where the previous experiment stored the proof

        with open(join(proof_dir, "model_and_proof.pickle"), "rb") as f:
            model, proof = pickle.load(f)
        with open(join(proof_dir, "solver_cons_to_user_cons.pickle"), "rb") as f:
            solver_cons_to_user_cons = pickle.load(f)
        with open(join(proof_dir, "timings.json"), "rb") as f:
            timings = json.load(f)
        with open(join(proof_dir, "config.json"), "rb") as f:
            proof_config = json.load(f)

        return dict(
            model=model,
            cpm_proof=proof,
            solver_cons_to_user_cons=solver_cons_to_user_cons,
            model_info = proof_config['instance'],
            timings=timings,
            **config['algorithm_setup']
        )

def run_configs_on_model(model, configs, proof_name=None, proof_prefix=".", results_prefix="results/", experiment_index=None):

    if proof_name is None:
        file = tempfile.NamedTemporaryFile(delete=False).name
    else:
        file = join(proof_prefix, proof_name.replace(".gz", "")+ f"_{experiment_index}.drcp.gz")


    os.makedirs(proof_prefix, exist_ok=True)
    os.makedirs(results_prefix, exist_ok=True)

    start = time.time()
    solver = PumpkinProofParser(model)
    assert solver.solve(proof=file) is False, "Only support proofs of unsatisfiability for now"
    solve_time = time.time() - start

    results = []
    for kwargs in configs:
        kwargs = dict(kwargs)
        type = kwargs['type']
        del kwargs['type']
        start = time.time()
        try:
            proof2seq.START_TIME = start
            if type == 'proof':
                # set verbosity and do_sanity_check to false for proper timing results
                seq = compute_sequence(model, verbose=0, do_sanity_check=False,
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

    if experiment_index is not None:
        results['experiment_index'] = experiment_index

    with open(results_prefix + f"{experiment_index}.pk", "wb") as f:
        import pickle
        pickle.dump(results, f)

    return results

def run_experiments(models, configs, name=None):

    num_workers = NUM_WORKERS
    if NUM_WORKERS == 1:
        results = [run_configs_on_model(model, configs,
                                        proof_prefix=f"./proofs_{name}",
                                        proof_name=f"proof",
                                        results_prefix=f"results_{name}/",
                                        experiment_index=i) for i, model in enumerate(models)]
    else:
        with Pool(num_workers) as p:
            results = p.starmap(_wrap_func, [(run_configs_on_model, dict(model=model, configs=configs,
                                                                         proof_prefix=f"./proofs_{name}",
                                                                         proof_name=f"proof",
                                                                         results_prefix=f"results_{name}/",
                                                                         experiment_index=i)) for i,model in enumerate(models)])

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

    plt.xscale("log")
    plt.show()


# if __name__ == "__main__":


    # configs = [
    #     dict(type="proof", minimization_phase1="trim", minimization_phase2="trim"),
    #     dict(type="proof", minimization_phase1="trim", minimization_phase2="local"),
    #     dict(type="proof", minimization_phase1="trim", minimization_phase2="global"),
    #     dict(type="proof", minimization_phase1="local", minimization_phase2="proof"),
    #     dict(type="proof", minimization_phase1="local", minimization_phase2="local"),
    #     dict(type="proof", minimization_phase1="global", minimization_phase2="proof"),
    #     dict(type="proof", minimization_phase1="global", minimization_phase2="local"),
    #     # dict(type="stepwise")
    # ]
    #
    # models = []
    #
    # benchmark = "sudoku"
    # num_experiments = 5
    # only_plot = True
    #
    # if benchmark == "sudoku":
    #     models = [generate_unsat_sudoku_model("benchmarks/expert_sudokus.csv", seed=i) for i in range(num_experiments)]
    #
    # if benchmark == "jobshop":
    #     models = [generate_unsat_jobshop_model(n_machines=5, n_jobs=5, horizon=50, factor=0.999999, seed=i) for i in range(num_experiments)]
    #
    # if benchmark == "modeling_examples":
    #     model_dir = "benchmarks/modeling_examples"
    #     models = [cp.Model.from_file(join(model_dir,fname)) for fname in sorted(listdir(model_dir))[:num_experiments]]
    #
    # if only_plot is False:
    #     experiment_result = run_experiments(models, configs, name=benchmark)
    #     experiment_result.to_pickle(f"{benchmark}_experiments.df")
    # else:
    #     results = []
    #     for fname in os.listdir(f"results_{benchmark}"):
    #         with open(os.path.join(f"results_{benchmark}",fname), "rb") as f:
    #             results += pickle.load(f)
    #
    #     experiment_result = pd.DataFrame(results)
    #     # plot_runtime(experiment_result)
    #
    #     print(experiment_result.columns)


if __name__ == "__main__":

    import json
    from runexp import default_parser

    parser = default_parser()

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.loads(f.read())
        runner = eval(args.runner)(func=eval(args.func),
                                   output=args.output,
                                   memory_limit=args.memory_limit,
                                   printlog=True)

        if args.unravel is True:
            runner.run_batch(config, parallel=args.parallel, num_workers=args.num_workers)
        else:
            runner.run_one(config)
