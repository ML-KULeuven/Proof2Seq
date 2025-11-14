import random
import math
import pandas as pd

import cpmpy as cp


def generate_jobshop_instance(n_jobs, n_machines, horizon, seed=0):
    """
        Generate a jobshop instance with n_jobs and n_machines and a horizon.
    """

    random.seed(seed)
    maxtime = horizon // (n_jobs + 1)

    duration = [[0] * n_machines for i in range(0, n_jobs)]
    for i in range(0, n_jobs):
        for j in range(0, n_machines):
            duration[i][j] = random.randint(1, maxtime)

    order = [list(range(0, n_machines)) for i in range(0, n_jobs)]
    for i in range(0, n_jobs):
        random.shuffle(order[i])

    # return pandas dataframe with jobs and tasks
    return pd.DataFrame([
        dict(job_id=i, task_id=j, machine=order[i][j], duration=duration[i][j]) for i in range(n_jobs) for j in
        range(n_machines)
    ])


def jobshop_model(jobs: pd.DataFrame, horizon: int, with_precedences=True):
    """
        Construct jobshop model from data.
        jobs: list of tasks with given duration and machine to run on
                tasks in a job are ordered (i.e., have precedence constraints between them)
        horizon (optional):maximum time to complete all jobs
    """

    # decision variables
    start = cp.intvar(0, horizon, shape=len(jobs), name="s")
    for i, var in enumerate(start):
        var.name = f"start_{i}"
    end = cp.intvar(0, horizon, shape=len(jobs), name="e")
    for i, var in enumerate(end):
        var.name = f"end_{i}"

    model = cp.Model()

    # add precedence constraints
    if with_precedences:
        for idx, group in jobs.groupby("job_id"):
            for t1 in group.index[:-1]:
                for t2 in group.index[t1 + 1:]:
                    model += end[t1] <= start[t2]

    # add cumulative constraints for each set of tasks on the same machine
    for machine, group in jobs.groupby("machine"):
        idx = group.index
        model += cp.Cumulative(start[idx], group["duration"].tolist(), end[idx], demand=1, capacity=1)

    model.minimize(cp.max(end))

    return model

def generate_unsat_jobshop_model(n_jobs, n_machines, horizon, seed=0, factor=1, with_precedences=True):
    jobs = generate_jobshop_instance(n_jobs, n_machines, horizon, seed)
    model = jobshop_model(jobs, horizon, with_precedences=with_precedences)
    assert model.solve(solver="ortools",
                       num_workers=1)  # force num-workers to 1 to avoid parallelization in experiments

    unsat_horizon = math.floor(factor * model.objective_value())

    return jobshop_model(jobs, unsat_horizon, with_precedences=with_precedences)

