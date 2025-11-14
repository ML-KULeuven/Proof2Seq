
import cpmpy as cp

def sudoku_model(givens):
    givens = np.array(givens)
    assert givens.ndim == 2

    dim = givens.shape[0]
    bsize = int(dim ** 0.5)

    cells = cp.intvar(1, dim, shape=givens.shape, name="cells")

    model = cp.Model()

    model += [cp.AllDifferent(row) for row in cells]
    model += [cp.AllDifferent(col) for col in cells.T]

    for i in range(0, dim, bsize):
        for j in range(0, dim, bsize):
            model += cp.AllDifferent(cells[i:i + bsize, j:j + bsize])

    model += cells[givens != 0] == givens[givens != 0]

    return model, cells


if __name__ == '__main__':
    import numpy as np

    e = 0  # value for empty cells
    given = np.array([
        [4, e, e, 2, e, 5, e, e, e],
        [e, 9, e, e, e, e, 7, 3, e],
        [e, e, 2, e, e, 9, e, 6, e],

        [2, e, e, e, e, e, 4, e, 9],
        [e, e, e, e, 7, e, e, e, e],
        [6, e, 9, e, e, e, e, e, 1],

        [e, 8, e, 4, e, e, 1, e, e],
        [e, 6, 3, e, e, e, e, 8, e],
        [e, e, e, 6, e, 8, e, e, e]])

    model,cells = sudoku_model(given)

    from proof2seq.pipeline import compute_sequence

    compute_sequence(model,
                     minimization_phase1="trim",
                     minimization_phase2="global",
                     mus_solver="exact",
                     verbose=2)



