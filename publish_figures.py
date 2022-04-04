from itertools import product
from pathlib import Path

from adcp.exp import run
import adcp.exp.psearch2d_trials as p2t

trials_folder = Path(__file__).parent / "adcp" / "exp" / "psearch2d_trials"

trials = (2, 12, 18, 20)
variants = ("a", "c", "d", "e")

for t, v in product(trials, variants):
    trial = getattr(p2t, "trial" + str(t))
    variant = getattr(p2t, "var_" + v)
    run(
        variant.ex,
        prob_params=trial.solve_params,
        sim_params=variant.sim_params,
        logfile="trials.db",
        trials_folder=trials_folder,
        debug=False,
    )
