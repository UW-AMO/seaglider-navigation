from pathlib import Path
from itertools import product

from adcp import exp
import adcp.exp.live_trials as trials

trials_folder = Path(__file__).parent / "adcp" / "exp" / "live_trials"


ts = ("trial5", "trial6", "trial7", "trial8")
vs = ("var_a", "var_b", "var_c", "var_d", "var_e", "var_f", "var_g", "var_h")
t_v = product(ts, vs)
for t, v in t_v:
    trial = getattr(trials, t)
    variant = getattr(trials, v)
    exp.run(
        trials.Cabage17,
        prob_params=trial.solve_params,
        sim_params=variant.data_params,
        logfile="live_trials.db",
        trials_folder=trials_folder,
    )
