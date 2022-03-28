from pathlib import Path
from itertools import product

from adcp import exp
import adcp.exp.live_trials as trials

trials_folder = Path(__file__).parent / "adcp" / "exp" / "live_trials"


ts = (
    "trial22",
    "trial23",
    "trial24",
    "trial25",
    "trial26",
    "trial27",
    "trial28",
)
vs = ("var_a",)
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
