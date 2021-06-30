from pathlib import Path
import argparse

import adcp.exp
import adcp.exp.psearch2d_trials as p2t

parser = argparse.ArgumentParser(description="Run a trial.")
parser.add_argument("trial", metavar="trial", help="trial id/variant to run")


namespace = parser.parse_args()
try:
    trial = getattr(p2t, namespace.trial)
except NameError:
    try:
        trial = p2t.__getattr__(p2t, namespace.trial)
    except AttributeError:
        raise NameError(f"There is no trial named {namespace.trial}")
adcp.exp.run(
    trial.experiment,
    prob_params=trial.prob_params,
    sim_params=trial.sim_params,
    trials_folder=Path(__file__).absolute().parent,
)
