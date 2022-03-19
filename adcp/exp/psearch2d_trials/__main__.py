from pathlib import Path
import argparse

import adcp.exp
import adcp.exp.psearch2d_trials as p2t

parser = argparse.ArgumentParser(description="Run a trial.")
parser.add_argument("trial", metavar="trial", help="trial id to run")
parser.add_argument("variant", metavar="variant", help="variant id to run")
parser.add_argument("--debug", action="store_true", help="debug mode?")


namespace = parser.parse_args()
try:
    trial = getattr(p2t, "trial" + namespace.trial)
except NameError:
    raise NameError(f"There is no trial named trial{namespace.trial}")
try:
    variant = getattr(p2t, "var_" + namespace.variant)
except NameError:
    raise NameError(f"There is no trial named var_{namespace.variant}")

adcp.exp.run(
    variant.ex,
    prob_params=trial.solve_params,
    sim_params=variant.sim_params,
    trials_folder=Path(__file__).absolute().parent,
    debug=namespace.debug,
)
