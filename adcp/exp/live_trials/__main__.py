from pathlib import Path
import argparse

import adcp.exp
import adcp.exp.live_trials as lt

parser = argparse.ArgumentParser(description="Run a trial on actual data.")
parser.add_argument("trial", metavar="trial", help="trial id/variant to run")
parser.add_argument("--debug", action="store_true", help="debug mode?")


namespace = parser.parse_args()
try:
    trial = getattr(lt, namespace.trial)
except NameError:
    try:
        trial = lt.__getattr__(lt, namespace.trial)
    except AttributeError:
        raise NameError(f"There is no trial named {namespace.trial}")

adcp.exp.run(
    trial.ex,
    prob_params=trial.prob_params,
    sim_params=trial.sim_params,
    logfile="live_trials.db",
    trials_folder=Path(__file__).absolute().parent,
    debug=namespace.debug,
)
