from pathlib import Path
import argparse

import adcp.exp
import adcp.exp.live_trials as lt

parser = argparse.ArgumentParser(description="Run a trial on actual data.")
parser.add_argument("trial", metavar="trial", help="trial id to run")
parser.add_argument("variant", metavar="variant", help="variant id to run")
parser.add_argument("--debug", action="store_true", help="debug mode?")
parser.add_argument(
    "--dpi", action="store", default=72, help="matplotlib figure dpi."
)


namespace = parser.parse_args()
try:
    trial = getattr(lt, "trial" + namespace.trial)
except NameError:
    raise NameError(f"There is no trial named trial{namespace.trial}")
try:
    variant = getattr(lt, "var_" + namespace.variant)
except NameError:
    raise NameError(f"There is no trial named var_{namespace.variant}")


adcp.exp.run(
    trial.ex,
    prob_params=trial.solve_params,
    sim_params=variant.data_params,
    logfile="live_trials.db",
    trials_folder=Path(__file__).absolute().parent,
    debug=namespace.debug,
    matplotlib_dpi=namespace.dpi,
)
