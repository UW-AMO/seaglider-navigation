from pathlib import Path

from adcp.exp import run
import adcp.exp.psearch2d_trials as trials
from adcp.exp.psearch2d_trials import trial2, trial2c, trial2d, trial2e

trials_folder = Path(__file__).parent / "adcp" / "exp" / "psearch2d_trials"

trial2.__main__()
trial2c.__main__()
run(**trials.trial12a._asdict(), trials_folder=trials_folder)
run(**trials.trial12c._asdict(), trials_folder=trials_folder)
run(**trials.trial18a._asdict(), trials_folder=trials_folder)
run(**trials.trial18c._asdict(), trials_folder=trials_folder)
run(**trials.trial20a._asdict(), trials_folder=trials_folder)
run(**trials.trial20c._asdict(), trials_folder=trials_folder)
trial2d.__main__()
trial2e.__main__()
run(**trials.trial12d._asdict(), trials_folder=trials_folder)
run(**trials.trial12e._asdict(), trials_folder=trials_folder)
run(**trials.trial18d._asdict(), trials_folder=trials_folder)
run(**trials.trial18e._asdict(), trials_folder=trials_folder)
run(**trials.trial20d._asdict(), trials_folder=trials_folder)
run(**trials.trial20e._asdict(), trials_folder=trials_folder)
