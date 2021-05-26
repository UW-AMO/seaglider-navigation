from pathlib import Path
from datetime import datetime, timezone
import logging

import git

repo = git.Repo(Path(__file__).parent.parent.parent)

exp_logger = logging.Logger(Path(__file__) / "experiments")
exp_logger.setLevel(20)
exp_logger.addHandler(logging.FileHandler(
    'experiment_log.txt',
    encoding='utf-8'
))
exp_logger.addHandler(logging.StreamHandler())

class Experiment:
    def run():
        raise NotImplementedError


def run(ex: Experiment, debug=False):
    if not debug and repo.is_dirty():
        raise RuntimeError(
            "Git Repo is dirty.  For repeatable tests,"
            " clean the repo by committing or stashing all changes and "
            "untracked files."
        )
    utc_now = datetime.now(timezone.utc)
    log_msg = (
        f"Running experiment {ex.name} with variant {ex.variant} at time: "
        + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
        + f".  Current repo hash: {repo.head.commit.hexsha}"
    )
    if debug:
        log_msg += ".  In debugging mode."
    exp_logger.info(log_msg)
    results = ex.run()
    utc_now = datetime.now(timezone.utc)
    exp_logger.info(
        "Finished experiment at time: "
        + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
        + f".  Results: {results['metrics']}"
    )
