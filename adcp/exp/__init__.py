from pathlib import Path
from datetime import datetime, timezone

import git

repo = git.Repo(Path(__file__).parent.parent.parent)


class Experiment:
    def run():
        raise NotImplementedError


def run(ex: Experiment):
    if repo.is_dirty():
        raise RuntimeError(
            "Git Repo is dirty.  For repeatable tests,"
            " clean the repo by committing or stashing all changes and "
            "untracked files."
        )
    print(f"Current repo hash: {repo.head.commit.hexsha}")
    utc_now = datetime.now(timezone.utc)
    print(
        f"Running experiment {ex.name} at time: ",
        utc_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )
    ex.run()
    utc_now = datetime.now(timezone.utc)
    print(
        "Finished experiment at time: ",
        utc_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )
