from pathlib import Path

import git

repo = git.Repo(Path(__file__).parent.parent.parent)


class Experiment:
    def __init__(self):
        pass

    def run():
        pass


def run(exp: Experiment):
    exp.run()
