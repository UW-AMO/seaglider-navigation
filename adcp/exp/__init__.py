from math import isnan
from pathlib import Path
from typing import List
from datetime import datetime, timezone
import logging
import git
from time import process_time
from jsonschema.validators import create

import pandas as pd
from sqlalchemy import (
    create_engine,
    inspection,
    insert,
    Table,
    Column,
    Integer,
    Float,
    String,
    MetaData,
)

from adcp.exp._trial_types import trial_df, variant_df

trial_db = Path(__file__).resolve().parent / "trials.db"
eng = create_engine("sqlite:///" + str(trial_db))
trial_df.to_sql("trial_types", eng, if_exists="replace")
variant_df.to_sql("variant_types", eng, if_exists="replace")


repo = git.Repo(Path(__file__).parent.parent.parent)
trials_columns = [
    Column("id", Integer, primary_key=True),
    Column("variant", Integer, primary_key=True),
    Column("iteration", Integer, primary_key=True),
    Column("commit", String, nullable=False),
    Column("cpu_time", Float),
    Column("results", String, nullable=False),
    Column("filename", String, nullable=False),
    Column("overflow", String),
]


class DBHandler(logging.Handler):
    def __init__(
        self,
        filename: str,
        table_name: str,
        cols: List[Column],
        separator: str = "--",
    ):
        self.separator = separator
        if Path(filename).is_absolute():
            self.db = Path(filename)
        else:
            self.db = Path(__file__).resolve().parent / filename

        md = MetaData()
        self.log_table = Table(table_name, md, *cols)

        self.eng = create_engine("sqlite:///" + str(self.db))
        if not inspection.inspect(self.eng).has_table(table_name):
            md.create_all(self.eng)
        super().__init__()
        self.addFilter(lambda rec: self.separator in rec.getMessage())

    def emit(self, record: logging.LogRecord):
        vals = self.parse_record(record.getMessage())
        ins = insert(self.log_table, vals)
        with self.eng.connect() as conn:
            result = conn.execute(ins)

    def parse_record(self, msg: str) -> List[str]:
        return msg.split(self.separator)[1:]


exp_logger = logging.Logger("experiments")
exp_logger.setLevel(20)
db_h = DBHandler("trials.db", "trials", trials_columns)
exp_logger.addHandler(db_h)
exp_logger.addHandler(logging.StreamHandler())


class Experiment:
    def run():
        raise NotImplementedError


def run(ex: Experiment, debug=False, trial=1, variant=1, trials_folder=None):
    if not debug and repo.is_dirty():
        raise RuntimeError(
            "Git Repo is dirty.  For repeatable tests,"
            " clean the repo by committing or stashing all changes and "
            "untracked files."
        )
    utc_now = datetime.now(timezone.utc)
    cpu_now = process_time()
    commit = repo.head.commit.hexsha
    log_msg = (
        f"Running experiment {ex.name} with variant {ex.variant} at time: "
        + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
        + f".  Current repo hash: {commit}"
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
    cpu_time = process_time() - cpu_now

    # Save save results in trial database
    with db_h.eng.connect() as conn:
        trials_df = pd.read_sql("SELECT * FROM trials", conn)
    matching_rows = (trials_df["id"] == trial) & (
        trials_df["variant"] == variant
    )
    next_iteration = trials_df.loc[matching_rows, "iteration"].max() + 1
    if isnan(next_iteration):
        next_iteration = 1
    new_filename = f"trial{trial}_{variant}_{next_iteration}.ipynb"
    exp_logger.info(
        "trial entry"
        + f"--{trial}"
        + f"--{variant}"
        + f"--{next_iteration}"
        + f"--{commit}"
        + f"--{cpu_time}"
        + f'--{results["metrics"]}'
        + f"--{new_filename}--"
    )
