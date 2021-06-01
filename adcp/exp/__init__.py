from pathlib import Path
from typing import List
from datetime import datetime, timezone
import logging
import git
from time import process_time
import warnings
from collections import OrderedDict

import pandas as pd
from sqlalchemy import (
    create_engine,
    inspection,
    select,
    insert,
    Table,
    Column,
    Integer,
    Float,
    String,
    MetaData,
)


REPO = git.Repo(Path(__file__).parent.parent.parent)
TRIALS_COLUMNS = [
    Column("id", Integer, primary_key=True),
    Column("variant", Integer, primary_key=True),
    Column("iteration", Integer, primary_key=True),
    Column("commit", String, nullable=False),
    Column("cpu_time", Float),
    Column("results", String, nullable=False),
    Column("filename", String, nullable=False),
    Column("overflow", String),
]
TRIAL_TYPES = [
    Column("id", Integer, primary_key=True),
    Column("short_name", String, unique=True),
    Column("prob_params", String, unique=True),
]

VARIANT_TYPES = [
    Column("variant", Integer, primary_key=True),
    Column("short_name", String, unique=True),
    Column("sim_params", String, unique=True),
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
        url = "sqlite:///" + str(self.db)
        self.eng = create_engine(url)
        with self.eng.connect() as conn:
            if not inspection.inspect(conn).has_table(table_name):
                md.create_all(conn)

        super().__init__()
        self.addFilter(lambda rec: self.separator in rec.getMessage())

    def emit(self, record: logging.LogRecord):
        vals = self.parse_record(record.getMessage())
        ins = insert(self.log_table, vals)
        with self.eng.connect() as conn:
            conn.execute(ins)

    def parse_record(self, msg: str) -> List[str]:
        return msg.split(self.separator)[1:]


class Experiment:
    def run():
        raise NotImplementedError


def _init_logger(trial_log, table_name):
    """Create a Trials logger with a database handler"""
    exp_logger = logging.Logger("experiments")
    exp_logger.setLevel(20)
    exp_logger.addHandler(logging.StreamHandler())
    db_h = DBHandler(trial_log, table_name, TRIALS_COLUMNS)
    exp_logger.addHandler(db_h)
    return exp_logger, db_h.log_table


def _init_id_variant_tables(trial_log):
    eng = create_engine("sqlite:///" + str(trial_log))
    md = MetaData()
    id_table = Table("trial_types", md, *TRIAL_TYPES)
    var_table = Table("variant_types", md, *VARIANT_TYPES)
    inspector = inspection.inspect(eng)
    if not inspector.has_table("trial_types") and not inspector.has_table(
        "variant_types"
    ):
        md.create_all(eng)
    return id_table, var_table


def _id_variant_iteration(
    trial_log, trials_table, *, var_table, sim_params, id_table, prob_params
):
    """Identify, from the db_log, which trial id and variant the current
    problem matches, then give the iteration.  If no matches are found,
    increment id or variant appropriately.

    Args:
        trial_log (path-like): location of the trial log database
        trials_table (sqlalchemy.Table): the main record of each
            trial/variant
        var_table (sqlalchemy.Table): the lookup table for simulation
            variants
        sim_params (dict): parameters used in simulated experimental
            data
        id_table (sqlalchemy.Table): the lookup table for trial ids
        prob_params (dict): Parameters used to create the problem/solver
            in the experiment
    """
    eng = create_engine("sqlite:///" + str(trial_log))
    sim_params = dict(sim_params)
    prob_params = dict(prob_params)

    def lookup_or_add_params(tb, cols, params, index, lookup_col):
        params = OrderedDict({k: v for k, v in sorted(dict(params).items())})
        df = pd.read_sql(select(tb), eng)
        ind_equal = df.loc[:, lookup_col] == str(params)
        if ind_equal.sum() == 0:
            new_val = 1 if df.empty else df[index].max() + 1
            stmt = insert(tb, values={index: new_val, lookup_col: str(params)})
            eng.execute(stmt)
            return new_val, True
        else:
            return df.loc[ind_equal, index].iloc[0], False

    trial_id, new_id = lookup_or_add_params(
        id_table, TRIAL_TYPES, prob_params, "id", "prob_params"
    )
    variant, new_var = lookup_or_add_params(
        var_table, VARIANT_TYPES, sim_params, "variant", "sim_params"
    )
    if new_var or new_id:
        iteration = 1
    else:
        stmt = select(trials_table).where(
            (trials_table.c.id == int(trial_id))
            & (trials_table.c.variant == int(variant))
        )
        df = pd.read_sql(stmt, eng)
        iteration = df["iteration"].max() + 1

    return trial_id, variant, iteration


def run(
    ex: Experiment,
    debug=False,
    *,
    logfile="trials.db",
    prob_params=None,
    sim_params=None,
    trials_folder=Path(__file__).absolute().parent / "trials",
):
    if not debug and REPO.is_dirty():
        raise RuntimeError(
            "Git Repo is dirty.  For repeatable tests,"
            " clean the repo by committing or stashing all changes and "
            "untracked files."
        )
    trial_db = Path(trials_folder).absolute() / logfile
    exp_logger, trials_table = _init_logger(trial_db, "trials")
    id_table, var_table = _init_id_variant_tables(trial_db)
    trial, variant, iteration = _id_variant_iteration(
        trial_db,
        trials_table,
        sim_params=sim_params,
        var_table=var_table,
        prob_params=prob_params,
        id_table=id_table,
    )

    utc_now = datetime.now(timezone.utc)
    cpu_now = process_time()
    commit = REPO.head.commit.hexsha
    if isinstance(ex, type):
        log_msg = (
            f"Running experiment {ex.__name__}, trial {trial}, simulation type"
            f" {variant} at time: "
            + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
            + f".  Current repo hash: {commit}"
        )
    else:
        log_msg = (
            f"Running experiment {ex.name}, trial {trial}, simulation type"
            f" {variant} at time: "
            + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
            + f".  Current repo hash: {commit}"
        )
    if debug:
        log_msg += ".  In debugging mode."
    exp_logger.info(log_msg)

    if isinstance(ex, type):
        results = ex(**sim_params, **prob_params).run()
    else:
        warnings.warn(
            "Passing an experiment object is deprecated.  Pass an experiment"
            " class, with sim_params, and prob_params separately"
        )
        results = ex.run()

    utc_now = datetime.now(timezone.utc)
    exp_logger.info(
        "Finished experiment at time: "
        + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
        + f".  Results: {results['metrics']}"
    )
    cpu_time = process_time() - cpu_now

    # Save save results in trial database
    if not debug:
        new_filename = f"trial{trial}_{variant}_{iteration}.ipynb"
        exp_logger.info(
            "trial entry"
            + f"--{trial}"
            + f"--{variant}"
            + f"--{iteration}"
            + f"--{commit}"
            + f"--{cpu_time}"
            + f'--{results["metrics"]}'
            + f"--{new_filename}--"
        )
