from pathlib import Path
from sqlalchemy import (
    create_engine,
    inspection,
    insert,
    Table,
    Column,
    Integer,
    Float,
    String,
    ForeignKey,
    MetaData,
)
import pandas as pd

trial_types = [
    Column("id", Integer, primary_key=True),
    Column("short_name", String),
    Column("factors", String),
]

variant_types = [
    Column("variant", Integer, primary_key=True),
    Column("short_name", String),
]

trial_df = pd.DataFrame(
    [
        [1, "base", "[]"],
        [2, "no t_scale", '["no t_scale"]'],
        [3, "no conditioner", '["no cond"]'],
        [4, "super basic", '["no t_scale", "no cond"]'],
        [5, "Higher Current order", '["current_order"]'],
        [6, "Higher Vehicle order", '["vehicle_order"]'],
        [7, "ttw", '["ttw"]'],
        [8, "ttw-curr", '["ttw", "current order"]'],
        [9, "ttw-veh", '["ttw", "vehicle_order"]'],
        [10, "ttw-order", '["ttw", "current_order", "vehicle_order"]'],
        [11, "Higher order", '["current_order", "vehicle_order"]'],
        [
            12,
            "Better order",
            '["no t_scale", "current_order", "vehicle_order"]',
        ],
        [13, "Better current", '["no t_scale", "current_order"]'],
        [14, "Better vehicle", '["no t_scale", "vehicle_order"]'],
        [
            15,
            "Super basic order",
            '["no cond", "no t_scale", "current_order", "vehicle_order"]',
        ],
        [
            16,
            "Better ttw order",
            '["ttw", "no t_scale", "current_order", "vehicle_order"]',
        ],
        [17, "better ttw", '["no t_scale", "ttw"]'],
    ],
    columns=["id", "short_name", "factors"],
)

variant_df = pd.DataFrame(
    [
        [1, "endpoints gps"],
        [1, "first gps"],
        [1, "multi-first gps"],
        [1, "multi-run endpoints gps"],
        [1, "multi-run multi-first gps"],
    ],
    columns=["variant", "short_name"],
)

print(Path(__file__).absolute().parent)
