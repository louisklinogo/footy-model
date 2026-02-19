import os

import psycopg2
from dotenv import load_dotenv

_ = load_dotenv()


def get_database_url() -> str:
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    dev_url = os.getenv("DEV_DATABASE_URL")
    if dev_url:
        return dev_url

    prod_url = os.getenv("PROD_DATABASE_URL")
    if prod_url:
        return prod_url

    raise RuntimeError(
        "Missing DATABASE_URL (or DEV_DATABASE_URL / PROD_DATABASE_URL). Set DATABASE_URL before running this script."
    )


def connect_db():
    return psycopg2.connect(get_database_url())
