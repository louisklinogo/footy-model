# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedFunctionDecorator=false, reportUnknownParameterType=false, reportCallInDefaultInitializer=false, reportExplicitAny=false
from __future__ import annotations

import sys
from collections.abc import Generator
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from psycopg2.extensions import connection
from psycopg2.extras import RealDictCursor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


MODEL_NAME = "premium_gbm"
MODEL_VERSION = "v3"
MARKETS = ("o15", "o25", "c85")

WEBAPP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEBAPP_DIR / "templates"
STATIC_DIR = WEBAPP_DIR / "static"

app = FastAPI(title="Predictions Browser", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def get_db_conn() -> Generator[connection, None, None]:
    conn = connect_db()
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(conn: connection, query: str, params: tuple[object, ...]) -> list[dict[str, object]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def fetch_one(conn: connection, query: str, params: tuple[object, ...]) -> dict[str, object] | None:
    rows = fetch_all(conn, query, params)
    return rows[0] if rows else None


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/predictions")


@app.get("/predictions")
def predictions_page(
    request: Request,
    date_value: date | None = Query(default=None, alias="date"),
    league: str | None = Query(default=None),
    conn: connection = Depends(get_db_conn),
):
    target_date = date_value or utc_today()
    query = """
    SELECT
        f.fixture_id,
        f.flashscore_id,
        f.match_datetime_utc,
        f.league_code,
        th.team_name AS home_team,
        ta.team_name AS away_team,
        MAX(CASE WHEN p.market_code = 'o15' THEN p.p_model END) AS p_o15,
        MAX(CASE WHEN p.market_code = 'o25' THEN p.p_model END) AS p_o25,
        MAX(CASE WHEN p.market_code = 'c85' THEN p.p_model END) AS p_c85
    FROM fixtures f
    JOIN teams th ON th.team_id = f.home_team_id
    JOIN teams ta ON ta.team_id = f.away_team_id
    LEFT JOIN predictions p
        ON p.fixture_id = f.fixture_id
       AND p.model_name = %s
       AND p.model_version = %s
       AND p.market_code IN ('o15', 'o25', 'c85')
    WHERE f.status = 'scheduled'
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc::date = %s
    """
    params: list[object] = [MODEL_NAME, MODEL_VERSION, target_date]

    if league:
        query += " AND f.league_code = %s"
        params.append(league)

    query += """
    GROUP BY
        f.fixture_id,
        f.flashscore_id,
        f.match_datetime_utc,
        f.league_code,
        th.team_name,
        ta.team_name
    ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC
    """

    fixtures = fetch_all(conn, query, tuple(params))
    return templates.TemplateResponse(
        request,
        "predictions.html",
        {
            "selected_date": target_date.isoformat(),
            "selected_league": league or "",
            "fixtures": fixtures,
        },
    )


@app.get("/fixture/{flashscore_id}")
def fixture_page(
    request: Request,
    flashscore_id: str,
    conn: connection = Depends(get_db_conn),
):
    fixture = fetch_one(
        conn,
        """
        SELECT
            f.fixture_id,
            f.flashscore_id,
            f.match_datetime_utc,
            f.league_code,
            f.status,
            th.team_name AS home_team,
            ta.team_name AS away_team
        FROM fixtures f
        JOIN teams th ON th.team_id = f.home_team_id
        JOIN teams ta ON ta.team_id = f.away_team_id
        WHERE f.flashscore_id = %s
        """,
        (flashscore_id,),
    )

    if fixture is None:
        raise HTTPException(status_code=404, detail="Fixture not found")

    prediction_rows = fetch_all(
        conn,
        """
        SELECT
            p.market_code,
            p.p_model,
            p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = %s
          AND p.model_name = %s
          AND p.model_version = %s
          AND p.market_code IN ('o15', 'o25', 'c85')
        ORDER BY p.market_code ASC
        """,
        (fixture["fixture_id"], MODEL_NAME, MODEL_VERSION),
    )

    by_market = {row["market_code"]: row for row in prediction_rows}
    market_rows: list[dict[str, object]] = [
        {
            "market": market,
            "p_model": by_market.get(market, {}).get("p_model"),
            "metadata": by_market.get(market, {}).get("metadata_json"),
        }
        for market in MARKETS
    ]

    return templates.TemplateResponse(
        request,
        "fixture.html",
        {
            "fixture": fixture,
            "market_rows": market_rows,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
        },
    )
