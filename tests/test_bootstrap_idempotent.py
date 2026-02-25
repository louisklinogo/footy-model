# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnusedParameter=false

from conftest import run_script


def test_bootstrap_schema_idempotent(db_case):
    first = run_script(["src/ingest/bootstrap_fixtures_schema_v1.py"])
    assert first.returncode == 0, first.stderr or first.stdout

    second = run_script(["src/ingest/bootstrap_fixtures_schema_v1.py"])
    assert second.returncode == 0, second.stderr or second.stdout
