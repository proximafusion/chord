"""Tests for the identifier scan, and the scan of the package itself."""

import pathlib

import pytest
from tools import identifiers

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Built rather than written out, so this file does not trip the scan it exercises.
STORAGE_URI = "gs" + "://example-artifacts/object"
MONOREPO_IMPORT = "from " + "experimental" + ".someone.thing import helper"
AUTHOR_LINE = "Written by Santiago " + "Ca" + "dena."


def test_the_package_is_ready_to_leave_the_monorepo() -> None:
    """Nothing here may reference infrastructure an outside reader cannot reach."""
    findings = identifiers.scan(ROOT, identifiers.PUBLIC)

    assert not findings, "\n".join(f.describe(ROOT) for f in findings)


@pytest.mark.parametrize("suffix", [".py", ".json", ".ipynb", ".sh"])
def test_a_cloud_storage_uri_is_caught(tmp_path, suffix: str) -> None:
    (tmp_path / f"artifact{suffix}").write_text(f'BASE = "{STORAGE_URI}"\n')

    findings = identifiers.scan(tmp_path, identifiers.PUBLIC)

    assert [f.pattern.name for f in findings] == ["cloud_storage_uri"]


@pytest.mark.parametrize("host_prefix", ["", "example-artifacts."])
def test_a_cloud_storage_url_is_caught(tmp_path, host_prefix: str) -> None:
    location = (
        "https://"
        + host_prefix
        + "storage."
        + "googleapis.com/example-artifacts/object"
    )
    (tmp_path / "manifest.json").write_text(location)
    findings = identifiers.scan(tmp_path, identifiers.PUBLIC)
    assert [f.pattern.name for f in findings] == ["cloud_storage_url"]


def test_the_scanner_source_is_also_checked(tmp_path) -> None:
    (tmp_path / "identifiers.py").write_text(STORAGE_URI)
    assert [f.pattern.name for f in identifiers.scan(tmp_path)] == ["cloud_storage_uri"]


def test_a_monorepo_import_is_caught(tmp_path) -> None:
    (tmp_path / "x.py").write_text(MONOREPO_IMPORT + "\n")

    findings = identifiers.scan(tmp_path, identifiers.PUBLIC)

    assert [f.pattern.name for f in findings] == ["monorepo_path"]


def test_author_names_pass_the_public_level_but_not_the_anonymous_one(
    tmp_path,
) -> None:
    """The public repository credits its authors; an archive for review cannot."""
    (tmp_path / "README.md").write_text(AUTHOR_LINE + "\n")

    assert not identifiers.scan(tmp_path, identifiers.PUBLIC)
    assert [
        f.pattern.name for f in identifiers.scan(tmp_path, identifiers.ANONYMOUS)
    ] == ["author_name"]


def test_an_email_is_caught_at_the_anonymous_level(tmp_path) -> None:
    (tmp_path / "README.md").write_text("Contact someone@" + "lab.org for details.\n")

    findings = identifiers.scan(tmp_path, identifiers.ANONYMOUS)

    assert [f.pattern.name for f in findings] == ["email"]


def test_the_anonymous_level_includes_the_public_one(tmp_path) -> None:
    """An archive for review has to clear both bars, not trade one for the other."""
    (tmp_path / "x.py").write_text(f'BASE = "{STORAGE_URI}"\n')

    names = {f.pattern.name for f in identifiers.scan(tmp_path, identifiers.ANONYMOUS)}

    assert "cloud_storage_uri" in names


def test_clean_source_produces_nothing(tmp_path) -> None:
    (tmp_path / "x.py").write_text("VALUE = 1\n")

    assert not identifiers.scan(tmp_path, identifiers.ANONYMOUS)


def test_compiled_and_vendored_directories_are_skipped(tmp_path) -> None:
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "x.py").write_text(f'BASE = "{STORAGE_URI}"\n')

    assert not identifiers.scan(tmp_path, identifiers.PUBLIC)


def test_findings_point_at_a_line() -> None:
    pattern = identifiers.PATTERNS[0]
    finding = identifiers.Finding(ROOT / "pyproject.toml", 7, pattern, "text")

    assert finding.describe(ROOT).startswith("pyproject.toml:7:")


@pytest.mark.parametrize("pattern", identifiers.PATTERNS, ids=lambda p: p.name)
def test_every_pattern_explains_itself(pattern: identifiers.Pattern) -> None:
    """A finding is only actionable if it says why the string may not be there."""
    assert pattern.reason
    assert pattern.level in (identifiers.PUBLIC, identifiers.ANONYMOUS)


def test_a_licence_is_scanned_despite_having_no_suffix(tmp_path) -> None:
    """A licence names its copyright holder, which is the organisation."""
    (tmp_path / "LICENSE").write_text("Copyright (c) 2026 " + "Proxima" + " Fusion\n")

    findings = identifiers.scan(tmp_path, identifiers.ANONYMOUS)

    assert [f.pattern.name for f in findings] == ["organisation"]


def test_the_documentation_domain_is_not_an_author(tmp_path) -> None:
    """example.com is reserved for documentation, so it identifies nobody."""
    (tmp_path / "README.md").write_text("Contact someone@example.com for details.\n")

    assert not identifiers.scan(tmp_path, identifiers.ANONYMOUS)
