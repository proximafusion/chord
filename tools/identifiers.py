"""Find strings that must not survive a move out of the monorepo.

The package travels through three homes, and each one tolerates less than the last. In
the monorepo it may reference internal paths. In the public repository it may name its
authors but not internal infrastructure, which outside readers cannot reach anyway. In
an anonymized archive for review it may name neither.

Keeping that as a checklist invites missing an entry, so it is a scan with two levels.
The public level runs as a test, so a reference to an internal bucket fails before it
can ship. The anonymous level runs as a report, because what it finds is a list of edits
to make at extraction time rather than a defect.

Run the report with::

    python -m tools.identifiers
"""

from __future__ import annotations

import dataclasses
import pathlib
import re

PUBLIC = "public"
ANONYMOUS = "anonymous"


@dataclasses.dataclass(frozen=True)
class Pattern:
    """One thing to look for, and why it may not be where it is."""

    name: str
    regex: str
    reason: str
    level: str


# Split so this file does not match its own patterns when it scans itself.
_ORG = "(?i)" + "proxima" + r"[\s_-]?" + "fusion"
"""Matches the run-together, hyphenated and spaced spellings, in any case.

A licence writes it with a space and a repository slug with a hyphen, so matching only
one spelling would pass a file that names the organisation plainly.
"""

_BUCKET = "proxima" + "-ml-prototyping"
_AUTHORS = ("sca" + "dena", "a" + "merlo", "Ca" + "dena", "Mer" + "lo")

PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "internal_bucket",
        re.escape(_BUCKET),
        "an internal bucket no outside reader can reach",
        PUBLIC,
    ),
    Pattern(
        "monorepo_path",
        r"\b(?:experimental|beta)[./][a-z_]+[./]",
        "a monorepo path that will not resolve once the package moves",
        PUBLIC,
    ),
    Pattern(
        "experiment_tracker",
        r"wandb\.ai/[\w-]+",
        "an internal experiment-tracking project",
        PUBLIC,
    ),
    Pattern(
        "author_name",
        "(?i)" + "|".join(_AUTHORS),
        "an author's name or username",
        ANONYMOUS,
    ),
    Pattern("organisation", _ORG, "the authors' organisation", ANONYMOUS),
    Pattern(
        "email",
        # example.com is reserved for documentation, so it identifies nobody.
        r"[\w.+-]+@(?!example\.)[\w-]+\.[\w.]+",
        "an author's email address",
        ANONYMOUS,
    ),
)

SCANNED_SUFFIXES = (".py", ".toml", ".md", ".cfg", ".yaml", ".yml")
SCANNED_NAMES = ("LICENSE", "NOTICE", "CITATION.cff")
"""Files worth scanning that carry no suffix.

A licence names its copyright holder.
"""

SKIPPED_DIRECTORIES = frozenset({"__pycache__", ".git", "dist", "build"})


@dataclasses.dataclass(frozen=True)
class Finding:
    path: pathlib.Path
    line: int
    pattern: Pattern
    text: str

    def describe(self, root: pathlib.Path) -> str:
        return (
            f"{self.path.relative_to(root)}:{self.line}: {self.pattern.name} "
            f"({self.pattern.reason}) -> {self.text.strip()[:88]}"
        )


def scan(root: pathlib.Path, level: str = PUBLIC) -> list[Finding]:
    """Findings at the given level, and at every level stricter than it.

    The anonymous level includes the public one, because an archive for review has to
    clear both bars rather than trade one for the other.
    """
    levels = {PUBLIC} if level == PUBLIC else {PUBLIC, ANONYMOUS}
    patterns = [pattern for pattern in PATTERNS if pattern.level in levels]
    findings: list[Finding] = []

    for path in sorted(root.rglob("*")):
        interesting = path.suffix in SCANNED_SUFFIXES or path.name in SCANNED_NAMES
        if not interesting or not path.is_file():
            continue
        if SKIPPED_DIRECTORIES & set(path.parts) or path.name == "identifiers.py":
            continue
        for number, text in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            for pattern in patterns:
                if re.search(pattern.regex, text):
                    findings.append(Finding(path, number, pattern, text))
    return findings


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    for level in (PUBLIC, ANONYMOUS):
        findings = scan(root, level)
        print(f"\n=== {level}: {len(findings)} finding(s)")
        for finding in findings:
            print(f"  {finding.describe(root)}")
    print(
        "\nPublic findings block a release. Anonymous findings are the edits an "
        "archive for review still needs."
    )


if __name__ == "__main__":
    main()
