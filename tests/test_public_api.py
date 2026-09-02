"""The public API surface is pinned against a checked-in baseline.

`pysie2d` is a published package, so a signature change is a change to other
people's code. Every other convention in this repository is enforced by a test;
this is that test for the API itself.

**What it is for.** Additions pass freely — a new export, a new optional
keyword, a new method. Only *removals and incompatible changes* fail: a dropped
export, a renamed or reordered parameter, a parameter becoming required, a
changed default. When one is intended, `api_baseline.txt` is regenerated
deliberately (see `test_baseline_regeneration_recipe`) and the regeneration
shows up in the diff as its own reviewable hunk — which is the moment the
`BREAKING CHANGE:` footer gets written, and the point of putting it here.

**What it does not catch**, stated so nobody mistakes a green run for a promise:
behavioural breaks behind a stable signature — a flipped sign, a changed unit, a
different normalisation. Nothing mechanical catches those; `test_conventions.py`
is where they are pinned one at a time.

The v0.5 provenance: `Geometry.__init__` acquired a required `theta` and it took
a hand-run comparison against the v0.4.2 tag to notice. This file is that
comparison, made permanent.
"""

import inspect
import pathlib

import pysie2d

BASELINE = pathlib.Path(__file__).parent / "api_baseline.txt"


def _surface() -> list[str]:
    """One sorted line per public callable, carrying its full signature."""
    lines = []
    for name in sorted(pysie2d.__all__):
        obj = getattr(pysie2d, name)
        if inspect.isclass(obj):
            lines.append(f"class {name}")
            for attr in sorted(dir(obj)):
                if attr.startswith("_") and attr != "__init__":
                    continue
                member = inspect.getattr_static(obj, attr)
                if isinstance(member, property):
                    lines.append(f"{name}.{attr} -> property")
                    continue
                target = getattr(obj, attr)
                if callable(target):
                    lines.append(f"{name}.{attr}{inspect.signature(target)}")
        elif callable(obj):
            lines.append(f"{name}{inspect.signature(obj)}")
        else:
            # A module-level constant: pin its type and value, since both are
            # part of the contract (SHAPE_STEP is quoted in conventions §10).
            lines.append(f"{name}: {type(obj).__name__} = {obj!r}")
    return lines


def test_public_api_matches_the_baseline():
    current = _surface()
    recorded = BASELINE.read_text().splitlines()

    removed = [line for line in recorded if line not in current]
    added = [line for line in current if line not in recorded]

    # Removals are the failure. Reported separately from additions because the
    # two mean opposite things: a removal breaks a user, an addition does not.
    assert not removed, (
        "public API entries changed or disappeared:\n  "
        + "\n  ".join(removed)
        + "\n\nIf this is intended it is a BREAKING CHANGE: say so in the commit "
        "footer, document the migration in README.md, and regenerate "
        f"{BASELINE.name}.\n\nWhat replaced them, if anything:\n  " + "\n  ".join(added)
    )
    # Additions are allowed, but the baseline is still updated for them so the
    # file stays a true record of the surface rather than a floor under it.
    assert not added, (
        "public API gained entries (this is not a break, just regenerate "
        f"{BASELINE.name}):\n  " + "\n  ".join(added)
    )


def test_baseline_regeneration_recipe():
    """The regeneration command, kept executable so it cannot drift.

    A recipe in a comment rots. This asserts that the documented command really
    reproduces the baseline, so the instruction in the failure message above is
    always correct.
    """
    # uv run python -c "from tests.test_public_api import _surface;
    #                   print('\n'.join(_surface()))" > tests/api_baseline.txt
    assert "\n".join(_surface()) + "\n" == BASELINE.read_text()


def test_every_export_is_documented_in_the_package_docstring():
    """`__all__` and the module docstring's API list must not drift apart.

    The docstring is what a user reads first, and an export missing from it is
    invisible. Cheap to keep true, and it caught nothing only because it did
    not exist yet.
    """
    doc = pysie2d.__doc__ or ""
    missing = [name for name in pysie2d.__all__ if name not in doc]
    assert not missing, f"exported but absent from the module docstring: {missing}"
