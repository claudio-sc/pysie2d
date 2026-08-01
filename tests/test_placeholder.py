import importlib.metadata

import pysie2d


def test_import():
    # Compared, not pinned to a literal. The previous version of this test
    # asserted "0.2.0" outright, which let __init__.py fall three releases
    # behind pyproject.toml with CI green throughout: semantic-release bumps
    # only the files named in its config, and __init__.py was not one of them
    # until version_variables was added. Asserting the two agree makes that
    # drift fail here rather than surfacing in a user's pysie2d.__version__.
    assert pysie2d.__version__ == importlib.metadata.version("pysie2d")
