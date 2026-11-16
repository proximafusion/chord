"""Shared test configuration.

Every test starts from the same random state. Most tests here draw arbitrary inputs
whose values the assertions do not depend on, so seeding is not what keeps them from
flaking. It is what makes a failure reproducible from the test alone, rather than from
the particular run that produced it.

Seeding once here rather than at each call site also means a test added later is covered
without anyone remembering to do it.
"""

import pytest
import torch

SEED = 0


@pytest.fixture(autouse=True)
def seeded() -> None:
    """Reset torch's global random state before every test.

    Only torch. Every numpy draw in this package already goes through
    `np.random.default_rng` with an explicit seed, so seeding numpy's legacy global
    state here would mutate shared state that nothing reads.
    """
    torch.manual_seed(SEED)
