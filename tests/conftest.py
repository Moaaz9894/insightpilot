import os
import tempfile

# Isolated store for the whole test session (set BEFORE importing the package).
os.environ["INSIGHTPILOT_HOME"] = tempfile.mkdtemp(prefix="ip_tests_")
for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_SEARCH_ENDPOINT", "API_KEY"):
    os.environ.pop(k, None)

import pytest  # noqa: E402
from insightpilot.bootstrap import ensure  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def warehouse():
    ensure()
