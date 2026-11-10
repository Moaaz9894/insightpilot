import os

# Keep the test suite offline even when a real .env exists
for _k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT",
           "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY"):
    os.environ[_k] = ""
import tempfile

# Isolated store for the whole test session (set BEFORE importing the package).
os.environ["INSIGHTPILOT_HOME"] = tempfile.mkdtemp(prefix="ip_tests_")
os.environ.pop("API_KEY", None)

import pytest  # noqa: E402
from insightpilot.bootstrap import ensure  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def warehouse():
    ensure()
