from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def part_xml() -> bytes:
    return (FIXTURES / "part-9999.xml").read_bytes()
