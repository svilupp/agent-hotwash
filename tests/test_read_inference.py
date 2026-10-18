"""Observable shell read evidence stays conservative about shell syntax."""

import pytest

from agent_hotwash.primitives.reads import read_paths


@pytest.mark.parametrize(
    "command,expected",
    [
        ('echo "sample; cat fake.py; text"', []),
        ("echo 'sample | cat fake.py' && cat real.py", ["real.py"]),
        ("rg -e TODO src/a.py", ["src/a.py"]),
        ("rg --regexp=TODO src/a.py", ["src/a.py"]),
        ("grep -f patterns.txt src/a.py", ["patterns.txt", "src/a.py"]),
        ("cat < src/a.py > output.txt", ["src/a.py"]),
        ("rg TODO src/a.py 2> errors.log", ["src/a.py"]),
    ],
)
def test_shell_read_evidence(command, expected):
    assert read_paths("Bash", {"command": command}) == expected
    from agent_hotwash.sources.codex_items import _artifacts_from_compound

    artifacts = _artifacts_from_compound(command, None)
    assert sorted({artifact.path for artifact in artifacts}) == expected
