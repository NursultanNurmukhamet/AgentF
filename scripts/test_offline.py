"""Run the archived offline suites without contacting AWS."""
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[1]
tests = ["test_v6_direct_attack.py", "test_fast_path.py", "test_scenarios.py", "test_agent_base_nova.py", "test_parsing.py"]
for name in tests:
    subprocess.run([sys.executable, "-B", str(root / "team" / "lib" / name)], check=True, cwd=root)
print("All five offline suites passed.")
