"""Prepare fresh local runtime source folders; never calls AWS."""
import pathlib
import shutil

root = pathlib.Path(__file__).resolve().parents[1]
out = root / "build"
if out.exists():
    raise SystemExit("build already exists; choose a clean checkout to preserve prior output")
for role in ("ai-gk", "ai-def", "ai-mid", "ai-fwd1", "ai-fwd2"):
    target = out / role
    shutil.copytree(root / "team" / role, target)
    shutil.copytree(root / "team" / "lib", target / "lib", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
print("Prepared five runtime source folders in build; no cloud deployment performed.")
