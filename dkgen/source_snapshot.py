"""Keep the working source files with each run, including uncommitted edits."""

from pathlib import Path
import shutil


def save_source_snapshot(output_directory):
    """Copy this installation's source into a new run's source/ directory."""
    code_root = Path(__file__).resolve().parents[1]
    output = Path(output_directory).resolve()
    patterns = ("dkgen/**/*.py", "tests/**/*.py", "scripts/**/*",
                "Makefile", "README.md", "requirements.txt")
    excluded = {"__pycache__", ".pytest_cache", ".git", "results"}
    # Collect before creating the snapshot, and exclude the output even if a
    # custom --output directory is nested inside one of the included scopes.
    sources = sorted({path for pattern in patterns for path in code_root.glob(pattern)
                      if path.is_file() and not path.is_symlink()
                      and path.suffix not in {".pyc", ".pyo"}
                      and not excluded.intersection(path.relative_to(code_root).parts)
                      and output not in path.parents},
                     key=lambda path: path.relative_to(code_root).as_posix())
    destination = output / "source"
    destination.mkdir(exist_ok=False)
    files = []
    for source in sources:
        relative = source.relative_to(code_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files.append(relative.as_posix())
    return {"directory": "source", "files": files}
