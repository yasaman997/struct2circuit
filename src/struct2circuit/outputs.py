"""Small preflight check for deterministic experiment output destinations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def preflight_outputs(paths: Iterable[Path]) -> None:
    """Refuse all existing destinations before simulation or directory creation.

    Unrelated files in an output directory are allowed. A directory or dangling
    symlink at a destination also counts as a collision. No Git metadata or
    automatic output-path selection is involved. Parent traversal is rejected:
    creating a missing directory before ``..`` can expose an existing artifact.
    """
    paths = tuple(paths)
    traversal = [path for path in paths if ".." in path.parts]
    if traversal:
        raise FileExistsError(
            "Refusing output paths containing '..' components: "
            + ", ".join(str(path) for path in traversal)
            + ". Choose an output directory or filename without parent traversal "
            "using --output."
        )
    collisions = [path for path in paths if path.exists() or path.is_symlink()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing output artifacts: "
            + ", ".join(str(path) for path in collisions)
            + ". Choose a fresh output directory or filename with --output."
        )
