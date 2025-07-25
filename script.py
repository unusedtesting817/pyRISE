# Let's start by creating the PyRISE package structure and implementing the core components
# First, create the directory structure

import os
import sys
from pathlib import Path

# Create the main PyRISE package structure
project_structure = {
    "src/pyrise": [
        "__init__.py",
        "core/",
        "solvers/",
        "regimes/",
        "estimation/",
        "simulation/",
        "plotting/",
        "utils/"
    ],
    "tests": [
        "__init__.py",
        "test_core/",
        "test_solvers/",
        "test_regimes/",
        "test_estimation/",
        "test_simulation/",
        "integration/"
    ],
    "docs": [
        "index.md",
        "api/",
        "tutorials/",
        "examples/"
    ],
    ".github/workflows": [],
    "examples": []
}

def create_directory_structure():
    """Create the PyRISE directory structure"""
    for base_dir, subdirs in project_structure.items():
        # Create base directory
        Path(base_dir).mkdir(parents=True, exist_ok=True)

        # Create subdirectories and files
        for item in subdirs:
            if item.endswith('/'):
                # It's a directory
                Path(base_dir, item.rstrip('/')).mkdir(exist_ok=True)
            else:
                # It's a file
                Path(base_dir, item).touch()

create_directory_structure()

# Display the created structure
def show_tree(path, prefix="", is_last=True):
    """Display directory tree structure"""
    if path.name.startswith('.'):
        return

    print(f"{prefix}{'└── ' if is_last else '├── '}{path.name}")

    if path.is_dir():
        children = [p for p in path.iterdir() if not p.name.startswith('.')]
        children.sort(key=lambda x: (x.is_file(), x.name))

        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            show_tree(child, new_prefix, is_last_child)

print("PyRISE Package Structure Created:")
print("pyrise/")
show_tree(Path("."), is_last=True)