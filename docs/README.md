# pyQuTree Documentation

This directory contains the Sphinx documentation for pyQuTree.

## Building the Documentation Locally

### Install Dependencies

```bash
# Using poetry (recommended)
poetry install --with docs

# Or using pip
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Other Formats

```bash
make latexpdf  # Build PDF
make epub      # Build EPUB
make help      # See all available formats
```

## Documentation Structure

- `index.rst` - Main documentation index
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `usage/` - Detailed usage guides
  - `ttnopt.rst` - TTNOpt optimization guide
  - `tree_structures.rst` - Tree network topologies
- `api/` - API reference documentation

## ReadTheDocs

The documentation is automatically built and hosted on ReadTheDocs at:
https://pyqutree.readthedocs.io

The configuration is in `.readthedocs.yml` in the project root.

## Contributing to Documentation

When adding new features, please update the relevant documentation:

1. Add docstrings to your Python code (Google or NumPy style)
2. Update or create .rst files in the appropriate section
3. Test the build locally before submitting a PR
4. Ensure there are no Sphinx warnings or errors
