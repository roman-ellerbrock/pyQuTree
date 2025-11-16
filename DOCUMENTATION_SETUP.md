# Documentation Setup Guide

This document explains how the documentation for pyQuTree has been set up with Sphinx and ReadTheDocs.

## What Was Created

### Documentation Files

1. **docs/** - Main documentation directory
   - `conf.py` - Sphinx configuration
   - `index.rst` - Main documentation index
   - `installation.rst` - Installation instructions
   - `quickstart.rst` - Quick start guide
   - `Makefile` - Build automation
   - `requirements.txt` - Documentation dependencies
   - `README.md` - Documentation build instructions

2. **docs/usage/** - Detailed usage guides
   - `ttnopt.rst` - Comprehensive guide to TTNOpt optimization
   - `tree_structures.rst` - Complete guide to tree tensor network topologies

3. **docs/api/** - API reference
   - `index.rst` - Auto-generated API documentation

4. **.readthedocs.yml** - ReadTheDocs configuration file

### Key Features

#### TTNOpt Usage Guide ([docs/usage/ttnopt.rst](docs/usage/ttnopt.rst))

Comprehensive documentation covering:
- Basic usage and parameters
- Complete code examples
- Using start points for warm starting
- Progress tracking and logging
- Understanding sweeps
- Performance considerations (bond dimension, grid resolution, dimensionality)
- Caching behavior
- Common issues and troubleshooting

#### Tree Structures Guide ([docs/usage/tree_structures.rst](docs/usage/tree_structures.rst))

Detailed comparison and usage of different network topologies:
- **Tensor Train (TT)** networks
  - Linear chain topology
  - O(f × N × r²) complexity
  - Best for f < 20 dimensions
- **Balanced Tree (BT)** networks
  - Hierarchical binary tree
  - O(f × N × r + r³ × log₂(f)) complexity
  - Best for f > 20 dimensions
- Complete examples for both structures
- Performance benchmarks and comparisons
- Best practices and workflow recommendations

## Building Documentation Locally

### Install Dependencies

```bash
# Using poetry (recommended)
poetry install --with docs

# Or using pip
cd docs
pip install -r requirements.txt
```

### Build HTML

```bash
cd docs
make html
```

The documentation will be in `docs/_build/html/index.html`.

### Build Other Formats

```bash
make latexpdf  # PDF
make epub      # EPUB
make help      # See all options
```

## ReadTheDocs Setup

### Automatic Deployment

The documentation is configured to build automatically on ReadTheDocs when you:

1. Push changes to GitHub
2. Create a new tag/release

### Configuration

- **File**: `.readthedocs.yml`
- **Python version**: 3.11
- **Build method**: Poetry with docs group
- **Output formats**: HTML, PDF, EPUB

### Setup Steps for ReadTheDocs

1. Go to https://readthedocs.org/
2. Sign in with your GitHub account
3. Import the `pyqutree` repository
4. The build will start automatically
5. Documentation is now live at: **https://pyqutree-ttn.readthedocs.io**

### Badge for README

The following badge has been added to README.md:

```markdown
[![Documentation Status](https://readthedocs.org/projects/pyqutree-ttn/badge/?version=latest)](https://pyqutree-ttn.readthedocs.io/en/latest/?badge=latest)
```

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main index
├── installation.rst       # Installation guide
├── quickstart.rst        # Quick start tutorial
├── usage/
│   ├── ttnopt.rst        # TTNOpt optimization guide
│   └── tree_structures.rst # Tree network topologies
├── api/
│   └── index.rst         # API reference
├── Makefile              # Build automation
├── requirements.txt      # Doc dependencies
└── README.md            # Build instructions
```

## Documentation is Live! ✓

The documentation is now hosted at: **https://pyqutree-ttn.readthedocs.io**

Key pages:
- [Home](https://pyqutree-ttn.readthedocs.io)
- [Quick Start](https://pyqutree-ttn.readthedocs.io/en/latest/quickstart.html)
- [TTNOpt Guide](https://pyqutree-ttn.readthedocs.io/en/latest/usage/ttnopt.html)
- [Tree Structures](https://pyqutree-ttn.readthedocs.io/en/latest/usage/tree_structures.html)
- [API Reference](https://pyqutree-ttn.readthedocs.io/en/latest/api/index.html)

### Updating Documentation

When you make changes:

1. Edit the relevant `.rst` files in `docs/`
2. Test locally: `cd docs && make html`
3. Commit and push
4. ReadTheDocs will automatically rebuild

### Adding New Pages

1. Create new `.rst` file in appropriate directory
2. Add it to a `toctree` directive in `index.rst` or parent page
3. Test build locally
4. Commit and push

## Documentation Features

- **Auto-generated API docs** from docstrings
- **Syntax highlighting** for code examples
- **Cross-references** between pages
- **Search functionality**
- **Multiple output formats** (HTML, PDF, EPUB)
- **Responsive design** with RTD theme
- **Version support** on ReadTheDocs

## Customization

### Theme

The documentation uses the Sphinx RTD theme. To customize:

Edit `docs/conf.py`:
```python
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    # ... other options
}
```

### Adding Extensions

Add to `extensions` list in `docs/conf.py`:
```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    # Add more extensions here
]
```

## Maintenance

### Dependencies

Documentation dependencies are in:
- `pyproject.toml` - Poetry docs group
- `docs/requirements.txt` - ReadTheDocs

Keep both synchronized.

### Version Updates

Update version in:
1. `pyproject.toml`
2. `docs/conf.py`
3. `qutree/__init__.py`

## Support

For issues with:
- **Documentation content**: Update the `.rst` files
- **Build failures**: Check ReadTheDocs build logs
- **Local builds**: Ensure dependencies are installed

## Examples

The documentation includes extensive examples:

- Basic optimization with tensor trains
- High-dimensional problems with balanced trees
- Using start points
- Performance benchmarking
- Progress tracking
- Visualization

All examples are runnable and tested.
