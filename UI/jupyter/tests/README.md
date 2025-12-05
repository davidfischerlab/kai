# Jupyter Interface Tests

This directory contains tests for the Jupyter interface module.

## Running Tests

Run all basic tests:
```bash
python UI/jupyter/tests/test_basic.py
```

## Test Coverage

- **test_basic.py**: Basic functionality tests
  - Module imports
  - Notebook loading and saving
  - ContextBuilder initialization and context generation
  - NotebookExecutor availability (without kernel startup)

## Test Files

- `test_basic.py`: Main test script
- `test_basic.ipynb`: Temporary notebook created during tests (auto-cleaned)
- `__init__.py`: Makes this directory a Python package

## Notes

- Tests do not start actual Jupyter kernels to keep them fast and isolated
- For full end-to-end testing, use the CLI: `python -m UI.jupyter --help`
- Full agent tests are located in `full_agent_test/` directory
