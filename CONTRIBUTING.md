# Contributing

## Before you commit

```bash
ruff format
ruff check
python -m pytest
```

`make format`, `make lint` and `make test` do the same if you have GNU make. While you work, `python -m pytest -m "not slow"` runs only the unit tests and takes a few seconds.

## Code style

- Docstrings follow the [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html). Put array shapes and units in the `Parameters` and `Returns` sections, for example `eeg : ndarray, shape (n_trials, n_channels, n_samples)`.
- Every module, class and public function has a docstring. Private helpers (names starting with `_`) only need one when what they do isn't obvious.
- Keep comments for what the code can't say by itself, usually why something is done in a surprising way. Explanations of what a function does belong in its docstring.
- Ruff checks formatting, import order and docstrings, so `ruff check` catches most style problems.

## When a change alters the results

`tests/test_golden.py` runs every baseline on small random data and compares the result tables with the ones saved in `tests/golden/`. If it fails, your change altered the results.

If that is a mistake, fix the change. If you meant to change the results, regenerate the tables and say why in the commit message:

```bash
python tests/make_golden.py
```

Keep such changes in their own commit, so the history shows which commit changed the numbers.

## Settings and data

New settings go into `Config` in `mtp/config.py`, with a default that keeps the current behavior and an entry in its `Attributes` section. They then become command-line options automatically.

Never commit data. `data/` is ignored by git, and the recordings are human-subject data.
