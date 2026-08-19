# Test organization

The test tree covers two different maintenance boundaries:

- runtime and package tests exercise the current application and clean release;
- protocol tests preserve exact historical freeze, audit, and recovery
  contracts.

Some historical preflight tests intentionally reject the current filesystem
after a one-time protocol has advanced to its final canonical state. Those
failures mean an immutable snapshot no longer describes the present lifecycle;
they are not runtime regressions and their source files must not be edited just
to make a completed one-time operation runnable again.

For current application cleanup, run at least:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_api_helpers.py `
  tests\test_automatic_runtime.py `
  tests\test_hybrid.py `
  tests\test_release_builder.py
```

Run the complete suite when changing research code, then classify any failure
against its frozen lifecycle before altering a hash-bound file.

