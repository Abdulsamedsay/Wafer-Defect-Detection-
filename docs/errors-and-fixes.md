# Errors and Fixes

> Running log of bugs, errors, and fixes encountered during the project.

---

## Error: pandas.indexes ModuleNotFoundError (Legacy Pickle Compatibility)

### Context
Loading the WM-811K dataset (`LSWMD.pkl`) with `pd.read_pickle()` in pandas 3.x.

### Error message
```
ModuleNotFoundError: No module named 'pandas.indexes'
```
Followed by:
```
UnicodeDecodeError: 'ascii' codec can't decode byte 0x9a in position 6
```

### Root cause
The WM-811K pickle file was created with an old version of pandas (pre-1.0) under Python 2. Two issues compound:

1. **Namespace rename:** pandas 0.x used `pandas.indexes.*` as its index module namespace. In pandas 1.x it was moved to `pandas.core.indexes.*`. In pandas 2.x+ the old namespace was completely removed.
2. **Python 2 string encoding:** Python 2 pickles encode strings as bytes using ASCII by default. Python 3's `pickle.load()` needs `encoding='latin1'` to decode them correctly.

### Fix
Bypass `pd.read_pickle()` entirely and load with the standard `pickle` module using `encoding='latin1'`:

```python
import pickle

with open("data/raw/LSWMD.pkl", "rb") as f:
    df = pickle.load(f, encoding="latin1")
```

Additionally, patch `sys.modules` before loading to satisfy any `__import__` calls inside the pickle stream:

```python
import sys, types, importlib

REMAP = {
    "pandas.indexes": "pandas.core.indexes",
    "pandas.indexes.base": "pandas.core.indexes.base",
    # ... (full list in eda_runner.py / src/data_loader.py)
}
for old, new in REMAP.items():
    if old not in sys.modules:
        try:
            sys.modules[old] = importlib.import_module(new)
        except ImportError:
            sys.modules[old] = types.ModuleType(old)

with open("data/raw/LSWMD.pkl", "rb") as f:
    df = pickle.load(f, encoding="latin1")
```

### Lesson learned
- WM-811K cannot be loaded with modern pandas directly. Always use `pickle.load(..., encoding='latin1')`.
- The `sys.modules` patch is required before loading, not after.
- This fix is captured in `src/data_loader.py` so it doesn't need to be rediscovered.
- Log the fix here so it is never forgotten across sessions.

---

## Error: Python was not found (Windows Store stub)

### Context
Running `python` in the Claude Code terminal on a fresh Windows installation.

### Error message
```
Python was not found; run without arguments to install from the Microsoft Store,
or disable this shortcut from Settings > Apps > Advanced app settings > App execution aliases.
```

### Root cause
Windows 11 ships with a `python.exe` stub in `WindowsApps` that redirects to the Microsoft Store instead of running Python. It appears in PATH before any real Python installation.

### Fix
Install Python via Miniforge3. The miniforge Python is at:
```
C:\Users\samed\miniforge3\python.exe
```
To make it the default in Claude Code terminal, use the full path or activate the conda environment before running scripts.

### Lesson learned
Always use the full miniforge path `C:\Users\samed\miniforge3\python.exe` when running scripts from the Claude Code terminal, since the system `python` resolves to the Windows Store stub.
