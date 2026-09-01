# Sketch to G-Code

Desktop Tkinter app that converts sketch or line-art images into optimized
G-code for a drawing robot.

## Setup

```powershell
uv venv --python 3.11
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For development and tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

## Run

```powershell
.\.venv\Scripts\python.exe sketch_to_gcode.py
```

The project now uses a standard `src/` package layout. Root-level
`sketch_to_gcode.py`, `hatching.py`, and `reporting/` are compatibility shims
for older imports and commands.

