# MeerKAT OPT Schedule Block Review 

This repository contains Python code and example inputs used to review
MeerKAT Open Time schedule blocks exported from the MeerKAT Observation
Planning Tool (OPT).

The workflow is run roughly once per year for the Open Time call, and this repo
exists so that future runs are reproducible and easy to repeat or hand over.

## What the script does

`check_opt_json.py`:

- Reads a **master proposal catalogue** in CSV format (`--master-csv`).
- Parses one or more **OPT simulation JSON epoch files** (`--epoch-json`).
- Tallies time spent on each science target and calibrator across epochs.
- Checks that every scheduled science target in the JSON files is present in
  the master catalogue.
- Updates the master CSV **in place** (e.g. a `Notes` column) with information
  about which epochs each source is used in.
- Prints a human-readable summary of scan lengths and total project time.

## Dependencies

- Python 3.x
- `numpy`
- `pandas`

Install them with:

```bash
pip install -r requirements.txt

