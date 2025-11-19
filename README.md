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
```

## `check_opt_json.py` doc_string:
Audit Open Time proposal schedule blocks using JSON epoch files exported from the OPT.
The user must first 'simulate' each schedule block in the OPT and save the resulting JSON
files locally. These files contain the detailed observation sequence and timing information
needed for this analysis.

The script reads a master proposal target list in CSV format, then for each JSON epoch file:
it parses the simulation log embedded in each JSON, tallies target/calibrator scan
lengths across all epochs, and checks that every science target scheduled in the JSON
is listed in the master proposal catalogue (CSV). The catalogue file is updated in-place
with Notes describing which epoch(s) use each source.

Example:
```
    python check_opt_json.py \\
        --master-csv 2025/observations/testing/targets-SCI-20241101-SB-01.csv \\
        --epoch-json 2025/observations/testing/SCI-20241101-SB-01_*.json \\
        --show-target-scans
```
