#!/usr/bin/env python3
"""
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

    python check_opt_json.py \\
        --master-csv 2025/observations/testing/targets-SCI-20241101-SB-01.csv \\
        --epoch-json 2025/observations/testing/SCI-20241101-SB-01_*.json \\
        --show-target-scans
"""

import argparse
import ast
import glob
import json
import os
import re
import sys
import traceback
from typing import Dict, List, Sequence

import numpy as np
import pandas as p


def redtext(mystr: str) -> str:
    """Format a string in red for emphasis."""
    return f"\x1b[31m{mystr}\x1b[0m"


def bred(mystr: str) -> str:
    """Format a string in bold red for high-visibility warnings."""
    return f"\x1B[1m\x1b[31m{mystr}\x1b[0m\x1B[0m"


def bblue(mystr: str) -> str:
    """Format a string in bold blue for informative messages."""
    return f"\x1B[1m\x1b[94m{mystr}\x1b[0m\x1B[0m"


def split_string(mystring: str, split_character: str) -> List[str]:
    """Split `mystring` using `split_character` while tolerating malformed data."""
    try:
        return mystring.split(str(split_character))
    except Exception:
        return []


def separation_radec(RA1, Dec1, RA2, Dec2, print_sep: bool = False):
    """
    Return angular separation between two sky positions.

    Parameters are accepted either in decimal degrees or as HH:MM:SS strings; the
    function tries both representations before falling back to NaN.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    try:
        c1 = SkyCoord(RA1 * u.deg, Dec1 * u.deg, frame="icrs")
        c2 = SkyCoord(RA2 * u.deg, Dec2 * u.deg, frame="icrs")
    except Exception:
        try:
            RA1s = split_string(str(RA1), ":")
            Dec1s = split_string(str(Dec1), ":")
            RA2s = split_string(str(RA2), ":")
            Dec2s = split_string(str(Dec2), ":")

            RA1 = f"{RA1s[0]}h{RA1s[1]}m{RA1s[2]}s"
            RA2 = f"{RA2s[0]}h{RA2s[1]}m{RA2s[2]}s"
            Dec1 = f"{Dec1s[0]}d{Dec1s[1]}m{Dec1s[2]}s"
            Dec2 = f"{Dec2s[0]}d{Dec2s[1]}m{Dec2s[2]}s"

            c1 = SkyCoord(RA1, Dec1, frame="icrs")
            c2 = SkyCoord(RA2, Dec2, frame="icrs")
        except Exception:
            traceback.print_exc()
            return (np.nan, np.nan)
    try:
        sepd = c1.separation(c2).deg
        sepa = c1.separation(c2).arcsecond
    except Exception:
        traceback.print_exc()
        sepd = np.nan
        sepa = np.nan

    if print_sep:
        print(f" - Ang sep: {sepd:.4f} deg ({sepa:.4f}\")")

    return (sepd, sepa)


def target_true(string1: str, string2: str) -> bool:
    """
    Fuzzy target-name matcher.

    Returns True if two strings share at least four consecutive characters once whitespace
    is removed and case is normalised. This mimics the relaxed matching used in the notebook.
    """
    string1 = string1.replace(" ", "").lower()
    string2 = string2.replace(" ", "").lower()

    for i in range(max(len(string1) - 3, 0)):
        if string1[i : i + 4] in string2:
            return True
    return False


def _safe_literal_list(fragment: str) -> List[str]:
    """Safely parse list literals embedded in OPT log strings."""
    try:
        value = ast.literal_eval(fragment.strip())
        if isinstance(value, (list, tuple)):
            return list(value)
    except Exception:
        pass
    return []


def _normalise_tags(raw_tags) -> List[str]:
    """Return a lower-case list of target tags regardless of the input type."""
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        tags = [raw_tags]
    else:
        tags = list(raw_tags)
    return [str(tag).strip().lower() for tag in tags if str(tag).strip()]


def _fetch_epoch_targets(blocks: Sequence[dict]) -> List[dict]:
    """Extract per-block target metadata from an epoch JSON structure."""
    targets: List[dict] = []
    for block in blocks or []:
        for entry in block.get("targets", []):
            tags = _normalise_tags(entry.get("tags"))
            targets.append(
                {
                    "name": str(entry.get("name", "")).strip(),
                    "ra": str(entry.get("ra", "")).strip(),
                    "dec": str(entry.get("dec", "")).strip(),
                    "tags": tags,
                    "duration": float(entry.get("duration") or 0.0),
                }
            )
    return targets


def _load_epoch_json(path: str) -> dict:
    """Load an epoch JSON file and extract the simulation log and target list."""
    with open(path, "r") as handle:
        data = json.load(handle)
    last_sim = data.get("last_simulation") or ""
    blocks = data.get("blocks") or []
    return {
        "file": path,
        "proposal_id": data.get("proposal_id"),
        "simulation_lines": str(last_sim).splitlines(),
        "targets": _fetch_epoch_targets(blocks),
        "instrument": data.get("instrument") or {},
        "description": data.get("description", ""),
        "obs_setup": data.get("obs_setup") or {},
        "lst_start": data.get("lst_start"),
        "lst_start_end": data.get("lst_start_end"),
        "observation_type": data.get("observation_type"),
    }


def _infer_column(columns: Sequence[str], keyword: str) -> str:
    """Heuristically find a CSV column whose name contains `keyword`."""
    keyword = keyword.lower()
    for col in columns:
        if keyword in col.strip().lower():
            return col
    return ""


def _lst_duration_hours(lst_start: str, lst_end: str) -> float:
    """Return the duration in hours between two LST strings (HH:MM[:SS])."""
    def _to_minutes(value: str):
        if not value:
            return None
        try:
            parts = str(value).split(":")
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            seconds = float(parts[2]) if len(parts) > 2 else 0.0
            return hours * 60 + minutes + seconds / 60.0
        except Exception:
            return None

    start_min = _to_minutes(lst_start)
    end_min = _to_minutes(lst_end)
    if start_min is None or end_min is None:
        return None
    diff = end_min - start_min
    if diff < 0:
        diff += 24 * 60
    return diff / 60.0


def _append_note(existing: str, addition: str) -> str:
    """Append `addition` to a semi-colon separated note string without duplication."""
    parts = [part.strip() for part in existing.split(";") if part.strip()] if existing else []
    if addition not in parts:
        parts.append(addition)
    return "; ".join(parts)


def CheckSources(
    csv0: str,
    epoch_targets: Dict[str, List[dict]],
    ntag0: str = "",
    ratag0: str = "",
    dectag0: str = "",
):
    """
    Compare the proposal master catalogue against the science targets per JSON epoch.

    The master CSV is updated with a `Notes` column that records the epoch file(s)
    where each source appears, and a summary of positional separations is printed.
    """
    print("\n ******\n TARGET NAMES & COORDINATES checks\n ******")

    d0 = p.read_csv(str(csv0))
    if not len(d0.columns):
        raise ValueError("Master CSV file has no columns.")

    if not ntag0:
        ntag0 = d0.columns[0]
    if not ratag0:
        ratag0 = _infer_column(d0.columns, "ra")
    if not dectag0:
        dectag0 = _infer_column(d0.columns, "dec")

    if not ratag0 or not dectag0:
        raise ValueError("Could not infer RA/DEC columns from master CSV.")

    if "Notes" not in d0.columns:
        d0["Notes"] = ""

    master_records = []
    for idx, row in d0.iterrows():
        master_records.append(
            {
                "index": idx,
                "name": str(row[ntag0]),
                "ra": str(row[ratag0]),
                "dec": str(row[dectag0]),
            }
        )

    min_sep_dict = {record["name"]: float("inf") for record in master_records}
    unmatched_targets: List[tuple] = []

    for fpath, targets in epoch_targets.items():
        sci_targets = [
            target for target in targets if "target" in target.get("tags", [])
        ]
        print(f"\n'{os.path.basename(csv0)}'  v.s. '{os.path.basename(fpath)}'")
        if not sci_targets:
            print(bblue(" No science targets listed in this epoch file."))
            continue

        for epoch_target in sci_targets:
            best_match = None
            for record in master_records:
                if target_true(record["name"], epoch_target["name"]):
                    sep = separation_radec(
                        record["ra"], record["dec"], epoch_target["ra"], epoch_target["dec"]
                    )
                    if np.isnan(sep[1]):
                        continue
                    if (best_match is None) or (sep[1] < best_match["sep_arcsec"]):
                        best_match = {
                            "record": record,
                            "sep_arcsec": sep[1],
                            "sep_arcmin": sep[1] / 60.0,
                        }

            if best_match:
                record = best_match["record"]
                idx = record["index"]
                existing = str(d0.loc[idx, "Notes"]) if not p.isna(d0.loc[idx, "Notes"]) else ""
                addition = f"found in {os.path.basename(fpath)}"
                d0.loc[idx, "Notes"] = _append_note(existing, addition)
                min_sep_dict[record["name"]] = min(
                    min_sep_dict[record["name"]], best_match["sep_arcsec"]
                )
                src_string = (
                    f"{record['name']} RA: {record['ra']} DEC: {record['dec']} | "
                    f"{epoch_target['name']} RA: {epoch_target['ra']} DEC: {epoch_target['dec']}, "
                    f"sep: {best_match['sep_arcsec']:.6f}\" ({best_match['sep_arcmin']:.4f} arcmin)"
                )
                if best_match["sep_arcsec"] <= 5.0:
                    print(bblue(src_string))
                else:
                    print(bred(src_string))
            else:
                unmatched_targets.append((fpath, epoch_target["name"]))
                print(
                    bred(
                        f"Science target '{epoch_target['name']}' "
                        f"from {os.path.basename(fpath)} not found in master list."
                    )
                )

    if unmatched_targets:
        print("\nTargets present in epoch files but missing from the proposal catalogue:")
        for fpath, tgt in unmatched_targets:
            print(bred(f" - {tgt} (epoch file {os.path.basename(fpath)})"))
    else:
        print(bblue("\nAll epoch science targets are present in the proposal catalogue."))

    print(f"\n\n{'+' * 30}")
    print("Minimum separations between matched sources and their best epoch counterpart:")
    for record in master_records:
        name = record["name"]
        note = d0.loc[record["index"], "Notes"]
        note = "" if p.isna(note) else note
        note_entries = [entry.strip() for entry in note.split(";") if entry.strip()]
        if not note_entries:
            note_entries = [""]

        min_sep = min_sep_dict[name]
        if np.isfinite(min_sep):
            base_msg = f"{name}: {min_sep:.6f}\" ({min_sep/60.:.4f} arcmin)"
            printer = bblue
        else:
            base_msg = f"{name}: not scheduled in supplied epochs"
            printer = bred

        for entry in note_entries:
            suffix = f" | Notes: {entry};" if entry else " | Notes:"
            print(printer(f"{base_msg}{suffix}"))

    d0.to_csv(str(csv0), index=False)
    print(f"{'+' * 30}\n")
    return {
        "unmatched_epoch_targets": [name for _, name in unmatched_targets],
        "updated_catalogue": csv0,
    }


def _extract_tracked_duration(line: str, target: str) -> float:
    """Extract 'Tracked <target> for <seconds>' durations from a simulation line."""
    pattern = rf"Tracked {re.escape(target)} for ([\d.]+) seconds"
    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    return 0.0


def _extract_observed_duration(line: str, target: str) -> float:
    """Extract '<target> observed for <seconds>' durations from a simulation line."""
    pattern = rf"{re.escape(target)} observed for ([\d.]+) sec"
    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    return 0.0


def GetProjDuration(
    master_csv: str,
    epoch_files: Sequence[str],
    proj_description: str = "",
    show_targ_scans: bool = False,
    name_column: str = "",
    ra_column: str = "",
    dec_column: str = "",
):
    """
    Aggregate project duration, scan statistics, and target checks from JSON epochs.

    Parameters:
        master_csv: proposal catalogue used as the master target list.
        epoch_files: sequence of JSON paths containing OPT simulation data.
        proj_description: optional string to override the derived proposal_id.
        show_targ_scans: when True, prints each target scan duration encountered.
        name_column/ra_column/dec_column: optional overrides for master CSV columns.
    """
    if not epoch_files:
        raise ValueError("No epoch JSON files were provided.")

    epoch_data = [_load_epoch_json(f) for f in epoch_files]
    project_ids = {d.get("proposal_id") for d in epoch_data if d.get("proposal_id")}

    if not proj_description:
        if len(project_ids) == 1:
            proj_description = next(iter(project_ids))
        else:
            proj_description = ", ".join(sorted(pid for pid in project_ids if pid)) or "Unknown project"

    print(f"\n\n{'#' * 30}\nPROJECT: {proj_description}")
    print(f"{'#' * 30}\n")

    instrument_keys = [
        "product",
        "integration_time",
        "band",
        "center_freq",
        "pool_resources",
        "config_auth_host",
    ]

    Tmm = 0.0  # total obs time in minutes
    Tth = 0.0  # total time on target for entire project, in hrs
    all_target_scans: List[float] = []
    all_gain_scans: List[float] = []
    all_bp_scans: List[float] = []
    all_pol_scans: List[float] = []
    band_durations: Dict[str, float] = {}

    for epoch in epoch_data:
        gscans: List[float] = []
        bpscans: List[float] = []
        lsc: List[float] = []
        pscans: List[float] = []
        bp_scan_times: List[float] = []
        pol_scan_times: List[float] = []
        fname = os.path.basename(epoch["file"])
        epoch_duration_hr = 0.0

        print(f"\n{'*' * 30}\nEPOCH FILE: {fname}\n{'*' * 30}")
        instrument = epoch.get("instrument") or {}
        band_label = str(instrument.get("band") or "unknown").strip() or "unknown"

        if instrument:
            print(" Instrument info:")
            for key in instrument_keys:
                value = instrument.get(key, "N/A")
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                print(f"   {key:>16}: {value}")
        else:
            print(" Instrument info: not provided")

        description = epoch.get("description") or "N/A"
        print(f" Description: {description}")
        obs_setup = epoch.get("obs_setup") or {}
        general_comments = str(obs_setup.get("general_comments", "") or "").strip() or "N/A"
        
        print(f" Obs setup comments: {general_comments}")
        print(f"  - mandatory_night_obs : {obs_setup.get('mandatory_night_obs', 'N/A')}")
        print(f"  - avoid_sunrise_sunset: {obs_setup.get('avoid_sunrise_sunset', 'N/A')}")
        base_req = obs_setup.get("baseline_requirements")
        base_req = base_req if base_req else "N/A"
        print(f"  - baseline_requirements: {base_req}")
        print(f"  - minimum_antennas    : {obs_setup.get('minimum_antennas', 'N/A')}")
        lst_start = epoch.get("lst_start") or "N/A"
        lst_end = epoch.get("lst_start_end") or "N/A"
        lst_hours = _lst_duration_hours(epoch.get("lst_start"), epoch.get("lst_start_end"))
        if lst_hours is not None:
            print(f" LST span: {lst_start} -> {lst_end} ({lst_hours:.2f} hrs)")
        else:
            print(f" LST span: {lst_start} -> {lst_end} (duration unavailable)")
        obs_type = epoch.get("observation_type") or "N/A"
        print(f" Observation type: {obs_type}")

        sci = 0
        current_time = 0.0
        targs: List[str] = []
        gcals: List[str] = []
        bpcals: List[str] = []
        pcals: List[str] = []

        for line in epoch["simulation_lines"]:
            if "Setting up telescope for observation" in line:
                print(
                    bblue(
                        f"\n NB: Epoch sim START: {line.replace('Setting up telescope for observation', '').split('-', 1)[-1].strip()}"
                    )
                )

            if "Observation targets are" in line:
                targs = _safe_literal_list(line.split("targets are", 1)[1])
                print(f" Target names list : {targs}","\n BP & POL cal Scan sequence:")
            if "GAIN calibrators are" in line:
                gcals = _safe_literal_list(line.split("GAIN calibrators are", 1)[1])
                print(f" Gaincals names list : {gcals}")
            if "BP calibrators are" in line:
                bpcals = _safe_literal_list(line.split("BP calibrators are", 1)[1])
                print(f" BPcals names list : {bpcals}")
            if "POL calibrators are" in line:
                pcals = _safe_literal_list(line.split("POL calibrators are", 1)[1])
                print(f" POLcals names list : {pcals}")
            
            if "Resetting all noise diodes to" in line:
                print(bblue(f" NB: Epoch sim END: {line.replace('Resetting all noise diodes to', '').split('-', 1)[-1].split('\"off\"')[0]}"))
            for gc in gcals:
                dur = _extract_tracked_duration(line, gc)
                if dur:
                    gscans.append(dur)
            for pc in pcals:
                dur = _extract_tracked_duration(line, pc)
                if dur:
                    pscans.append(dur)
                    print(f" - POLcal scan: {line}")
                    pol_scan_times.append(current_time)
            for bc in bpcals:
                dur = _extract_tracked_duration(line, bc)
                if dur:
                    bpscans.append(dur)
                    bp_scan_times.append(current_time)
                    print(f" - BPcal scan : {line}")
            for tgt in targs:
                if "observed for" in line and tgt in line:
                    tt = _extract_observed_duration(line, tgt)
                    if tt:
                        print(
                            redtext(
                                f"\tTot. time on TARGET for this EPOCH {tgt}: {tt/60.:.1f} min ({tt/3600.:.3f} hrs)"
                            )
                        )
                dur = _extract_tracked_duration(line, tgt)
                if dur:
                    sci += 1
                    lsc.append(dur)
                    if show_targ_scans:
                        print(f"\t > {tgt} scan {sci} length: {dur/60.:.1f} min")

            match = re.search(r"Tracked .+ for ([\d.]+) seconds", line)
            if match:
                current_time += float(match.group(1))

            if "Total observation time" in line:
                mm = float(line.split("sec ")[1].strip("(min)"))
                Tmm += mm
                Tth += np.nansum(np.array(lsc)) / 3600.0
                epoch_duration_hr = mm / 60.0
                # print(f" Simulation file name: {fname}")
                print(bblue(f"\tEPOCH DURATION         : {mm/60.:.4f} hrs"))
                print(bblue(f"\tNumber of BPcal scans  : {len(bpscans):d} = {np.nansum(bpscans)/60.:.1f} min"))
                print(bblue(f"\tNumber of gaincal scans: {len(gscans):d} = {np.nansum(gscans)/60.:.1f} min"))
                print(bblue(f"\tNumber of POLcal scans : {len(pscans):d} = {np.nansum(pscans)/60.:.1f} min"))
                print(bblue(f"\tNumber of target scans : {len(lsc):d} = {np.nansum(lsc)/60.:.1f} min"))

        if len(bp_scan_times) > 1:
            print(bblue("\tTime between BPcal scans:"))
            for i in range(1, len(bp_scan_times)):
                diff_sec = bp_scan_times[i] - bp_scan_times[i - 1]
                diff_min = diff_sec / 60.0
                diff_hr = diff_sec / 3600.0
                print(f"\t\tBPcal scan {i}: {diff_min:.2f} min ({diff_hr:.3f} hr) since previous BPcal scan")
        else:
            print(bblue("Less than two BPcal scans found, no time differences to report."))
        if len(pol_scan_times) > 1:
            print(bblue("\tTime between POLcal scans:"))
            for i in range(1, len(pol_scan_times)):
                diff_sec = pol_scan_times[i] - pol_scan_times[i - 1]
                diff_min = diff_sec / 60.0
                diff_hr = diff_sec / 3600.0
                print(f"\t\tPOLcal scan {i}: {diff_min:.2f} min ({diff_hr:.3f} hr) since previous POLcal scan")
        else:
            print(bblue("Less than two POLcal scans found, no time differences to report."))

        all_target_scans.extend(lsc)
        all_gain_scans.extend(gscans)
        all_bp_scans.extend(bpscans)
        all_pol_scans.extend(pscans)
        band_durations[band_label] = band_durations.get(band_label, 0.0) + epoch_duration_hr

    lsc_arr = np.array(all_target_scans)
    gsc_arr = np.array(all_gain_scans)
    bsc_arr = np.array(all_bp_scans)
    psc_arr = np.array(all_pol_scans)

    print(bred(f"\n{'*' * 50}"))
    
    if len(lsc_arr):
        print(
            bblue(
                f"Target scan length(min, max, stdev) = {np.nanmin(lsc_arr)/60.:.1f},"
                f"{np.nanmax(lsc_arr)/60.:.1f},{np.nanstd(lsc_arr)/60.:.1f} min"
            )
        )
    if len(gsc_arr):
        print(
            bblue(
                f"Gaincal scan length(min, max, stdev)= {np.nanmin(gsc_arr)/60.:.1f},"
                f"{np.nanmax(gsc_arr)/60.:.1f},{np.nanstd(gsc_arr)/60.:.1f} min"
            )
        )
    if len(bsc_arr):
        print(
            bblue(
                f"BP cal scan length(min, max, stdev) = {np.nanmin(bsc_arr)/60.:.1f},"
                f"{np.nanmax(bsc_arr)/60.:.1f},{np.nanstd(bsc_arr)/60.:.1f} min"
            )
        )
    if len(psc_arr):
        print(
            bblue(
                f"POL cal scan length(min, max, stdev)= {np.nanmin(psc_arr)/60.:.1f},"
                f"{np.nanmax(psc_arr)/60.:.1f},{np.nanstd(psc_arr)/60.:.1f} min"
            )
        )

    print(bred(f"\n**** TOTAL time on target for all epochs: {Tth} hrs"))
    print(bred(f"**** TOTAL PROJECT TIME                 : {Tmm/60.:.1f} Hrs "))
    print(bred(f"**** TOTAL PROJECT TIME - 25% overheads : {(Tmm/60.)/(1.25):.1f} Hrs "))
    print(bred(f"{'*' * 50}\n"))
    if band_durations:
        print(bblue("Time per instrument band:"))
        for band, hours in sorted(band_durations.items()):
            print(bblue(f"  Band '{band.capitalize()}': {hours:.2f} hrs"))

    epoch_target_map = {epoch["file"]: epoch["targets"] for epoch in epoch_data}
    try:
        CheckSources(
            csv0=master_csv,
            epoch_targets=epoch_target_map,
            ntag0=name_column,
            ratag0=ra_column,
            dectag0=dec_column,
        )
    except KeyError:
        CheckSources(
            csv0=master_csv,
            epoch_targets=epoch_target_map,
            ntag0="Name",
            ratag0="RA",
            dectag0="DEC",
        )

    # print(f"\n\n{'*' * 30}\n")
    # print("MASTER LIST contents:")
    # print(f"{'*' * 30}\n")

    # with open(master_csv, "r") as handle:
    #     print(handle.read(), "\n")
    # print("\n >>>> END of MASTER LIST contents >>>\n")


def _expand_patterns(patterns: Sequence[str]) -> List[str]:
    """Expand file paths/glob patterns passed via CLI and warn if none match."""
    files: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.exists(pattern):
            files.append(pattern)
        else:
            print(redtext(f"Warning: '{pattern}' did not match any files."), file=sys.stderr)
    return files


def parse_args():
    """Create and return the command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="Audit OPT JSON epoch files and compare to a proposal target list.",
    )
    parser.add_argument(
        "--master-csv", required=True, help="Proposal catalogue CSV file (master target list)."
    )
    parser.add_argument(
        "--epoch-json",
        required=True,
        nargs="+",
        help="One or more JSON epoch files or glob patterns.",
    )
    parser.add_argument(
        "--proj-description",
        default="",
        help="Optional project description to override the proposal_id value.",
    )
    parser.add_argument(
        "--show-target-scans",
        action="store_true",
        help="Print individual science target scan lengths.",
    )
    parser.add_argument(
        "--name-column",
        default="",
        help="Override the master CSV column that stores source names.",
    )
    parser.add_argument(
        "--ra-column",
        default="",
        help="Override the master CSV column that stores RA values.",
    )
    parser.add_argument(
        "--dec-column",
        default="",
        help="Override the master CSV column that stores DEC values.",
    )
    return parser.parse_args()


def main():
    """Entry point that parses CLI arguments and executes the audit."""
    args = parse_args()
    epoch_files = _expand_patterns(args.epoch_json)
    if not epoch_files:
        print(redtext("No epoch JSON files available after glob expansion."), file=sys.stderr)
        sys.exit(1)
    GetProjDuration(
        master_csv=args.master_csv,
        epoch_files=sorted(epoch_files),
        proj_description=args.proj_description,
        show_targ_scans=args.show_target_scans,
        name_column=args.name_column,
        ra_column=args.ra_column,
        dec_column=args.dec_column,
    )


if __name__ == "__main__":
    main()
