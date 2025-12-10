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
import itertools
import json
import os
import re
import sys
import traceback
from contextlib import contextmanager
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


def _canonical_name(name: str) -> str:
    """Return a simplified, lower-case version of a target name for fuzzy matching."""
    return re.sub(r"[\s|]+", "", str(name)).lower()


def _ordinal_suffix(value: int) -> str:
    """Return the ordinal suffix for an integer (1st, 2nd, 3rd, etc.)."""
    if 10 <= value % 100 <= 20:
        return "th"
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    return suffixes.get(value % 10, "th")


def _match_entry(name: str, lookup: Dict[str, dict]) -> dict:
    """Find the best-matching entry for `name` in a pre-built lookup."""
    key = _canonical_name(name)
    if key in lookup:
        return lookup[key]
    for stored_key, entry in lookup.items():
        if target_true(stored_key, key) or target_true(key, stored_key):
            return entry
    return {}


def _build_coord_lookup(targets: Sequence[dict], tag: str) -> Dict[str, dict]:
    """Create a name->entry lookup for targets matching a specific tag."""
    lookup: Dict[str, dict] = {}
    for entry in targets or []:
        tags = entry.get("tags") or []
        if tag in tags:
            key = _canonical_name(entry.get("name", ""))
            if key and key not in lookup:
                lookup[key] = entry
    return lookup


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


def _build_category_lookup(targets: Sequence[dict]) -> Dict[str, dict]:
    """Return a mapping of canonical target name -> {'name', 'category'}."""
    lookup: Dict[str, dict] = {}
    for entry in targets or []:
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        category = _categorise_target(entry.get("tags") or [])
        key = _canonical_name(name)
        if key and key not in lookup:
            lookup[key] = {"name": name, "category": category}
    return lookup


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


def _categorise_target(tags: Sequence[str]) -> str:
    """Return a coarse category label for a target based on its tags."""
    tags = _normalise_tags(tags)
    if "bpcal" in tags or "bandpass" in tags:
        return "bpcal"
    if "gaincal" in tags:
        return "gaincal"
    if "polcal" in tags or "polarisation" in tags or "polarization" in tags:
        return "polcal"
    if "target" in tags:
        return "target"
    return "other"


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
        "desired_start_time": data.get("desired_start_time"),
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


def _dec_to_degrees(value) -> float:
    """Convert a declination string or number to decimal degrees; return None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("d", ":").replace("m", ":").replace("s", ":")
    parts = text.split(":")
    try:
        deg = float(parts[0])
        sign = -1.0 if str(parts[0]).strip().startswith("-") else 1.0
        minutes = float(parts[1]) if len(parts) > 1 else 0.0
        seconds = float(parts[2]) if len(parts) > 2 else 0.0
        return sign * (abs(deg) + minutes / 60.0 + seconds / 3600.0)
    except Exception:
        return None


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
    matched_pairs: List[dict] = []
    best_match_per_master: Dict[str, dict] = {}

    for fpath, targets in epoch_targets.items():
        sci_targets = [
            target for target in targets if "target" in target.get("tags", [])
        ]
        # print(f"\n'{os.path.basename(csv0)}'  v.s. '{os.path.basename(fpath)}'")
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
                existing_best = best_match_per_master.get(record["name"])
                if (existing_best is None) or (best_match["sep_arcsec"] < existing_best["sep_arcsec"]):
                    best_match_per_master[record["name"]] = {
                        "epoch_name": epoch_target["name"],
                        "master_name": record["name"],
                        "sep_arcsec": best_match["sep_arcsec"],
                        "sep_arcmin": best_match["sep_arcmin"],
                        "epoch_file": os.path.basename(fpath),
                        "ra_master": record["ra"],
                        "dec_master": record["dec"],
                        "ra_epoch": epoch_target["ra"],
                        "dec_epoch": epoch_target["dec"],
                    }
                matched_pairs.append(
                    {
                        "sep_arcsec": best_match["sep_arcsec"],
                        "master_name": record["name"],
                        "epoch_name": epoch_target["name"],
                        "epoch_file": os.path.basename(fpath),
                    }
                )
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

    if best_match_per_master:
        print("\nClosest epoch cross-match per master source:")
        for name in sorted(best_match_per_master.keys()):
            info = best_match_per_master[name]
            base_str = (
                f"{info['master_name']} (master) -> {info['epoch_name']} (epoch {info['epoch_file']}): "
                f"{info['sep_arcsec']:.6f}\" ({info['sep_arcmin']:.4f} arcmin)"
            )
            printer = bblue if info["sep_arcsec"] <= 5.0 else redtext
            print(printer(base_str))
    else:
        print(bred("\nNo matched epoch/master source pairs to report separations."))

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
    crossmatch_extremes = None
    if matched_pairs:
        min_pair = min(matched_pairs, key=lambda item: item["sep_arcsec"])
        max_pair = max(matched_pairs, key=lambda item: item["sep_arcsec"])
        crossmatch_extremes = {"min": min_pair, "max": max_pair}

    return {
        "unmatched_epoch_targets": [name for _, name in unmatched_targets],
        "updated_catalogue": csv0,
        "crossmatch_extremes": crossmatch_extremes,
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


def _find_first_tracked_source(
    sim_lines: Sequence[str], name_categories: Dict[str, dict]
) -> dict:
    """Return info about the first tracked source encountered in the simulation log."""
    name_keys = [
        (value.get("name", key), key) for key, value in name_categories.items()
    ]
    for line in sim_lines:
        if "Tracked" not in line:
            continue
        matched = _match_name_in_line(line, name_keys)
        if matched:
            entry = name_categories.get(_canonical_name(matched), {})
            category = entry.get("category", "other") if isinstance(entry, dict) else entry
            duration = _extract_tracked_duration(line, matched)
            return {
                "name": entry.get("name", matched) if isinstance(entry, dict) else matched,
                "category": category,
                "duration": duration,
                "line": line,
            }
    return {}


def _match_name_in_line(line: str, name_keys: List[tuple]) -> str:
    """Return the original name that matches `line`, if any."""
    norm_line = _canonical_name(line)
    for original, key in name_keys:
        if key and (key in norm_line or target_true(key, norm_line) or target_true(norm_line, key)):
            return original
    return ""


def _find_unbracketed_targets(
    sim_lines: Sequence[str],
    target_names: Sequence[str],
    gaincal_names: Sequence[str],
    obs_start: p.Timestamp = None,
) -> List[dict]:
    """
    Identify target scan lines that are not directly bracketed by gaincal scans.

    Returns a list of dicts for target scans lacking gaincal bracketing with timing info.
    """
    target_keys = [(name, _canonical_name(name)) for name in target_names]
    gain_keys = [(name, _canonical_name(name)) for name in gaincal_names]
    ref_timestamp = obs_start
    target_order = 0
    classified = []

    for idx, line in enumerate(sim_lines):
        ts_match = re.match(r"^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2})Z", line)
        ts_val = None
        if ts_match:
            try:
                ts_val = ts_match.group(1)
                ts_dt = p.to_datetime(ts_val, format="%Y-%m-%d %H:%M:%S")
                if ref_timestamp is None:
                    ref_timestamp = ts_dt
            except Exception:
                ts_dt = None
        else:
            ts_dt = None

        if "Tracked" not in line:
            continue
        matched_target = _match_name_in_line(line, target_keys)
        if matched_target:
            target_order += 1
            classified.append((idx, line, "target", ts_dt, target_order))
            continue
        matched_gain = _match_name_in_line(line, gain_keys)
        if matched_gain:
            classified.append((idx, line, "gaincal", ts_dt, None))

    problems = []
    for pos, (idx, line, kind, ts_dt, order) in enumerate(classified):
        if kind != "target":
            continue
        prev_kind = classified[pos - 1][2] if pos > 0 else None
        next_kind = classified[pos + 1][2] if pos + 1 < len(classified) else None
        if prev_kind != "gaincal" or next_kind != "gaincal":
            delta_sec = None
            if ts_dt is not None and ref_timestamp is not None:
                delta = ts_dt - ref_timestamp
                delta_sec = delta.total_seconds()
            problems.append(
                {
                    "line": line,
                    "delta_sec": delta_sec,
                    "order": order if order is not None else idx,
                }
            )
    return problems


def _print_closest_gaincal_separations(
    targs: Sequence[str], gcals: Sequence[str], target_lookup: Dict[str, dict], gain_lookup: Dict[str, dict]
) -> None:
    """Print the closest gaincal for each target based on angular separation."""
    if not targs or not gcals:
        return
    print(" - Closest gaincal to each target (by angular separation):")
    for tgt_name in targs:
        t_entry = _match_entry(tgt_name, target_lookup)
        if not t_entry:
            print(redtext(f"  - Missing coordinates for target '{tgt_name}', skipping."))
            continue
        best_sep = None
        best_gain = None
        for gc_name in gcals:
            g_entry = _match_entry(gc_name, gain_lookup)
            if not g_entry:
                continue
            sep_deg, _ = separation_radec(t_entry.get("ra"), t_entry.get("dec"), g_entry.get("ra"), g_entry.get("dec"))
            if np.isnan(sep_deg):
                continue
            if (best_sep is None) or (sep_deg < best_sep):
                best_sep = sep_deg
                best_gain = g_entry.get("name", gc_name)
        if best_sep is not None and best_gain:
            print(bblue(f"  -> {t_entry.get('name', tgt_name)} -> {best_gain}: {best_sep:.4f} deg\n"))
        else:
            print(redtext(f"  - No usable gaincal coordinates found for '{t_entry.get('name', tgt_name)}'."))


def _report_gaincal_bracketing(
    sim_lines: Sequence[str],
    targs: Sequence[str],
    gcals: Sequence[str],
    target_lookup: Dict[str, dict],
    gain_lookup: Dict[str, dict],
    obs_start: p.Timestamp = None,
) -> bool:
    """
    Check that each target scan is preceded and followed by gaincal scans and print context.

    Returns True when all target scans are bracketed by gaincal scans; False otherwise.
    """
    target_names_for_check = targs or [entry["name"] for entry in target_lookup.values()]
    gaincal_names_for_check = gcals or [entry["name"] for entry in gain_lookup.values()]
    if not target_names_for_check or not gaincal_names_for_check:
        print(redtext("Insufficient target/gaincal names to check scan ordering."))
        return False
    problematic_scans = _find_unbracketed_targets(
        sim_lines, target_names_for_check, gaincal_names_for_check, obs_start
    )
    if problematic_scans:
        print(bred("(!) Target scans not bracketed by gaincal scans:"))
        sorted_scans = sorted(
            problematic_scans,
            key=lambda item: (float("inf") if item["delta_sec"] is None else item["delta_sec"], item["order"]),
        )
        for idx, item in enumerate(sorted_scans, start=1):
            delta_min = item["delta_sec"] / 60.0 if item["delta_sec"] is not None else None
            timing = f"{delta_min:.2f} min from obs start" if delta_min is not None else "time from obs start unknown"
            print(bred(f"  {idx}{_ordinal_suffix(idx)} target scan ({timing}): {item['line']}"))
        return False
    print(bblue(" All target scans are preceded and followed by gaincal scans."))
    return True


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
    band_epoch_counts: Dict[str, int] = {}
    overlap_candidates: List[dict] = []
    target_declinations: Dict[str, List[float]] = {}
    unique_science_targets = set()
    lst_span_records: List[dict] = []
    epoch_science_targets: Dict[str, set] = {}
    target_band_seconds: Dict[str, Dict[str, float]] = {}
    gaincal_duration_fail_epochs = 0
    gaincal_bracket_fail_epochs = 0
    first_bpcal_fail_epochs = 0

    for epoch in epoch_data:
        gscans: List[float] = []
        bpscans: List[float] = []
        lsc: List[float] = []
        pscans: List[float] = []
        bp_scan_times: List[float] = []
        pol_scan_times: List[float] = []
        fname = os.path.basename(epoch["file"])
        epoch_duration_hr = 0.0
        epoch_start_dt: p.Timestamp = None
        desired_start = epoch.get("desired_start_time")
        if desired_start:
            try:
                epoch_start_dt = p.to_datetime(desired_start)
            except Exception:
                epoch_start_dt = None
        target_lookup = _build_coord_lookup(epoch["targets"], "target")
        gain_lookup = _build_coord_lookup(epoch["targets"], "gaincal")
        separation_reported = False
        bracketing_reported = False
        bp_pol_header_printed = False
        epoch_endline = ""
        name_category_lookup = _build_category_lookup(epoch["targets"])
        first_tracked_info = _find_first_tracked_source(epoch["simulation_lines"], name_category_lookup)
        epoch_gaincal_duration_ok: bool = None  # type: ignore
        epoch_bracketing_ok: bool = None  # type: ignore
        epoch_first_bpcal_ok: bool = None  # type: ignore

        print(f"\n{'*' * 30}\nEPOCH FILE: {fname}\n{'*' * 30}")
        instrument = epoch.get("instrument") or {}
        band_label = str(instrument.get("band") or "unknown").strip() or "unknown"
        band_epoch_counts[band_label] = band_epoch_counts.get(band_label, 0) + 1

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
        print(f" Epoch DESCRIPTION: \"{description}\"")
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
        lst_span_records.append(
            {
                "file": fname,
                "start": lst_start,
                "end": lst_end,
                "duration_hours": lst_hours,
            }
        )
        if first_tracked_info:
            duration_min = first_tracked_info.get("duration", 0.0) / 60.0
            category = first_tracked_info.get("category", "other")
            if category == "bpcal":
                print(bblue(f" First tracked source is BPcal '{first_tracked_info['name']}' ({duration_min:.2f} min)"))
                epoch_first_bpcal_ok = True
            else:
                print(
                    bred(
                        f" First tracked source is NOT a BPcal (got '{first_tracked_info['name']}', category '{category}', "
                        f"{duration_min:.2f} min)"
                    )
                )
                epoch_first_bpcal_ok = False
        else:
            print(bred(" Could not identify the first tracked astronomical source in the simulation log."))
            epoch_first_bpcal_ok = False

        sci = 0
        current_time = 0.0
        targs: List[str] = []
        gcals: List[str] = []
        bpcals: List[str] = []
        pcals: List[str] = []

        for line in epoch["simulation_lines"]:
            if "Setting up telescope for observation" in line:
                ts_match = re.match(r"^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2})Z", line)
                if ts_match:
                    try:
                        epoch_start_dt = p.to_datetime(ts_match.group(1), format="%Y-%m-%d %H:%M:%S")
                    except Exception:
                        epoch_start_dt = None
                print(
                    bblue(
                        f"\n NB: Epoch sim START: {line.replace('Setting up telescope for observation', '').split('-', 1)[-1].strip()}"
                    ),"  ---- ||"
                )

            if "Observation targets are" in line:
                targs = _safe_literal_list(line.split("targets are", 1)[1])
                print(f" Target names list : {targs}")
                if not bracketing_reported and gcals:
                    epoch_bracketing_ok = _report_gaincal_bracketing(
                        epoch["simulation_lines"], targs, gcals, target_lookup, gain_lookup, epoch_start_dt
                    )
                    bracketing_reported = True
                if gcals and not separation_reported:
                    _print_closest_gaincal_separations(targs, gcals, target_lookup, gain_lookup)
                    separation_reported = True
                if not bp_pol_header_printed:
                    print("\n BP & POL cal Scan sequence:")
                    bp_pol_header_printed = True
            if "GAIN calibrators are" in line:
                gcals = _safe_literal_list(line.split("GAIN calibrators are", 1)[1])
                print(f" Gaincals names list : {gcals}")
                if targs and not bracketing_reported:
                    epoch_bracketing_ok = _report_gaincal_bracketing(
                        epoch["simulation_lines"], targs, gcals, target_lookup, gain_lookup, epoch_start_dt
                    )
                    bracketing_reported = True
                if targs and not separation_reported:
                    _print_closest_gaincal_separations(targs, gcals, target_lookup, gain_lookup)
                    separation_reported = True
                if not bp_pol_header_printed:
                    # print("\n BP & POL cal Scan sequence:")
                    bp_pol_header_printed = True
            if "BP calibrators are" in line:
                bpcals = _safe_literal_list(line.split("BP calibrators are", 1)[1])
                print(f" BPcals names list : {bpcals}")
            if "POL calibrators are" in line:
                pcals = _safe_literal_list(line.split("POL calibrators are", 1)[1])
                print(f" POLcals names list : {pcals}")
            
            if "Resetting all noise diodes to" in line:
                epoch_endline = (bblue(f"\nNB: Epoch sim END: {line.replace('Resetting all noise diodes to', '').split('-', 1)[-1].split('\"off\"')[0]}"))
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
                            bblue(
                                f"\tTot. time on TARGET for this EPOCH {tgt}: {tt/60.:.1f} min ({tt/3600.:.3f} hrs)"
                            )
                        )
                dur = _extract_tracked_duration(line, tgt)
                if dur:
                    sci += 1
                    lsc.append(dur)
                    tgt_entry = target_band_seconds.setdefault(tgt, {})
                    tgt_entry[band_label] = tgt_entry.get(band_label, 0.0) + dur
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
                print(f"\nDurations:")
                print(bblue(f"\tEPOCH DURATION         : {mm/60.:.4f} hrs"))
                print(bblue(f"\tNumber of BPcal scans  : {len(bpscans):d} = {np.nansum(bpscans)/60.:.1f} min"))
                print(bblue(f"\tNumber of gaincal scans: {len(gscans):d} = {np.nansum(gscans)/60.:.1f} min"))
                print(bblue(f"\tNumber of POLcal scans : {len(pscans):d} = {np.nansum(pscans)/60.:.1f} min"))
                print(bblue(f"\tNumber of target scans : {len(lsc):d} = {np.nansum(lsc)/60.:.1f} min"))

        if gscans:
            tol_seconds = 5.0
            all_two_min = all(abs(dur - 120.0) <= tol_seconds for dur in gscans)
            if all_two_min:
                print(
                    bblue(
                        f" Gaincal scan duration check: all {len(gscans)} scans ~2 min "
                        f"(mean {np.nanmean(gscans)/60.:.2f} min)"
                    )
                )
                epoch_gaincal_duration_ok = True
            else:
                print(
                    bred(
                        f" Gaincal scan duration check FAILED: expected 2.00 min, "
                        f"found min/max {np.nanmin(gscans)/60.:.2f}/{np.nanmax(gscans)/60.:.2f} min"
                    )
                )
                off_vals = [f"{dur/60.:.2f}" for dur in gscans if abs(dur - 120.0) > tol_seconds]
                if off_vals:
                    print(bred(f"  Offending durations (min): {', '.join(off_vals)}"))
                epoch_gaincal_duration_ok = False
        else:
            print(bred(" Gaincal scan duration check skipped: no gaincal scans found."))
            epoch_gaincal_duration_ok = False

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
            print(bblue("\tLess than two POLcal scans found, no time differences to report."))

        print(f"{epoch_endline} ---- ||\n")
        all_target_scans.extend(lsc)
        all_gain_scans.extend(gscans)
        all_bp_scans.extend(bpscans)
        all_pol_scans.extend(pscans)
        band_durations[band_label] = band_durations.get(band_label, 0.0) + epoch_duration_hr
        if epoch_gaincal_duration_ok is not True:
            gaincal_duration_fail_epochs += 1
        if epoch_bracketing_ok is not True:
            gaincal_bracket_fail_epochs += 1
        if epoch_first_bpcal_ok is not True:
            first_bpcal_fail_epochs += 1
        for entry in epoch.get("targets", []):
            if "target" in entry.get("tags", []):
                name = str(entry.get("name", "")).strip()
                if name:
                    unique_science_targets.add(name)
                    dec_val = _dec_to_degrees(entry.get("dec"))
                    if dec_val is not None:
                        target_declinations.setdefault(name, []).append(dec_val)
        epoch_science_targets[fname] = {
            str(entry.get("name", "")).strip()
            for entry in epoch.get("targets", [])
            if "target" in entry.get("tags", []) and str(entry.get("name", "")).strip()
        }

        # Collect info for overlapping observations with the same target(s).
        if epoch_start_dt is not None and epoch_duration_hr and targs:
            desc_lower = (description or "").lower()
            overlap_candidates.append(
                {
                    "file": fname,
                    "targets": set(targs),
                    "start": epoch_start_dt,
                    "end": epoch_start_dt + p.Timedelta(hours=epoch_duration_hr),
                    "description": description,
                    "desc_lower": desc_lower,
                    "band": band_label,
                }
            )

    lsc_arr = np.array(all_target_scans)
    gsc_arr = np.array(all_gain_scans)
    bsc_arr = np.array(all_bp_scans)
    psc_arr = np.array(all_pol_scans)

    if overlap_candidates:
        print(bblue("Overlap between epochs with matching targets (rising/setting or same band):"))
        found_overlap = False
        for left, right in itertools.combinations(overlap_candidates, 2):
            if left["targets"] != right["targets"]:
                continue
            desc_cond = (
                ("rising" in left["desc_lower"] or "setting" in left["desc_lower"])
                and ("rising" in right["desc_lower"] or "setting" in right["desc_lower"])
            )
            band_cond = (
                bool(left["band"])
                and bool(right["band"])
                and left["band"].lower() == right["band"].lower()
            )
            if not (desc_cond or band_cond):
                continue
            latest_start = max(left["start"], right["start"])
            earliest_end = min(left["end"], right["end"])
            overlap = (earliest_end - latest_start).total_seconds() / 3600.0
            overlap = max(0.0, overlap)
            reason_parts = []
            if desc_cond:
                reason_parts.append("rising/setting")
            if band_cond:
                reason_parts.append(f"same band ({left['band'].capitalize()})")
            reason = " and ".join(reason_parts) if reason_parts else "matching targets"
            found_overlap = True
            print(f"  {left['file']} (\"{left['description']}\") and {right['file']} (\"{right['description']}\") ",
                    bblue(f"overlap by {overlap:.3f} hrs [{reason}]") )
                
        if not found_overlap:
            print(bblue("  No qualifying rising/setting or same-band pairs with matching targets found."))

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
    requested_seconds = _sum_requested_time(master_csv)
    observed_seconds = Tth * 3600.0
    if requested_seconds is not None:
        req_hours = requested_seconds / 3600.0
        diff_hours = observed_seconds / 3600.0 - req_hours
        print("\nRequested vs simulated on-target time:")
        print(bblue(f" - Requested from master CSV: {req_hours:.3f} hrs"))
        print(bblue(f" - Simulated from epochs    : {observed_seconds/3600.0:.3f} hrs"))
        if abs(diff_hours) <= 1e-3:
            print(bblue(" Requested on-target time matches the simulated total (within 0.001 hr)."))
        else:
            print(bred(f" Requested vs simulated mismatch: {diff_hours:+.3f} hrs"))
    else:
        print(bred("\nRequested on-target time could not be computed from master CSV."))
    print(bred(f"{'*' * 50}\n"))
    if band_durations:
        print(bblue("Time per instrument band:"))
        for band, hours in sorted(band_durations.items()):
            print(bblue(f"  Band '{band.capitalize()}': {hours:.2f} hrs"))

    epoch_target_map = {epoch["file"]: epoch["targets"] for epoch in epoch_data}
    source_check_result = {}
    try:
        source_check_result = CheckSources(
            csv0=master_csv,
            epoch_targets=epoch_target_map,
            ntag0=name_column,
            ratag0=ra_column,
            dectag0=dec_column,
        )
    except KeyError:
        source_check_result = CheckSources(
            csv0=master_csv,
            epoch_targets=epoch_target_map,
            ntag0="Name",
            ratag0="RA",
            dectag0="DEC",
        )

    if len(epoch_files) > 1:
        total_project_hours = Tmm / 60.0
        mean_target_time = Tth / float(len(epoch_files)) if epoch_files else 0.0
        unmatched_count = len(source_check_result.get("unmatched_epoch_targets", []))
        crossmatch_extremes = source_check_result.get("crossmatch_extremes")
        all_decs = [deg for values in target_declinations.values() for deg in values]
        near_limit_targets = sorted(
            (name, decs)
            for name, decs in target_declinations.items()
            if any(abs(dec - 40.0) <= 15.0 for dec in decs)
        )
        north_of_limit_targets = sorted(
            name for name, decs in target_declinations.items() if any(dec > 40.0 for dec in decs)
        )

        print()
        print("=" * 60)
        print("PROJECT SUMMARY")
        print("=" * 60)
        print(f"Epoch JSON files processed: {len(epoch_files)}")
        print(f"Unique science targets    : {len(unique_science_targets)}")
        print("\nTime statistics (hours):")
        print(f" - Total on-target time         : {Tth:.3f}")
        print(f" - Mean on-target time per epoch: {mean_target_time:.3f}")
        print(f" - Total project time           : {total_project_hours:.3f}")
        print(f" - Project time minus 25% ovhds : {(total_project_hours/1.25):.3f}")
        print("\nQuality checks (epoch counts):")
        print(f" - Gaincal duration failures   : {gaincal_duration_fail_epochs}")
        print(f" - Gaincal bracketing failures : {gaincal_bracket_fail_epochs}")
        print(f" - BPcal-first failures        : {first_bpcal_fail_epochs}")
        if requested_seconds is not None:
            req_hours = requested_seconds / 3600.0
            sim_hours = observed_seconds / 3600.0
            diff_hours = sim_hours - req_hours
            print("\nRequested vs simulated on-target time:")
            print(bblue(f" - Requested from master CSV: {req_hours:.3f} hrs"))
            print(bblue(f" - Simulated from epochs    : {sim_hours:.3f} hrs"))
            if abs(diff_hours) <= 1e-3:
                print(bblue(" Requested on-target time matches the simulated total (within 0.001 hr)."))
            else:
                print(bred(f" Requested vs simulated mismatch: {diff_hours:+.3f} hrs"))
        else:
            print(bred("\nRequested on-target time could not be computed from master CSV."))
        if crossmatch_extremes:
            print("\nCross-match separation extremes:")
            min_pair = crossmatch_extremes.get("min")
            max_pair = crossmatch_extremes.get("max")
            if min_pair:
                print(
                    bblue(
                        f" - Closest: {min_pair['epoch_name']} (epoch {min_pair['epoch_file']}) "
                        f"-> {min_pair['master_name']} (master) = {min_pair['sep_arcsec']:.3f}\""
                    )
                )
            if max_pair:
                print(
                    bblue(
                        f" - Farthest: {max_pair['epoch_name']} (epoch {max_pair['epoch_file']}) "
                        f"-> {max_pair['master_name']} (master) = {max_pair['sep_arcsec']:.3f}\""
                    )
                )
        if band_durations:
            print("\nBand breakdown:")
            for band in sorted(band_durations.keys()):
                hours = band_durations[band]
                count = band_epoch_counts.get(band, 0)
                print(f" - Band '{band}': {hours:.3f} hrs across {count} epoch(s)")
        if all_decs:
            print("\nDeclination coverage (science targets):")
            print(f" - Range: {min(all_decs):.2f} deg to {max(all_decs):.2f} deg")
            if near_limit_targets:
                formatted_near = []
                for name, decs in near_limit_targets:
                    unique_decs = sorted({dec for dec in decs})
                    dec_list = ", ".join(f"{dec:.2f}" for dec in unique_decs)
                    formatted_near.append(f"{name} ({dec_list})")
                print(
                    f" - Targets within 15 deg of +40: {len(near_limit_targets)} -> "
                    f"{', '.join(formatted_near)}"
                )
            else:
                print(" - Targets within 15 deg of +40: 0")
            if north_of_limit_targets:
                print(
                    f" - Targets north of +40 (unobservable): {len(north_of_limit_targets)} -> "
                    f"{', '.join(north_of_limit_targets)}"
                )
            else:
                print(" - Targets north of +40 (unobservable): 0")
        if target_band_seconds:
            print("\nTime on target per band (hours):")
            for tgt in sorted(target_band_seconds.keys()):
                entries = target_band_seconds[tgt]
                band_strings = [f"{band.capitalize()}: {seconds/3600.:.3f}" for band, seconds in sorted(entries.items())]
                print(f" - {tgt}: " + "; ".join(band_strings))
        if lst_span_records:
            valid_spans = [rec for rec in lst_span_records if rec["duration_hours"] is not None]
            if valid_spans:
                min_span = min(rec["duration_hours"] for rec in valid_spans)
                max_span = max(rec["duration_hours"] for rec in valid_spans)
                print("\nLST span coverage:")
                print(f" - Range (hrs): {min_span:.2f} to {max_span:.2f}")
                shortest_epochs = [rec for rec in valid_spans if rec["duration_hours"] == min_span]
                for rec in shortest_epochs:
                    print(
                        f" - Shortest LST start epoch: {rec['file']} [{rec['start']} -> {rec['end']}, "
                        f"{rec['duration_hours']:.2f} hrs]"
                    )
                    base_targets = epoch_science_targets.get(rec["file"], set())
                    shared = []
                    for other_file, targets in epoch_science_targets.items():
                        if other_file == rec["file"]:
                            continue
                        common_targets = sorted(base_targets & targets)
                        if common_targets:
                            shared.append((other_file, common_targets))
                    if shared:
                        print("   Shared science targets with:")
                        for other_file, common_targets in shared:
                            print(f"    - {other_file}: {', '.join(common_targets)}")
                    else:
                        print("   No other epochs share science targets with this shortest-LST epoch.")
            else:
                print("\nLST span coverage: no valid LST start/end values available.")
        if unmatched_count:
            print(
                f"\nEpoch targets missing from master catalogue: {unmatched_count} "
                "(see detailed listing above)."
            )
        else:
            print("\nEpoch targets missing from master catalogue: 0")

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


def _derive_log_path(epoch_files: Sequence[str]) -> str:
    """Return the path to the run log beside the supplied epoch JSON files."""
    if not epoch_files:
        return "OPT_SBsummary.md"
    abs_paths = [os.path.abspath(f) for f in epoch_files]
    dirs = {os.path.dirname(path) or "." for path in abs_paths}
    log_dir = sorted(dirs)[0]
    return os.path.join(log_dir, "OPT_SBsummary.md")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences for log cleanliness."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@contextmanager
def _tee_stdout(log_path: str):
    """Mirror stdout to a markdown log file while preserving on-screen output."""
    original_stdout = sys.stdout
    with open(log_path, "w") as log_file:
        class _StdoutTee:
            def write(self, data):
                original_stdout.write(data)
                log_file.write(_strip_ansi(data))
                return len(data)

            def flush(self):
                original_stdout.flush()
                log_file.flush()

        sys.stdout = _StdoutTee()
        try:
            yield
        finally:
            sys.stdout = original_stdout


def _sum_requested_time(master_csv: str):
    """
    Return the total requested time on target in seconds from the master CSV.

    Returns None if the column is missing or cannot be read.
    """
    try:
        df = p.read_csv(master_csv)
    except Exception as exc:
        print(bred(f"Could not read master CSV '{master_csv}': {exc}"))
        return None
    column = "Time on Target (seconds)"
    if column not in df.columns:
        print(bred(f"Column '{column}' not found in master CSV; requested time unavailable."))
        return None
    values = p.to_numeric(df[column], errors="coerce")
    total_seconds = float(values.sum(skipna=True))
    return total_seconds


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
    epoch_files = sorted(epoch_files)
    log_path = _derive_log_path(epoch_files)
    with _tee_stdout(log_path):
        print(bblue(f"Logging output to: {log_path}"))
        if len({os.path.dirname(os.path.abspath(f)) or "." for f in epoch_files}) > 1:
            print(redtext("Note: epoch JSON files span multiple directories; log placed with the first directory."))
        GetProjDuration(
            master_csv=args.master_csv,
            epoch_files=epoch_files,
            proj_description=args.proj_description,
            show_targ_scans=args.show_target_scans,
            name_column=args.name_column,
            ra_column=args.ra_column,
            dec_column=args.dec_column,
        )
    print(bblue(f"Run log saved to: {log_path}"))


if __name__ == "__main__":
    main()
