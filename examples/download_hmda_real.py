#!/usr/bin/env python3
"""
download_hmda_real.py: Download real HMDA data from CFPB
========================================================

Run this on YOUR machine (not in the sandbox) to download real HMDA data.

The CFPB provides HMDA Loan Application Register (LAR) data via:

  1. DATA BROWSER API (filtered CSV, smaller downloads):
     https://ffiec.cfpb.gov/v2/data-browser-api/view/csv
     - Requires: year + at least one geographic param (state/msamd) OR LEI
     - Returns: streamed CSV with selected filters
     - Example: curl -L "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=NY&years=2022&actions_taken=1,3"

  2. SNAPSHOT NATIONAL LOAN-LEVEL DATASET (bulk, ~2-5 GB per year):
     https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/
     - Full national LAR data, published annually
     - Available from S3: https://s3.amazonaws.com/cfpb-hmda-public/prod/snapshot-data/
     - Available years: 2018-2023+

  3. DYNAMIC NATIONAL LOAN-LEVEL DATASET (updated quarterly):
     https://ffiec.cfpb.gov/data-publication/dynamic-national-loan-level-dataset/

  4. CFPB HISTORIC DATA (pre-2017):
     https://www.consumerfinance.gov/data-research/hmda/historic-data/

=== DATA BROWSER API PARAMETER REFERENCE ===

Geographic parameters (at least one required unless using LEI):
  states        Comma-separated state abbreviations (e.g., NY,CA,TX)
  msamds        MSA/MD codes (e.g., 35620 for New York metro)

Required:
  years         Year(s) of data (e.g., 2022 or 2018,2019)

HMDA data filter parameters:
  actions_taken      1=Originated, 2=Approved not accepted, 3=Denied,
                     4=Withdrawn, 5=Incomplete, 6=Purchased,
                     7=Preapproval denied, 8=Preapproval approved not accepted
  races              White, Black or African American, Asian,
                     American Indian or Alaska Native,
                     Native Hawaiian or Other Pacific Islander,
                     2 or more minority races, Joint,
                     Free Form Text Only, Race Not Available
  sexes              Male, Female, Joint, Sex Not Available
  ethnicities        Hispanic or Latino, Not Hispanic or Latino, Joint,
                     Ethnicity Not Available, Free Form Text Only
  loan_types         1=Conventional, 2=FHA, 3=VA, 4=USDA/FSA
  loan_purposes      1=Home Purchase, 2=Home Improvement, 31=Refinancing,
                     32=Cash-out refinancing, 4=Other, 5=Not applicable
  lien_statuses      1=First lien, 2=Subordinate lien
  construction_methods  1=Site-built, 2=Manufactured
  dwelling_categories   Single Family (1-4 Units):Site-Built,
                        Single Family (1-4 Units):Manufactured,
                        Multifamily:Site-Built, Multifamily:Manufactured
  total_units         1, 2, 3, 4, 5-24, 25-49, 50-99, 100-149, >149

Usage:
    # Method 1: Data Browser API (filtered, state-by-state)
    python download_hmda_real.py --method api --years 2022 --states NY CA TX

    # Method 2: Snapshot bulk download (full national dataset)
    python download_hmda_real.py --method snapshot --years 2022

    # Method 3: Just process existing files
    python download_hmda_real.py --skip-download --process-only

Estimated download sizes:
    Data Browser API, single state/year:    ~50-200 MB
    Data Browser API, 20 states/year:       ~1-2 GB
    Snapshot full year:                     ~2-5 GB compressed
"""

import argparse
import requests
import pandas as pd
import numpy as np
import os
import io
import time
import sys
from pathlib import Path


# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

# Data Browser API endpoint
DATA_BROWSER_API = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"

# Snapshot bulk download base URLs
# Format: https://s3.amazonaws.com/cfpb-hmda-public/prod/snapshot-data/{year}/
SNAPSHOT_BASE = "https://s3.amazonaws.com/cfpb-hmda-public/prod/snapshot-data"

# Known snapshot file patterns (these may change; check CFPB site for current)
SNAPSHOT_FILES = {
    2018: f"{SNAPSHOT_BASE}/2018/2018_public_lar_csv.zip",
    2019: f"{SNAPSHOT_BASE}/2019/2019_public_lar_csv.zip",
    2020: f"{SNAPSHOT_BASE}/2020/2020_public_lar_csv.zip",
    2021: f"{SNAPSHOT_BASE}/2021/2021_public_lar_csv.zip",
    2022: f"{SNAPSHOT_BASE}/2022/2022_public_lar_csv.zip",
    2023: f"{SNAPSHOT_BASE}/2023/2023_public_lar_csv.zip",
}

# All US states + DC
ALL_STATES = [
    'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL',
    'GA','HI','ID','IL','IN','IA','KS','KY','LA','ME',
    'MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
    'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI',
    'SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
]

# Key states for a representative sample (covers all regions, diverse demographics)
SAMPLE_STATES = [
    'NY', 'CA', 'TX', 'FL', 'IL',  # Large, diverse
    'GA', 'NC', 'SC', 'AL', 'MS',  # South / high Black population
    'OH', 'MI', 'PA', 'MD', 'VA',  # Mid-Atlantic / Midwest
    'MA', 'NJ', 'CT',              # Northeast
    'WA', 'CO', 'AZ',              # West
]


# ══════════════════════════════════════════════════════════
# METHOD 1: DATA BROWSER API DOWNLOAD
# ══════════════════════════════════════════════════════════

def download_via_api(year, states=None, output_dir='../data/raw',
                     filters=None):
    """
    Download HMDA data using the CFPB Data Browser API.

    The API requires: year + at least one geographic param + at least one
    HMDA data filter.

    Parameters
    ----------
    year : int
        HMDA reporting year (2018-2023+).
    states : list of str, optional
        State abbreviations. Defaults to SAMPLE_STATES.
    output_dir : str
        Where to save downloaded CSV chunks.
    filters : dict, optional
        Additional API filters. Defaults to fair lending analysis filters.
    """
    states = states or SAMPLE_STATES
    os.makedirs(output_dir, exist_ok=True)

    # Default filters for fair lending analysis
    if filters is None:
        filters = {
            'actions_taken': '1,3',          # Originated + Denied only
            'loan_types': '1',                # Conventional loans
            'loan_purposes': '1',             # Home purchase
            'lien_statuses': '1',             # First lien
            'dwelling_categories': 'Single Family (1-4 Units):Site-Built',
            'races': 'White,Black or African American',
        }

    total_records = 0
    successful_states = []
    failed_states = []

    print(f"\n{'='*60}")
    print(f"Downloading HMDA {year} via Data Browser API")
    print(f"Filters: {filters}")
    print(f"States: {', '.join(states)}")
    print(f"{'='*60}")

    for state in states:
        params = {
            'years': str(year),
            'states': state,
            **filters
        }

        try:
            print(f"  {state} ({year})...", end=' ', flush=True)
            resp = requests.get(DATA_BROWSER_API, params=params,
                              timeout=180, stream=True)

            if resp.status_code == 200:
                # Check if we got actual CSV data (not error page)
                content = resp.text
                if len(content) > 200 and 'activity_year' in content[:500]:
                    chunk = pd.read_csv(io.StringIO(content), low_memory=False)
                    n = len(chunk)
                    total_records += n
                    print(f"{n:,} records")

                    # Save chunk
                    outpath = os.path.join(output_dir, f'hmda_{year}_{state}.csv')
                    chunk.to_csv(outpath, index=False)
                    successful_states.append(state)
                    del chunk
                else:
                    print(f"empty or error response")
                    failed_states.append(state)
            elif resp.status_code == 400:
                print(f"bad request (check filters)")
                failed_states.append(state)
            else:
                print(f"HTTP {resp.status_code}")
                failed_states.append(state)

            time.sleep(0.5)  # Rate limiting — be respectful to CFPB servers

        except requests.exceptions.Timeout:
            print(f"timeout (try later or use snapshot)")
            failed_states.append(state)
        except Exception as e:
            print(f"error: {e}")
            failed_states.append(state)

    print(f"\n  Summary for {year}:")
    print(f"    Total records: {total_records:,}")
    print(f"    Successful states: {len(successful_states)}")
    if failed_states:
        print(f"    Failed states: {', '.join(failed_states)}")

    return total_records


# ══════════════════════════════════════════════════════════
# METHOD 2: SNAPSHOT BULK DOWNLOAD
# ══════════════════════════════════════════════════════════

def download_snapshot(year, output_dir='../data/raw'):
    """
    Download the full national HMDA snapshot dataset.

    WARNING: These files are very large (2-5 GB each).
    Make sure you have sufficient disk space.
    """
    os.makedirs(output_dir, exist_ok=True)

    url = SNAPSHOT_FILES.get(year)
    if not url:
        # Try constructing URL
        url = f"{SNAPSHOT_BASE}/{year}/{year}_public_lar_csv.zip"

    outpath = os.path.join(output_dir, f'hmda_{year}_national.zip')

    if os.path.exists(outpath):
        size_mb = os.path.getsize(outpath) / (1024 * 1024)
        print(f"  File already exists: {outpath} ({size_mb:.0f} MB)")
        return outpath

    print(f"\n{'='*60}")
    print(f"Downloading HMDA {year} National Snapshot")
    print(f"URL: {url}")
    print(f"WARNING: This file is ~2-5 GB. Ensure sufficient disk space.")
    print(f"{'='*60}")

    try:
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}. URL may have changed.")
            print(f"  Check: https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/{year}")

            # Try alternative URL patterns
            alt_urls = [
                f"{SNAPSHOT_BASE}/{year}/snapshot_data_{year}.zip",
                f"{SNAPSHOT_BASE}/{year}/{year}_public_lar.zip",
            ]
            for alt in alt_urls:
                print(f"  Trying: {alt}")
                resp = requests.get(alt, stream=True, timeout=30)
                if resp.status_code == 200:
                    url = alt
                    break
            else:
                print(f"\n  Could not find snapshot URL for {year}.")
                print(f"  Please download manually from:")
                print(f"  https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/{year}")
                return None

        # Stream download with progress
        total_size = int(resp.headers.get('content-length', 0))
        downloaded = 0

        with open(outpath, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Downloading: {mb:.0f} MB ({pct:.1f}%)", end='', flush=True)

        print(f"\n  Saved: {outpath}")
        return outpath

    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Please download manually from the CFPB Data Browser.")
        return None


# ══════════════════════════════════════════════════════════
# DATA PROCESSING
# ══════════════════════════════════════════════════════════

def process_hmda_data(input_dir, output_path, years):
    """
    Process raw HMDA downloads into analysis-ready format.

    Creates the variables needed by CausalFair:
    - A: Protected attribute (0=White, 1=Black)
    - Y: Denial outcome (0=originated, 1=denied)
    - Mediators M: dti_numeric, ltv, income_quintile, credit_score_quintile
    - Covariates W: tract_income_pct, tract_minority_pct, year, lender_type
    """
    print(f"\n{'='*60}")
    print("Processing HMDA data for CausalFair analysis")
    print(f"{'='*60}")

    all_chunks = []

    for year in years:
        # Look for per-state files from API download
        pattern = f'hmda_{year}_*.csv'
        files = sorted(Path(input_dir).glob(pattern))

        for f in files:
            if f.name.endswith('_national.zip'):
                continue  # Skip zip files
            try:
                chunk = pd.read_csv(f, low_memory=False)
                all_chunks.append(chunk)
                print(f"  Loaded {f.name}: {len(chunk):,} records")
            except Exception as e:
                print(f"  Warning: Could not read {f.name}: {e}")

    if not all_chunks:
        print("\nERROR: No data files found!")
        print("Options:")
        print("  1. Run with --method api to download via Data Browser")
        print("  2. Run with --method snapshot for full national dataset")
        print("  3. Download manually from https://ffiec.cfpb.gov/data-browser/")
        return None

    df = pd.concat(all_chunks, ignore_index=True)
    print(f"\nTotal raw records: {len(df):,}")
    print(f"Columns available: {list(df.columns)[:10]}...")

    # ── A: Protected attribute (race) ────────────────────
    race_map = {'White': 0, 'Black or African American': 1}
    if 'derived_race' in df.columns:
        df = df[df['derived_race'].isin(race_map.keys())].copy()
        df['A'] = df['derived_race'].map(race_map)
    elif 'applicant_race-1' in df.columns:
        # Fallback to raw race field
        df = df[df['applicant_race-1'].isin([5, 3])].copy()  # 5=White, 3=Black
        df['A'] = (df['applicant_race-1'] == 3).astype(int)
    else:
        print("  ERROR: No race column found!")
        return None

    # ── Y: Denial outcome ────────────────────────────────
    df['Y'] = (df['action_taken'] == 3).astype(int)
    df = df[df['action_taken'].isin([1, 3])].copy()

    # ── Mediator: DTI ────────────────────────────────────
    if 'debt_to_income_ratio' in df.columns:
        dti_map = {
            '<20%': 15, '20%-<30%': 25, '30%-<36%': 33,
            '36': 36, '37': 37, '38': 38, '39': 39, '40': 40,
            '41': 41, '42': 42, '43': 43, '44': 44, '45': 45,
            '46': 46, '47': 47, '48': 48, '49': 49,
            '50%-60%': 55, '>60%': 65,
        }
        df['dti_numeric'] = df['debt_to_income_ratio'].map(dti_map)
        for val in df['debt_to_income_ratio'].unique():
            if val not in dti_map and pd.notna(val):
                try:
                    df.loc[df['debt_to_income_ratio'] == val, 'dti_numeric'] = float(str(val).replace('%',''))
                except (ValueError, AttributeError):
                    pass

    # ── Mediator: LTV ────────────────────────────────────
    if 'combined_loan_to_value_ratio' in df.columns:
        df['ltv'] = pd.to_numeric(df['combined_loan_to_value_ratio'], errors='coerce')
    elif 'loan_amount' in df.columns and 'property_value' in df.columns:
        pv = pd.to_numeric(df['property_value'], errors='coerce')
        la = pd.to_numeric(df['loan_amount'], errors='coerce')
        df['ltv'] = (la / pv * 100).clip(0, 200)
    df['loan_amount_num'] = pd.to_numeric(df.get('loan_amount', pd.Series()), errors='coerce')
    df['property_value_num'] = pd.to_numeric(df.get('property_value', pd.Series()), errors='coerce')

    # ── Mediator: Income ─────────────────────────────────
    df['income_num'] = pd.to_numeric(df.get('income', pd.Series()), errors='coerce')

    # ── Quintiles ────────────────────────────────────────
    for col, new_col in [('income_num', 'income_quintile'),
                          ('dti_numeric', 'dti_quintile')]:
        if col in df.columns:
            valid = df[col].notna()
            try:
                df.loc[valid, new_col] = pd.qcut(
                    df.loc[valid, col], q=5, labels=[1,2,3,4,5], duplicates='drop'
                ).astype(float)
            except ValueError:
                pass

    # ── Mediator: Credit score quintile ──────────────────
    # HMDA does not report credit scores directly.
    # Following Bartlett et al. (2022), we impute from denial reason codes.
    if 'denial_reason-1' in df.columns:
        credit_proxy = np.zeros(len(df))
        for col in ['denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4']:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce')
                credit_proxy += (vals == 3).astype(float) * -2    # Credit history
                credit_proxy += (vals == 9).astype(float) * -1.5  # Credit app incomplete
                credit_proxy += (vals == 1).astype(float) * -0.5  # DTI ratio
                credit_proxy += (vals == 7).astype(float) * -0.5  # Insufficient cash
        # For originated loans (Y=0), assume credit was adequate
        credit_proxy[df['Y'] == 0] = np.random.normal(1, 0.5, (df['Y'] == 0).sum())
        df['credit_score_proxy'] = credit_proxy
        try:
            valid = df['credit_score_proxy'].notna()
            df.loc[valid, 'credit_score_quintile'] = pd.qcut(
                df.loc[valid, 'credit_score_proxy'], q=5,
                labels=[1,2,3,4,5], duplicates='drop'
            ).astype(float)
        except ValueError:
            df['credit_score_quintile'] = 3.0  # fallback

    # ── Pre-treatment covariates (W) ─────────────────────
    df['tract_income_pct'] = pd.to_numeric(
        df.get('tract_to_msa_income_percentage', pd.Series()), errors='coerce'
    )
    df['tract_minority_pct'] = pd.to_numeric(
        df.get('tract_minority_population_percent', pd.Series()), errors='coerce'
    )
    df['year'] = pd.to_numeric(df.get('activity_year', pd.Series()), errors='coerce')

    # Lender type from agency code
    if 'agency_code' in df.columns:
        agency = pd.to_numeric(df['agency_code'], errors='coerce')
        df['lender_type'] = agency.map({
            1: 0, 2: 0, 3: 0,   # OCC, FRS, FDIC → bank
            5: 1,                 # NCUA → credit union
            7: 2, 9: 2,          # HUD, CFPB → nonbank/mortgage company
        }).fillna(0).astype(int)
    else:
        df['lender_type'] = 0

    # ── Select and clean ─────────────────────────────────
    keep = ['A', 'Y', 'dti_numeric', 'ltv', 'income_num',
            'income_quintile', 'dti_quintile', 'credit_score_quintile',
            'tract_income_pct', 'tract_minority_pct',
            'year', 'lender_type',
            'loan_amount_num', 'property_value_num']
    available = [c for c in keep if c in df.columns]
    df_clean = df[available].dropna(subset=['A', 'Y', 'dti_numeric'])

    # ── Summary ──────────────────────────────────────────
    white = df_clean[df_clean['A'] == 0]
    black = df_clean[df_clean['A'] == 1]

    print(f"\n{'='*60}")
    print(f"PROCESSED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total records:          {len(df_clean):,}")
    print(f"  White (A=0):          {len(white):,}")
    print(f"  Black (A=1):          {len(black):,}")
    print(f"  Originated (Y=0):     {(df_clean['Y']==0).sum():,}")
    print(f"  Denied (Y=1):         {(df_clean['Y']==1).sum():,}")
    print(f"\n  White denial rate:    {white['Y'].mean():.1%}")
    print(f"  Black denial rate:    {black['Y'].mean():.1%}")
    print(f"  Total effect:         {(black['Y'].mean()-white['Y'].mean())*100:+.1f} pp")
    print(f"\n  White mean DTI:       {white['dti_numeric'].mean():.1f}%")
    print(f"  Black mean DTI:       {black['dti_numeric'].mean():.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df_clean


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download HMDA data for CausalFair analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 2022 data for key states via API
  python download_hmda_real.py --method api --years 2022

  # Download full national 2022 snapshot
  python download_hmda_real.py --method snapshot --years 2022

  # Download specific states
  python download_hmda_real.py --method api --years 2022 --states NY CA TX GA

  # Process already-downloaded files
  python download_hmda_real.py --skip-download --process-only

Data sources:
  Data Browser:  https://ffiec.cfpb.gov/data-browser/
  API Docs:      https://ffiec.cfpb.gov/documentation/api/data-browser/
  Snapshots:     https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/
  Historic data: https://www.consumerfinance.gov/data-research/hmda/historic-data/
        """
    )
    parser.add_argument('--method', choices=['api', 'snapshot'],
                        default='api',
                        help='Download method: api (filtered) or snapshot (full national)')
    parser.add_argument('--years', nargs='+', type=int,
                        default=[2022],
                        help='Years to download (default: 2022)')
    parser.add_argument('--states', nargs='+', type=str,
                        default=None,
                        help='State abbreviations (default: 20 representative states)')
    parser.add_argument('--all-states', action='store_true',
                        help='Download all 50 states + DC')
    parser.add_argument('--output-dir', type=str, default='../data/raw',
                        help='Directory for raw downloads')
    parser.add_argument('--analysis-file', type=str, default='../data/hmda_analysis.csv',
                        help='Path for processed analysis file')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, just process existing files')
    parser.add_argument('--process-only', action='store_true',
                        help='Same as --skip-download')

    args = parser.parse_args()

    states = ALL_STATES if args.all_states else args.states

    if not (args.skip_download or args.process_only):
        for year in args.years:
            if args.method == 'api':
                download_via_api(year, states=states, output_dir=args.output_dir)
            elif args.method == 'snapshot':
                download_snapshot(year, output_dir=args.output_dir)

    # Process
    process_hmda_data(args.output_dir, args.analysis_file, args.years)

    print(f"""
{'='*60}
HMDA DATA READY FOR CAUSALFAIR ANALYSIS
{'='*60}

Next steps:
  1. Run the analysis:
     cd code && python run_analysis.py

  2. Or use the implementation guide:
     cd notebooks && python implementation_guide.py

  3. For your own institutional data, format it like
     hmda_analysis.csv with columns: A, Y, dti_numeric, ltv,
     income_num, income_quintile, credit_score_quintile,
     tract_income_pct, tract_minority_pct, year, lender_type

Reference:
  HMDA Data Browser FAQ: https://ffiec.cfpb.gov/documentation/tools/data-browser/data-browser-faq
  Data Browser API:      https://ffiec.cfpb.gov/documentation/api/data-browser/
  LAR Data Fields:       https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields
""")
