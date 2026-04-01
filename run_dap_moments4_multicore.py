#!/usr/bin/env python3
"""
Run the MaNGA DAP with moments=4 (V, sigma, h3, h4) stellar kinematics on
LOGCUBE files found in the current working directory. Emission-line fitting
and spectral indices are skipped entirely.

Usage:
    cd /path/to/your/LOGCUBE/directory
    python run_dap_moments4.py

Expected LOGCUBE directory structure:
    ./FindingCRDs_Pipeline/CUBES/

Output MAPS files are written to:
    ./MAPS_moments4/manga-[plate]-[ifu]-MAPS-VOR10-MILESHC-MOMENTS4.fits.gz

Fitting files are written to:
    ./MAPS_moments4/fitting_files/[plate]/[ifu]/

Requirements:
    - mangadap
    - The DRPall file, used to get each galaxy's redshift. Set DRPALL_FILE
      below, or set the MANGA_SPECTRO_REDUX and MANGADRP_VER environment
      variables so mangadap can find it automatically.
"""

import os
import glob
import time
import traceback
import warnings
import numpy as np
import astropy.constants
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from astropy.io import fits
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from mangadap.datacube import MaNGADataCube
from mangadap.proc.reductionassessments import ReductionAssessment, ReductionAssessmentDef
from mangadap.proc.spatiallybinnedspectra import SpatiallyBinnedSpectra, SpatiallyBinnedSpectraDef
from mangadap.proc.stellarcontinuummodel import StellarContinuumModel, StellarContinuumModelDef, StellarContinuumModelBitMask
from mangadap.dapfits import construct_maps_file
from mangadap.proc.ppxffit import PPXFFitPar, PPXFFit
from mangadap.util.pixelmask import SpectralPixelMask
from mangadap import __version__

dapver=__version__
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PATCH DAP: disable dispersion correction (breaks for moments > 2)
# -----------------------------------------------------------------------------
import mangadap.proc.ppxffit as ppxffit_mod
import numpy as np

def _patched_fit_dispersion_correction(self, templates, templates_rfft, result,
                                       baseline_dispersion=None):
    return np.zeros(self.nobj), np.zeros(self.nobj, dtype=bool)

ppxffit_mod.PPXFFit._fit_dispersion_correction = _patched_fit_dispersion_correction
# -----------------------------------------------------------------------------

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# User needs to set number of threads and processes to be 1. This code is to
# be used on a large number of galaxies, thus we are parallelizing across
# galaxies. In the terminal session before executing, run:
# "export OMP_NUM_THREADS=1" and "export MKL_NUM_THREADS=1".

# Root directory containing the LOGCUBE files.
# The script will cd here and scan for files.
LOGCUBE_DIR = os.path.join(os.getcwd(), 'CUBES')

# Where to write the output MAPS files.
OUTPUT_DIR = os.path.join(os.getcwd(), 'MAPS_moments4')

# Path to the DRPall file (needed to look up galaxy redshifts).
DRPALL_FILE = Path(os.getcwd()) / "drpall-v3_1_1.fits"

# =============================================================================

def run_with_logging(func, log_file, *args, **kwargs):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        with redirect_stdout(f), redirect_stderr(f):
            return func(*args, **kwargs)

def apply_dap_monkeypatch():
    import mangadap.dapfits as dapfits_mod

    if hasattr(dapfits_mod.finalize_dap_primary_header, "_patched"):
        return  # already patched

    _orig_finalize = dapfits_mod.finalize_dap_primary_header

    def _safe_finalize(*args, **kwargs):
        # The DAP passes `meta` as the third positional argument to
        # finalize_dap_primary_header. Inside that function, line 2306 does:
        #   if flag in dapqual.keys()
        # where dapqual comes from meta["dapqual"]. For some galaxies the DAP
        # sets this to a numpy.int32 (e.g. 0) rather than a dict, causing:
        #   AttributeError: "numpy.int32" object has no attribute "keys"
        # We fix this by coercing meta["dapqual"] to a dict before calling
        # the original function.
        args = list(args)
        if len(args) >= 3 and isinstance(args[2], dict):
            meta = dict(args[2])          # shallow copy so we do not mutate caller
            if "dapqual" in meta and not isinstance(meta["dapqual"], dict):
                meta["dapqual"] = {}
            args[2] = meta
        args = tuple(args)
        try:
            return _orig_finalize(*args, **kwargs)
        except AttributeError as e:
            # Safety net for any remaining dapqual-related AttributeErrors
            if "keys" in str(e) or "dapqual" in str(e):
                return args[0]            # return prihdr unchanged
            raise

    _safe_finalize._patched = True
    dapfits_mod.finalize_dap_primary_header = _safe_finalize

def get_drpall_row(plate, ifu, drpall_file):
    """
    Find the row in the drpall file corresponding to the current galaxy, so
    that its metadata can be accessed. 
    """
    from astropy.io import fits
    with fits.open(drpall_file) as hdu:
        data = hdu[1].data
        indx = data['PLATEIFU'] == f'{plate}-{ifu}'
        if not np.any(indx):
            raise ValueError(f'Could not find {plate}-{ifu} in DRPall file.')
        return data[indx][0]

def _sanitize_dapqual(obj):
    if hasattr(obj, 'dapqual') and not isinstance(obj.dapqual, dict):
        obj.dapqual = {}


def run_one_galaxy(plate, ifu, logcube_path, output_dir, drpall_file=None):
    """
    Run the DAP stellar-kinematics-only pipeline with moments=4 on one galaxy
    and write the output MAPS file.
    """
    apply_dap_monkeypatch()
    print(f"=== START {plate}-{ifu} ===")
    
    # Give each galaxy its own output sandbox so no issues with multicore
    galaxy_output_dir = os.path.join(output_dir, 'fitting_files', str(plate), str(ifu))
    if not os.path.exists(galaxy_output_dir):
        os.makedirs(galaxy_output_dir)
    
    #print(f'\n  Loading cube: {logcube_path}')

    # Load the datacube directly from the file path
    cube = MaNGADataCube(logcube_path)

    # Get metadata from cube
    row = get_drpall_row(plate, ifu, drpall_file)
    cube.meta['z'] = float(row['NSA_Z'])
    cube.meta['pa']   = float(row['NSA_ELPETRO_PHI'])
    cube.meta['ell']  = 1.0 - float(row['NSA_ELPETRO_BA'])
    cube.meta['reff'] = float(row['NSA_ELPETRO_TH50_R'])
    try:
        cube.meta['vdisp'] = float(row['NSA_VDISP']) if row['NSA_VDISP'] > 0 else 100.0
    except Exception:
        cube.meta['vdisp'] = 100.0
        
    #print(f"  Redshift: {cube.meta['z']:.5f}")
    
    metadata = cube.meta

    # S/N assessment (g-band)
    #print('  Running ReductionAssessment...')
    method = ReductionAssessmentDef(key='SNRG', overwrite=True,)
    rdxqa = ReductionAssessment(method,
                                cube,
                                output_path=galaxy_output_dir,
                                quiet=True)

    # Voronoi binning to S/N ~ 10 (matching the standard VOR10 scheme)
    #print('  Voronoi binning (VOR10)...')
    binning_method = SpatiallyBinnedSpectraDef.from_dict({
    'key': 'VOR10',
    'spatial_method': 'voronoi'})
    binned_spectra = SpatiallyBinnedSpectra(binning_method,
                                            cube,
                                            rdxqa,
                                            output_path=galaxy_output_dir,
                                            quiet=True,
                                            overwrite=True)

    # Stellar kinematics fit with moments=4
    #print('  Fitting stellar kinematics (moments=4)...')
    sc_def = StellarContinuumModelDef.from_dict({
        'key': 'MILESHC',
        'minimum_snr': 1.0,
        'fit_method': 'ppxf',
        'fit': {},
        'overwrite': True,
        'redo_postmodeling': False
    })

    wave = cube.wave
    pixelmask = SpectralPixelMask(waverange=[wave.min(), wave.max()])
    
    sc_def['fitpar']['pixelmask'] = pixelmask
    sc_def['fitpar']['moments'] = 4
    
    stellar_continuum = StellarContinuumModel(
        method=sc_def,
        binned_spectra=binned_spectra,
        guess_vel=cube.meta['z'] * astropy.constants.c.to('km/s').value,
        guess_sig=cube.meta['vdisp'],
        output_path=galaxy_output_dir,
        quiet=True
    )

    # Build output directory and write MAPS file.
    # Emission-line and spectral-index objects are passed as None to skip
    # those steps, as documented in the DAP CHANGES.md.
    #galaxy_output_dir = os.path.join(output_dir, str(plate), str(ifu))
    #os.makedirs(galaxy_output_dir, exist_ok=True)

    maps_file = os.path.join(
        output_dir,
        f'manga-{plate}-{ifu}-MAPS-VOR10-MILESHC-None.fits.gz'
    )

    # Sanitize ALL dapqual sources
    for obj in [rdxqa, binned_spectra, stellar_continuum]:
        _sanitize_dapqual(obj)

    # Also sanitize cube metadata
    if isinstance(cube.meta, dict):
        if 'dapqual' not in cube.meta or not isinstance(cube.meta['dapqual'], dict):
            cube.meta['dapqual'] = {}

    #print(f'  Writing MAPS file: {maps_file}')
    construct_maps_file(
        cube,
        rdxqa=rdxqa,
        binned_spectra=binned_spectra,
        stellar_continuum=stellar_continuum,
        emission_line_moments=None,     # Skip emission lines
        emission_line_model=None,       # Skip emission lines
        spectral_indices=None,          # Skip spectral indices
        output_file=maps_file,
        redshift=metadata['z'],
        overwrite=True
    )

    print(f"=== DONE {plate}-{ifu} ===")
    return maps_file


def run_one_galaxy_safe(args):
    plate, ifu, logcube_path, output_dir, drpall_file = args

    galaxy_output_dir = os.path.join(output_dir, 'fitting_files', str(plate), str(ifu))
    os.makedirs(galaxy_output_dir, exist_ok=True)

    log_file = os.path.join(galaxy_output_dir, f"{plate}-{ifu}.log")

    try:
        return run_with_logging(
            run_one_galaxy,
            log_file,
            plate, ifu, logcube_path, output_dir, drpall_file
        )
    except Exception as e:
        with open(log_file, "a") as f:
            f.write("\n\n=== ERROR TRACEBACK ===\n")
            f.write(traceback.format_exc())

        print(f"\nERROR processing {plate}-{ifu} (see {log_file})")
        return None


def main():
    t0 = time.perf_counter()

    # Scan for LOGCUBEs
    pattern = os.path.join(LOGCUBE_DIR, '**', '*-LOGCUBE.fits.gz')
    logcube_files = sorted(glob.glob(pattern, recursive=True))

    if len(logcube_files) == 0:
        print(f'No LOGCUBE files found under {LOGCUBE_DIR}')
        return

    print(f'Found {len(logcube_files)} LOGCUBE file(s).')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build task list
    tasks = []
    for logcube_path in logcube_files:
        filename = os.path.basename(logcube_path)
        parts = filename.split('-')

        try:
            plate = parts[1]
            ifu   = parts[2]
        except IndexError:
            print(f'Skipping unrecognised filename: {filename}')
            continue

        tasks.append((plate, ifu, logcube_path, OUTPUT_DIR, DRPALL_FILE))

    # Auto-detect CPU cores and leave some free for system
    total_cores = cpu_count()

    if total_cores <= 2:
        nproc = 1
    elif total_cores <= 4:
        nproc = total_cores - 1
    else:
        nproc = total_cores - 2   # leave 2 cores free on larger systems

    # Safety check
    nproc = max(1, nproc)

    print(f'\nDetected {total_cores} CPU cores.')
    print(f'Using {nproc} parallel processes (reserving {total_cores - nproc} cores for system).\n')

    # Run DAP and use progress bar
    with Pool(processes=nproc) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(run_one_galaxy_safe, tasks), total=len(tasks)):
            results.append(result)

    succeeded = []
    failed = []

    # Sort results
    for task, result in zip(tasks, results):
        plate, ifu = task[0], task[1]
        if result is None:
            failed.append(f'{plate}-{ifu}')
        else:
            succeeded.append(f'{plate}-{ifu}')

    # Summary
    elapsed = time.perf_counter() - t0
    print(f'\n{"="*50}')
    print(f'Finished in {elapsed/60:.1f} min.')
    print(f'Succeeded: {len(succeeded)}')
    print(f'Failed:    {len(failed)}')

    if failed:
        print('Failed galaxies:')
        for f in failed:
            print(f'  {f}')

    # Save logs
    if succeeded:
        np.savetxt(os.path.join(OUTPUT_DIR, 'succeeded.txt'), succeeded, fmt='%s')
    if failed:
        np.savetxt(os.path.join(OUTPUT_DIR, 'failed.txt'), failed, fmt='%s')


if __name__ == '__main__':
    main()
