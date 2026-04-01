#!/usr/bin/env python3
"""
Master pipeline: download → DAP (moments=4) → CRD finder.

Runs the following three steps back-to-back:
  1. download_cubes.py       — downloads all DR17 LOGCUBE + LOGRSS files into ./CUBES/
  2. run_dap_moments4_multicore.py — runs the MaNGA DAP (moments=4) on all LOGCUBEs,
                                     writing MAPS files into ./MAPS_moments4/
  3. FindingCRDs_NEW.py      — runs the CRD finder on all MAPS files,
                               writing results into ./FindingCRDs_Results/

Usage:
    cd /path/to/working/directory
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    python run_pipeline.py

Directory layout:
    ./drpall-v3_1_1.fits          — DRPall catalogue (downloaded in step 1)
    ./CUBES/                      — LOGCUBE and LOGRSS files (step 1 output)
    ./MAPS_moments4/              — MAPS files (step 2 output / step 3 input)
    ./FindingCRDs_Results/        — CRD classification results (step 3 output)
        analyzed_galaxies.txt
        no_CRD/no_CRD.txt
        maybe_CRD/maybe_CRD.txt
        maybe_CRD/[plate]-[ifu].txt
        problems.txt
"""

import time
import traceback

# =============================================================================
# STEP 1 — Download
# =============================================================================
def run_step1():
    print('\n' + '='*60)
    print('STEP 1 — DOWNLOADING LOGCUBE AND LOGRSS FILES')
    print('='*60 + '\n')
    import download_cubes
    download_cubes.main()


# =============================================================================
# STEP 2 — DAP (moments=4)
# =============================================================================
def run_step2():
    print('\n' + '='*60)
    print('STEP 2 — DAP STELLAR KINEMATICS (moments=4)')
    print('='*60 + '\n')
    import run_dap_moments4_multicore
    run_dap_moments4_multicore.main()


# =============================================================================
# STEP 3 — CRD Finder
# =============================================================================
def run_step3():
    print('\n' + '='*60)
    print('STEP 3 — CRD FINDER')
    print('='*60 + '\n')
    import FindingCRDs_NEW
    FindingCRDs_NEW.do_the_thing()


# =============================================================================
# MASTER
# =============================================================================
if __name__ == '__main__':
    t0 = time.perf_counter()

    steps = [
        #('Download',       run_step1),
        #('DAP moments=4',  run_step2),
        ('CRD Finder',     run_step3),
    ]

    for name, fn in steps:
        t_step = time.perf_counter()
        try:
            fn()
            elapsed = (time.perf_counter() - t_step) / 60
            print(f'\n✓ {name} completed in {elapsed:.1f} min.')
        except Exception:
            print(f'\n✗ {name} failed with an unhandled exception:')
            traceback.print_exc()
            print('\nPipeline aborted.')
            break

    total = (time.perf_counter() - t0) / 60
    print(f'\n{"="*60}')
    print(f'Pipeline finished. Total time: {total:.1f} min.')
