#!/usr/bin/env python
"""
#############################################################################

Copyright (C) 2005-2023, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

If you have found this software useful for your research,
we would appreciate an acknowledgement to use of the
`Method described in Appendix C of Krajnovic et al. (2006)'.
https://ui.adsabs.harvard.edu/abs/2006MNRAS.366..787K

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.
In particular, redistribution of the code is not allowed.

#############################################################################

 MODIFICATION HISTORY:
   V1.0.0 -- Created by Michele Cappellari, Leiden, 30 May 2005
   V1.1.0 -- Written documentation. MC, Oxford, 9 October 2007
   V1.1.1 -- Corrected handling of NSTEPS keyword. Thanks to Roland Jesseit.
       MC, Oxford, 17 October 2007
   V1.1.2 -- Force error to be larger than 1/2 of the angular step size.
       MC, Oxford, 19 October 2007
   V1.1.3 -- Determine plotting ranges from velSym instead of vel.
       Thanks to Anne-Marie Weijmans. Leiden, 25 May 2008
   V1.1.4 -- Clarified that systemic velocity has to be subtracted from VEL.
       MC, Oxford, 31 March 2009
   V1.1.5 -- Overplot best PA on data. Some changes to the documentation.
       MC, Oxford, 14 October 2009
   V1.2.0 -- Includes error in chi^2 in the determination of angErr.
       Thanks to Davor Krajnovic for reporting problems.
       MC, Oxford, 23 March 2010
   V1.3.0 -- The program is two orders of magnitude faster, thanks to a
       new cap_symmetrize_velfield routine. MC, Oxford, 8 October 2013
   V1.3.1 -- Uses CAP_RANGE routine to avoid potential naming conflicts.
       Uses TOLERANCE keyword of TRIANGULATE to try to avoid IDL error
       "TRIANGULATE: Points are co-linear, no solution."
       MC, Oxford, 2 December 2013
   V2.0.0 -- Translated from IDL into Python. MC, Oxford, 10 April 2014
   V2.0.1 -- Support both legacy Python 2.7 and Python 3. MC, Oxford, 25 May 2014
   V2.0.2 -- Fixed possible program stop. MC, Sydney, 2 February 2015
   V2.0.3 -- Changed color of plotted lines. Check matching of input sizes.
       MC, Oxford, 7 September 2015
   V2.0.4 -- Use np.percentile instead of deprecated Scipy version.
       MC, Oxford, 17 March 2017
   V2.0.5: Changed imports for plotbin as package. MC, Oxford, 17 April 2018
   V2.0.6: Dropped Python 2.7 support. MC, Oxford 12 May 2018
   V2.0.7: Some documentation updates. MC, Oxford, 23 March 2021
   Vx.x.x: Additional changes are documented in the CHANGELOG of the PaFit package.

"""

import numpy as np
import matplotlib.pyplot as plt

from plotbin.symmetrize_velfield import symmetrize_velfield
from plotbin.plot_velfield import plot_velfield

##############################################################################

def fit_kinematic_pa(x, y, vel, debug=False, nsteps=361,
                     quiet=False, plot=True, dvel=10, **kwargs):
    """
    fit_kinematic_pa
    ================

    Purpose
    -------

    Determine the global kinematic position angle of a galaxy from
    integral-field kinematics of the stars or gas with the method described in
    Appendix C of `Krajnovic et al. (2006) <https://ui.adsabs.harvard.edu/abs/2006MNRAS.366..787K>`_

    Calling Sequence
    ----------------

    .. code-block:: python

        angBest, angErr, vSyst = fit_kinematic_pa(
            x, y, vel, debug=False, nsteps=361, quiet=False, plot=True, dvel=10)

    Input Parameters
    ----------------

    xbin, ybin:
        Vectors with the coordinates of the bins (or pixels) measured from
        the centre of rotation (typically the galaxy centre).

        IMPORTANT: The routine will not give meaningful output unless
        ``(x, y) = (0, 0)`` is an estimate of the centre of rotation.
    vel:
        Measured velocity at the position ``(xbin, ybin)``.

        IMPORTANT: An estimate of the systemic velocity has to be already
        subtracted from this velocity [e.g. ``vel_corr = vel - np.median(vel)``].
        The routine will then provide in the output ``velsyst`` a correction
        to be added to the adopted systemic velocity.
    dvel:
        Scalar with the typical uncertainty in the velocity ``vel`` or vector
        with the uncertainty of each bin ``(xbin, ybin)``.

    Optional Keywords
    -----------------

    nsteps:
        Number of steps along which the angle is sampled.
        Default is 361 steps which gives a 0.5 degr accuracy.
        Decrease this number to limit computation time during testing.

    Output Parameters
    -----------------

    anglebest:
        Kinematical PA. Note that this is the angle along which
        ``|vel|`` is maximum (note modulus!). If one reverses the sense of
        rotation in a galaxy ``anglebest`` does not change. The sense of
        rotation can be trivially determined by looking at the map of Vel.
    angleerror:
        Corresponding error to assign to ``anglebest``.
    velsyst:
        Best-fitting correction to the adopted systemic velocity for the galaxy.

        If the median was subtracted to the input velocity ``vel`` before
        the ``pa`` fit, then the corrected systemic velocity will be

    ###########################################################################
    """
    x, y, vel = map(np.ravel, [x, y, vel])

    assert x.size == y.size == vel.size, 'Input vectors (x, y, vel) must have the same size'

    nbins = x.size
    n = nsteps
    angles = np.linspace(0, 180, n) # 0.5 degrees steps by default
    chi2 = np.empty_like(angles)
    for j, ang in enumerate(angles):
        velSym = symmetrize_velfield(x, y, vel, sym=1, pa=ang)
        chi2[j] = np.sum(((vel - velSym)/dvel)**2)
        if debug:
            print('Ang: %5.1f, chi2/DOF: %#.4g' % (ang, chi2[j]/nbins))
            plt.cla()
            plot_velfield(x, y, velSym, **kwargs)
            plt.pause(0.01)
    k = np.argmin(chi2)
    angBest = angles[k]

    # Compute fit at the best position
    #
    velSym = symmetrize_velfield(x, y, vel, sym=1, pa=angBest)
    if angBest < 0:
        angBest += 180

    # 3sigma confidence limit, including error on chi^2
    #
    f = chi2 - chi2[k] <= 9 + 3*np.sqrt(2*nbins)
    minErr = max(0.5, (angles[1] - angles[0])/2.0)
    if f.sum() > 1:
        angErr = (np.max(angles[f]) - np.min(angles[f]))/2.0
        if angErr >= 45:
            good = np.degrees(np.arctan(np.tan(np.radians(angles[f]))))
            angErr = (np.max(good) - np.min(good))/2.0
        angErr = max(angErr, minErr)
    else:
        angErr = minErr

    vSyst = np.median(vel - velSym)

    if not quiet:
        print('  Kin PA: %5.1f' % angBest, ' +/- %5.1f' % angErr, ' (3*sigma error)')
        print('Velocity Offset: %.2f' % vSyst)

    # Plot results
    #
    if plot:

        mn, mx = np.percentile(velSym, [2.5, 97.5])
        mx = min(mx, -mn)
        plt.subplot(121)
        plot_velfield(x, y, velSym, vmin=-mx, vmax=mx, **kwargs)
        plt.title('Symmetrized')

        plt.subplot(122)
        plot_velfield(x, y, vel - vSyst, vmin=-mx, vmax=mx, **kwargs)
        plt.title('Data and best PA')
        rad = np.sqrt(np.max(x**2 + y**2))
        ang = [0,np.pi] + np.radians(angBest)
        plt.plot(rad*np.cos(ang), rad*np.sin(ang), 'k--', linewidth=3) # Zero-velocity line
        plt.plot(-rad*np.sin(ang), rad*np.cos(ang), color="limegreen", linewidth=3) # Major axis PA

    return angBest, angErr, vSyst

##############################################################################

def pafit_example():
    """Usage example for the pafit package"""

    xbin, ybin = np.random.uniform(low=[-30, -20], high=[30, 20], size=(100, 2)).T
    inc = 60.                       # assumed galaxy inclination
    r = np.sqrt(xbin**2 + (ybin/np.cos(np.radians(inc)))**2) # Radius in the plane of the disk
    a = 40                          # Scale length in arcsec
    vr = 2000*np.sqrt(r)/(r + a)      # Assumed velocity profile
    vel = vr * np.sin(np.radians(inc))*xbin/r # Projected velocity field

    # The above lines are only used to produce some simulated input.
    # With real data, the user has to replace the three vectors (xbin, ybin, vel)
    # with their corresponding observed quantities for real galaxies.

    # Subtract an initial estimate of the systemic velocity
    vel_corr = vel - np.median(vel)

    plt.clf()
    fit_kinematic_pa(xbin, ybin, vel_corr, debug=True, nsteps=30)
    plt.pause(1)

##############################################################################

if __name__ == '__main__':

    pafit_example()
#