#!/usr/bin/env python
# coding: utf-8

"""
Counter-rotating disk galaxy finder using MaNGA DAP MAPS files.

If running on your device, make sure to change: 
1 - the "MAPS_DIR" to where all of your MAPS files are stored.
    1.5 - Change the analysis type of the MAPS in "hdu" on line 63 if using something different.
2 - the "OUTPUT_DIR" to where you want the output files to be stored.

The output files are the following:
1 - "analyzed_galaxies.txt" - full list of galaxies that have been analyzed.
2 - "no_CRD.txt" - list of galaxies that do not have counter-rotating properties
3 - "maybe_CRD.txt" - list of galaxies that have signs of counter-rotation.
4 - "problems.txt" - list of galaxies that could not be analyzed.

TO RUN:
First, cd into the directory with all of the MAPS files (should be the "MAPS_DIR"), then execute the program. 
It will automatically read all files that end with "fits.gz" and assume these are the MAPS files to be 
analyzed.

The code is currently written to run with SDSS-IV MaNGA data, thus the extensions to get the stellar velocity,
stellar velocity dispersion, bin snr, and bin flux are all tailored to those extensions. If using a different
IFU survey, those will have to be changed.
"""

import os
import glob
import numpy as np
import warnings
from astropy.io import fits
from mangadap.dapfits import DAPMapsBitMask
from mangadap.util.fileio import channel_dictionary
from fit_kinematic_pa import fit_kinematic_pa
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# =============================================================================
# USER CONFIGURATION
# The script uses the current working directory for both input and output.
# Simply cd into your MAPS directory and run the script — no path changes needed.
# Expected structure: MAPS_DIR/manga-[plate]-[ifu]-MAPS-VOR10-MILESHC-None.fits.gz
# =============================================================================

MAPS_DIR = os.path.join(os.getcwd(), 'MAPS_moments4')
OUTPUT_DIR = os.path.join(os.getcwd(), 'FindingCRDs_Results')

# =============================================================================


def find_rotation_angle(plate, ifu, original_center):
    # Define all the needed lists and variables
    x_vector_list = []
    y_vector_list = []
    new_x_vector_list = []
    new_y_vector_list = []
    corresponding_velocity = []
    corresponding_sigma = []
    new_corresponding_velocity = []
    maskboollistVel = []
    maskboollistSig = []
    maskedVelocityList = []
    maskedSigmaList = []
    shapedVelocityList = []
    shapedSigmaList = []
    allVelocityList = []
    allSigmaList = []
    newVelocityList = []
    newSigmaList = []

    # Read the MAPS file
    hdu = fits.open(os.path.join(MAPS_DIR, 'manga-'+plate+'-'+ifu+'-MAPS-VOR10-MILESHC-None.fits.gz'))

    # Save the masked H-alpha velocity and velocity variance
    emlc = channel_dictionary(hdu, 'EMLINE_GFLUX')
    bm = DAPMapsBitMask()

    # Get the masked lists for stellar velocity, velocity dispersion, bin S/N, and bin flux.
    stel_vel = np.ma.MaskedArray(hdu['STELLAR_VEL'].data,
                                 mask=bm.flagged(hdu['STELLAR_VEL_MASK'].data, flag='DONOTUSE'))
    stel_sig = np.ma.MaskedArray(hdu['STELLAR_SIGMA'].data,
                                 mask=bm.flagged(hdu['STELLAR_VEL_MASK'].data, flag='DONOTUSE'))
    bin_snr = np.ma.MaskedArray(hdu['BIN_SNR'].data,
                                mask=bm.flagged(hdu['STELLAR_VEL_MASK'].data, flag='DONOTUSE'))
    bin_flux = np.ma.MaskedArray(hdu['BIN_MFLUX'].data,
                                 mask=bm.flagged(hdu['STELLAR_VEL_MASK'].data, flag='DONOTUSE'))

    # Get the central x and y spaxels depending on what center of non-masked pixels are. This is to
    # account for if the galaxy is not centered in the aperture. If the newly-found center does not
    # have signs of CRD qualities, go back to the cube center and check again (a couple CRDs were
    # not correctly identified when using the un-masked center).
    if original_center:
        central_x = stel_vel.shape[0] // 2
        central_y = stel_vel.shape[0] // 2
    else:
        # Get the velocity list and get the maximum and minimum non-masked x-values of the velocities,
        # and find that center. Repeat the same process for the y-values of the velocities.
        velocity_list = stel_vel.tolist()
        x_list = []
        for x in range(len(velocity_list)):
            for y in range(len(velocity_list[x])):
                if velocity_list[y][x] != None:
                    x_list.append(x)
                    break
        big_x = max(x_list)
        small_x = min(x_list)
        central_x = round((big_x - small_x) / 2 + small_x)

        y_list = []
        for x in range(len(velocity_list)):
            for y in range(len(velocity_list[x])):
                if velocity_list[x][y] != None:
                    y_list.append(x)
                    break
        big_y = max(y_list)
        small_y = min(y_list)
        central_y = round((big_y - small_y) / 2 + small_y)

    # Get the non-masked x, y, and velocity vector lists. The x and y lists will be centered around
    # the above defined central_x and central_y
    for k in range(stel_vel.shape[0]):
        for j in range(stel_vel.shape[1]):
            if str(stel_vel[j, k]) != '--' and str(stel_vel[j, k]) != 'masked':
                velocity = stel_vel[j, k]
                sigma = stel_sig[j, k]
                y_vector_list.append(j - central_y)
                x_vector_list.append(k - central_x)
                corresponding_velocity.append(velocity)
                corresponding_sigma.append(sigma)

    # Get the mean and standard deviation of the velocities and sigma, as well as median of velocities
    mean_vel = np.mean(corresponding_velocity)
    std_vel = np.std(corresponding_velocity)
    mean_sig = np.mean(corresponding_sigma)
    std_sig = np.std(corresponding_sigma)
    median_vel = np.median(corresponding_velocity)

    # Get the corrected velocities that are below 3*std and their x and y spaxel values
    for k in range(stel_vel.shape[0]):
        for j in range(stel_vel.shape[1]):
            if (str(stel_vel[j, k]) != '--' and str(stel_vel[j, k]) != 'masked'
                    and abs(stel_vel[j, k]) < 3 * std_vel):
                velocity = stel_vel[j, k] - median_vel
                new_y_vector_list.append(j - central_y)
                new_x_vector_list.append(k - central_x)
                new_corresponding_velocity.append(velocity)

    # Get the PA of these new vector lists
    angBest, angErr, vSyst = fit_kinematic_pa(new_x_vector_list, new_y_vector_list,
                                              new_corresponding_velocity, quiet=True,
                                              plot=False)

    # Create the new systematic correction with median and returned vSyst
    systematic_correction = median_vel + vSyst

    # Get the new corrected velocities, sigma values, and their corresponding x and y spaxel values
    for k in range(stel_vel.shape[0]):
        for j in range(stel_vel.shape[1]):
            velocity = stel_vel[k, j] - systematic_correction
            sigma = stel_sig[k, j]
            new_y_vector_list.append(j - central_y)
            new_x_vector_list.append(k - central_x)
            allVelocityList.append(velocity)
            allSigmaList.append(sigma)

    # Get new mean and standard deviation after applying the systematic_correction to the velocities.
    mean_vel = np.mean(allVelocityList)
    std_vel = np.std(allVelocityList)

    # Create masked Velocity list for 3*std outliers
    for i in range(len(allVelocityList)):
        if abs(allVelocityList[i]) > 3 * std_vel:
            newVelocityList.append(3 * std_vel)
            maskedBoolean = True
        else:
            newVelocityList.append(allVelocityList[i])
            maskedBoolean = False
        maskboollistVel.append(maskedBoolean)
    maskedVelocityList = np.ma.array(newVelocityList, mask=maskboollistVel)

    # Create masked Sigma list for 3*std outliers
    for i in range(len(allSigmaList)):
        if abs(allSigmaList[i]) > mean_sig + 3 * std_sig:
            newSigmaList.append(3 * std_sig)
            maskedBoolean = True
        else:
            newSigmaList.append(allSigmaList[i])
            maskedBoolean = False
        maskboollistSig.append(maskedBoolean)
    maskedSigmaList = np.ma.array(newSigmaList, mask=maskboollistSig)

    # Create the 3D masked velocity and sigma lists
    counter = 0
    for i in range(stel_vel.shape[0]):
        inter_vel = []
        inter_sig = []
        for j in range(stel_vel.shape[1]):
            inter_vel.append(maskedVelocityList[counter])
            inter_sig.append(maskedSigmaList[counter])
            counter += 1
        shapedVelocityList.append(inter_vel)
        shapedSigmaList.append(inter_sig)

    # Get variables to plot the PAKin
    x, y, vel = map(np.ravel, [new_x_vector_list, new_y_vector_list, new_corresponding_velocity])
    rad = np.sqrt(np.max(x**2 + y**2))
    ang = [0, np.pi] + np.radians(angBest)

    del hdu

    # Return everything
    return (rad, ang, angBest, stel_vel, stel_sig, bin_snr, central_x, central_y, systematic_correction,
            shapedVelocityList, shapedSigmaList, angErr, bin_flux)


def get_pixel_pairs(rad, ang, central_x, central_y, stel_vel):
    # Define needed lists and variables
    pixel_pairs = []
    makeSmaller = True
    divideParam = 1.5

    # This is needed in order to define the length of the axis that corresponds to the PA_Kin. The ends
    # will always be overlapping masked values so no data will be lost, but need to make sure the maximum
    # and minimum values are within the dimensions of the MaNGA Datacube.
    while makeSmaller:
        x_list = -rad / divideParam * np.sin(ang) + central_x
        y_list = rad / divideParam * np.cos(ang) + central_y
        if round(max(x_list)) >= stel_vel.shape[0] or round(max(y_list)) >= stel_vel.shape[0]:
            divideParam += 0.1
        else:
            makeSmaller = False

    # Define the length of the x and y lists.
    len_x = round(x_list[1] - x_list[0])
    if len_x <= 1:
        len_x = 2

    if len_x % 2 != 0:
        first_y_length = round(len_x / 2) + 1
        second_y_length = round(len_x / 2)
    else:
        first_y_length = int(len_x / 2)
        second_y_length = int(len_x / 2)

    # Define the x_pixel_list to be first and last x values of the PA_Kin axis with length the same
    # as the dimension of the MaNGA Datacube. Define y_pixel_list in the exact same way.
    x_pixel_list = np.linspace(round(x_list[0]), round(x_list[1]), int(stel_vel.shape[0]))
    y_pixel_list = []

    first_half_y_pixel = np.linspace(y_list[0], central_y, int(stel_vel.shape[0] / 2))
    second_half_y_pixel = np.linspace(central_y, y_list[1], int(stel_vel.shape[0] / 2))
    y_pixel_list = np.concatenate([first_half_y_pixel, second_half_y_pixel])

    # Append each entry in the x_pixel_list and y_pixel_list together to make the pixel_pairs list.
    for l in range(len(x_pixel_list)):
        pixel_pairs.append((int(x_pixel_list[l]), int(y_pixel_list[l])))

    # Return the pixel_pairs list and divideParam to accurately plot PA_Kin axis.
    return pixel_pairs, divideParam


def get_PAKin_data(pixel_pairs, stel_vel, stel_sig, bin_snr, bin_flux, systematic_correction):
    # Define needed variables and lists.
    vel_spaxel_pair_offset = 0
    sig_spaxel_pair_offset = 0
    vel_count = True
    sig_count = True
    corresponding_v = []
    corresponding_sigma = []
    corresponding_flux = []
    full_v = []
    full_sigma = []
    full_SNR = []
    full_flux = []

    # For each pixel_pair, get its velocity, sigma, snr, and flux.
    for p in range(len(pixel_pairs)):
        x = pixel_pairs[p][0]
        y = pixel_pairs[p][1]
        v = stel_vel[y][x] - systematic_correction
        sigma = stel_sig[y][x]
        snr = bin_snr[y][x]
        flux = bin_flux[y][x]

        # For the non-masked velocity values, make sure the bin has a snr>4. If so, add that velocity
        # to the "good velocity" list. Same thing for sigma, except has to have snr>7.
        if str(v) != '--':
            # For velocity
            if snr > 4:
                vel_count = False
                corresponding_v.append(v)
            if snr > 6:
                sig_count = False
                corresponding_sigma.append(sigma)
                corresponding_flux.append(flux)

            # Also add all the raw values to these full lists.
            full_v.append(v)
            full_sigma.append(sigma)
            full_SNR.append(snr)
            full_flux.append(flux)

        # This is needed to see how far along the PA_Kin we must go to get to the first non-masked
        # velocity and sigma value (requiring the minimum snr as above). This is to accurately find
        # where the extrema occur along the PA_Kin axis.
        if vel_count:
            vel_spaxel_pair_offset += 1
        if sig_count:
            sig_spaxel_pair_offset += 1

    # Return everything.
    return (corresponding_v, corresponding_sigma, corresponding_flux, full_v, full_sigma, full_SNR,
            full_flux, vel_spaxel_pair_offset, sig_spaxel_pair_offset)


def do_the_thing():
    # Build the plateifu list by scanning MAPS_DIR for all MAPS files.
    # Relies on the standard MaNGA directory structure: MAPS_DIR/[plate]/[ifu]/
    maps_files = glob.glob(
        os.path.join(MAPS_DIR, '*-MAPS-VOR10-MILESHC-None.fits.gz')
    )

    plateifu = []
    for f in maps_files:
        fname = os.path.basename(f)
        parts = fname.split('-')
        plate = parts[1]
        ifu   = parts[2]
        plateifu.append(f"{plate}-{ifu}")

    plateifu = sorted(plateifu)

    # Create output subdirectories if they do not already exist.
    os.makedirs(os.path.join(OUTPUT_DIR, 'no_CRD'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'maybe_CRD'), exist_ok=True)

    # Define lists
    analyzed_galaxies = []
    no_CRD = []
    maybe_CRD = []
    problems = []

    # Define analysis booleans
    plot = False
    details = False
    writing = True

    for j in range(len(plateifu)):
        if plateifu[j] not in analyzed_galaxies:
            # Get plate and ifu
            plate = plateifu[j].split('-')[0]
            ifu = plateifu[j].split('-')[1]

            # First check the cube center spaxel
            original_center = True

            # Set CRD booleans to false
            ThreeVPeak = False
            FourVPeak = False
            Double_Sig = False
            CRD = False

            try:
                # Loop through the un-masked pixel center and the cube center
                for z in range(2):
                    if z == 1:
                        # If the cube center spaxel does not give evidence of crd in PA_Kin, check the masked
                        # spaxel center - sometimes these can differ if the object was not originally centered
                        # in the IFU aperture.
                        original_center = False

                    # Find rotation angle and get all the goodies
                    (rad, ang, angBest, stel_vel, stel_sig, bin_snr,
                     central_x, central_y, systematic_correction, shapedVelocityList,
                     shapedSigmaList, angErr, bin_flux) = find_rotation_angle(plate, ifu, original_center)

                    # Check the number of unique velocities -> number of voronoi bins. If greater than 60, continue.
                    # If not, do not bother analyzing the PAKin, too difficult.
                    non_masked_velocities = stel_vel[stel_vel.mask == False]
                    unique_velocities = []
                    for m in range(len(non_masked_velocities)):
                        if non_masked_velocities[m] not in unique_velocities:
                            unique_velocities.append(non_masked_velocities[m])

                    firstHalf_positivePeaks = None
                    firstHalf_negativePeaks = None
                    secondHalf_positivePeaks = None
                    secondHalf_negativePeaks = None

                    if len(unique_velocities) < 60:
                        print('\n' + plate + '-' + ifu, str(j) + '/' + str(len(plateifu)))
                        print('ThreeVPeak: ', ThreeVPeak)
                        print('FourVPeak: ', FourVPeak)
                        print('Double_Sig: ', Double_Sig)
                        print('--------------------')
                        if writing:
                            analyzed_galaxies.append(plate + '-' + ifu)
                            name = os.path.join(OUTPUT_DIR, 'analyzed_galaxies.txt')
                            np.savetxt(name, analyzed_galaxies, fmt='%s')
                            print('Writing ' + name)

                            no_CRD.append(plate + '-' + ifu)
                            name = os.path.join(OUTPUT_DIR, 'no_CRD', 'no_CRD.txt')
                            np.savetxt(name, no_CRD, fmt='%s')
                            print('Writing ' + name)

                    else:
                        # Check the returned angBest, and the upper- and lower-limit given the returned angErr.
                        for i in range(3):
                            new_angBest = 0
                            if i == 0:
                                new_angBest = angBest
                            if i == 1:
                                new_angBest = angBest + angErr
                            if i == 2:
                                new_angBest = angBest - angErr

                            KeepGoing = False

                            # Get the angle with/without the returned error
                            new_ang = [0, np.pi] + np.radians(new_angBest)

                            # Find the spaxel pairs along PAKin
                            pixel_pairs, divideParam = get_pixel_pairs(rad, new_ang, central_x, central_y,
                                                                       stel_vel)

                            # Get velocity, sigma, and snr for spaxel pairs
                            (v, sigma, flux, full_v, full_sigma, full_SNR, full_flux, vel_spaxel_pair_offset,
                             sig_spaxel_pair_offset) = get_PAKin_data(pixel_pairs, stel_vel, stel_sig, bin_snr,
                                                                      bin_flux, systematic_correction)

                            #----------------------------------------------------
                            # Check Velocity Map
                            #----------------------------------------------------
                            velocityCheck = True
                            middle_spaxel_pair = len(v) // 2
                            v_firstHalf = np.concatenate(([0], v[:middle_spaxel_pair]))
                            v_secondHalf = np.concatenate((v[middle_spaxel_pair:], [0]))

                            # Check if both sides' velocity is all positive or negative -> no CRD
                            if (all(v >= 0 for v in v_firstHalf) or all(v <= 0 for v in v_firstHalf)
                                    and all(v >= 0 for v in v_secondHalf) or all(v <= 0 for v in v_secondHalf)):
                                velocityCheck = False

                            if velocityCheck:
                                firstHalf_positivePeaks = find_peaks(v_firstHalf, height=0, width=2, prominence=2)
                                firstHalf_negativePeaks = find_peaks(-1 * v_firstHalf, height=0, width=2, prominence=2)
                                secondHalf_positivePeaks = find_peaks(v_secondHalf, height=0, width=2, prominence=2)
                                secondHalf_negativePeaks = find_peaks(-1 * v_secondHalf, height=0, width=2, prominence=2)

                                FH_Pos = False
                                SH_Pos = False
                                FH_Neg = False
                                SH_Neg = False

                                if len(firstHalf_positivePeaks[0]) != 0:
                                    # Make sure the first half velocity peak has at least 2 unique velocity values
                                    FH_Posleft = round(firstHalf_positivePeaks[1]['left_ips'][0]) + vel_spaxel_pair_offset
                                    FH_Posright = round(firstHalf_positivePeaks[1]['right_ips'][0] + vel_spaxel_pair_offset)
                                    FH_PosPeak = np.linspace(FH_Posleft, FH_Posright, FH_Posright - FH_Posleft + 1)
                                    FH_Pos_velocities = [v[int(element - vel_spaxel_pair_offset - 2)] for element in FH_PosPeak]
                                    FH_Pos_unique_velocities = []
                                    for i in range(len(FH_Pos_velocities)):
                                        if (FH_Pos_velocities[i] not in FH_Pos_unique_velocities and
                                                FH_Pos_velocities[i] > 0):
                                            FH_Pos_unique_velocities.append(FH_Pos_velocities[i])
                                    if len(FH_Pos_unique_velocities) >= 2:
                                        FH_Pos = True

                                if len(secondHalf_positivePeaks[0]) != 0:
                                    # Make sure the second half velocity peak has at least 2 unique velocity values
                                    SH_Posleft = round(secondHalf_positivePeaks[1]['left_ips'][0]) + middle_spaxel_pair + vel_spaxel_pair_offset
                                    SH_Posright = round(secondHalf_positivePeaks[1]['right_ips'][0] + middle_spaxel_pair + vel_spaxel_pair_offset)
                                    SH_PosPeak = np.linspace(SH_Posleft, SH_Posright, SH_Posright - SH_Posleft + 1)
                                    SH_Pos_velocities = [v[int(element - vel_spaxel_pair_offset - 2)] for element in SH_PosPeak]
                                    SH_Pos_unique_velocities = []
                                    for i in range(len(SH_Pos_velocities)):
                                        if (SH_Pos_velocities[i] not in SH_Pos_unique_velocities and
                                                SH_Pos_velocities[i] > 0):
                                            SH_Pos_unique_velocities.append(SH_Pos_velocities[i])
                                    if len(SH_Pos_unique_velocities) >= 2:
                                        SH_Pos = True

                                if len(firstHalf_negativePeaks[0]) != 0:
                                    # Make sure the first half velocity peak has at least 2 unique velocity values
                                    FH_Negleft = round(firstHalf_negativePeaks[1]['left_ips'][0]) + vel_spaxel_pair_offset
                                    FH_Negright = round(firstHalf_negativePeaks[1]['right_ips'][0] + vel_spaxel_pair_offset)
                                    FH_NegPeak = np.linspace(FH_Negleft, FH_Negright, FH_Negright - FH_Negleft + 1)
                                    FH_Neg_velocities = [v[int(element - vel_spaxel_pair_offset - 2)] for element in FH_NegPeak]
                                    FH_Neg_unique_velocities = []
                                    for i in range(len(FH_Neg_velocities)):
                                        if (FH_Neg_velocities[i] not in FH_Neg_unique_velocities and
                                                FH_Neg_velocities[i] < 0):
                                            FH_Neg_unique_velocities.append(FH_Neg_velocities[i])
                                    if len(FH_Neg_unique_velocities) >= 2:
                                        FH_Neg = True

                                if len(secondHalf_negativePeaks[0]) != 0:
                                    # Make sure the second half velocity peak has at least 2 unique velocity values
                                    SH_Negleft = round(secondHalf_negativePeaks[1]['left_ips'][0]) + middle_spaxel_pair + vel_spaxel_pair_offset
                                    SH_Negright = round(secondHalf_negativePeaks[1]['right_ips'][0] + middle_spaxel_pair + vel_spaxel_pair_offset)
                                    SH_NegPeak = np.linspace(SH_Negleft, SH_Negright, SH_Negright - SH_Negleft + 1)
                                    SH_Neg_velocities = [v[int(element - vel_spaxel_pair_offset - 2)] for element in SH_NegPeak]
                                    SH_Neg_unique_velocities = []
                                    for i in range(len(SH_Neg_velocities)):
                                        if (SH_Neg_velocities[i] not in SH_Neg_unique_velocities and
                                                SH_Neg_velocities[i] < 0):
                                            SH_Neg_unique_velocities.append(SH_Neg_velocities[i])
                                    if len(SH_Neg_unique_velocities) >= 2:
                                        SH_Neg = True

                                PeakBoolSum = int(FH_Pos) + int(SH_Pos) + int(FH_Neg) + int(SH_Neg)
                                if PeakBoolSum == 3:
                                    ThreeVPeak = True
                                if PeakBoolSum == 4:
                                    FourVPeak = True

                            #----------------------------------------------------
                            # Check Sigma Map
                            #----------------------------------------------------
                            # Add values to either side of recorded sigma to check if the peak occurs at the
                            # very edge of kinematic PA
                            first_sigma = sigma[0]
                            last_sigma = sigma[-1]
                            zero_sigma = None
                            end_sigma = None

                            for l in range(len(sigma)):
                                current_sigma = sigma[l]
                                if current_sigma != first_sigma:
                                    zero_sigma = current_sigma
                                    break

                            for l in range(len(sigma)):
                                current_sigma = sigma[-l - 1]
                                if current_sigma != last_sigma:
                                    end_sigma = current_sigma
                                    break

                            edited_sigma = np.concatenate(([np.median(sigma)], sigma, [np.median(sigma)]))

                            # Set the sigma threshold to be the mean and a fifth of a standard deviation. This
                            # added part was from trial and error to find largest threshold such that known
                            # two-sigma CRDs would still be flagged.
                            sigmaThreshold = np.mean(edited_sigma) + 0.20 * np.std(edited_sigma)

                            # Find the peaks. Using a larger width as well as requiring a prominence.
                            found_sigma_peaks = find_peaks(edited_sigma,
                                                           height=sigmaThreshold,
                                                           width=1.9,
                                                           prominence=5)

                            # Also find peaks in the flux along the kinematic PA.
                            flux_peaks = find_peaks(flux, width=1)

                            # Define all the needed analysis variables and lists.
                            SigmaPeak1Index = None
                            SigmaPeak2Index = None
                            sigmaPeak1 = None
                            sigmaPeak2 = None
                            fluxPeak = None
                            sigma_peaks = []
                            sigma_peak_locations = []
                            firstHalfSigmaPeaks = []
                            secondHalfSigmaPeaks = []
                            heights = []
                            widths = []
                            FSP_VP = []
                            FSP_unique_VP = []
                            FSP_above_sigmaThreshold = []
                            SSP_VP = []
                            SSP_unique_VP = []
                            SSP_above_sigmaThreshold = []
                            counter_sigmaPeak1 = 0
                            counter_sigmaPeak2 = 0

                            # If there are at least two sigma peaks and a flux peak, analyze them
                            if len(found_sigma_peaks[0]) > 2 and len(found_sigma_peaks[0]) < 5 and len(flux_peaks[0]) > 0:
                                # Define flux peak. If there are multiple, take the one that is the widest.
                                if len(flux_peaks[0]) > 1:
                                    widths = []
                                    for i in range(len(flux_peaks[0])):
                                        peakIndex = flux_peaks[0].tolist().index(flux_peaks[0][i])
                                        widths.append(flux_peaks[1]['widths'][peakIndex])
                                    sorted_widths = widths.copy()
                                    sorted_widths.sort()
                                    widesetPeak = sorted_widths[-1]
                                    fluxPeakIndex = flux_peaks[1]['widths'].tolist().index(widesetPeak)
                                else:
                                    fluxPeakIndex = 0

                                fluxPeak = np.linspace(round(flux_peaks[1]['left_ips'][fluxPeakIndex]) + sig_spaxel_pair_offset,
                                                       round(flux_peaks[1]['right_ips'][fluxPeakIndex]) + sig_spaxel_pair_offset,
                                                       round(flux_peaks[1]['right_ips'][fluxPeakIndex]) - round(flux_peaks[1]['left_ips'][fluxPeakIndex]) + 1)

                                # Split into halves
                                for i in range(len(found_sigma_peaks[0])):
                                    if found_sigma_peaks[0][i] <= middle_spaxel_pair:
                                        firstHalfSigmaPeaks.append(found_sigma_peaks[0][i])
                                    if found_sigma_peaks[0][i] > middle_spaxel_pair:
                                        secondHalfSigmaPeaks.append(found_sigma_peaks[0][i])

                                # Get the biggest peak in the first half
                                if len(firstHalfSigmaPeaks) >= 2:
                                    for i in range(len(firstHalfSigmaPeaks)):
                                        peakIndex = found_sigma_peaks[0].tolist().index(firstHalfSigmaPeaks[i])
                                        heights.append(found_sigma_peaks[1]['peak_heights'][peakIndex])
                                    sorted_heights = heights.copy()
                                    sorted_heights.sort()
                                    biggestPeak = sorted_heights[-1]
                                    SigmaPeak1Index = found_sigma_peaks[1]['peak_heights'].tolist().index(biggestPeak)
                                elif len(firstHalfSigmaPeaks) == 1:
                                    SigmaPeak1Index = found_sigma_peaks[0].tolist().index(firstHalfSigmaPeaks[0])
                                else:
                                    SigmaPeak1Index = None

                                # Get the biggest peak in the second half
                                if len(secondHalfSigmaPeaks) >= 2:
                                    for i in range(len(secondHalfSigmaPeaks)):
                                        peakIndex = found_sigma_peaks[0].tolist().index(secondHalfSigmaPeaks[i])
                                        heights.append(found_sigma_peaks[1]['peak_heights'][peakIndex])
                                    sorted_heights = heights.copy()
                                    sorted_heights.sort()
                                    biggestPeak = sorted_heights[-1]
                                    SigmaPeak2Index = found_sigma_peaks[1]['peak_heights'].tolist().index(biggestPeak)
                                elif len(secondHalfSigmaPeaks) == 1:
                                    SigmaPeak2Index = found_sigma_peaks[0].tolist().index(secondHalfSigmaPeaks[0])
                                else:
                                    SigmaPeak2Index = None

                            # If there are only two peaks and a flux peak, assign them accordingly
                            elif len(found_sigma_peaks[0]) == 2 and len(flux_peaks[0]) > 0:
                                # Define flux peak. If there are multiple, take the one that is the widest.
                                if len(flux_peaks[0]) > 1:
                                    for i in range(len(flux_peaks[0])):
                                        peakIndex = flux_peaks[0].tolist().index(flux_peaks[0][i])
                                        widths.append(flux_peaks[1]['widths'][peakIndex])
                                    sorted_widths = widths.copy()
                                    sorted_widths.sort()
                                    widesetPeak = sorted_widths[-1]
                                    fluxPeakIndex = flux_peaks[1]['widths'].tolist().index(widesetPeak)
                                else:
                                    fluxPeakIndex = 0

                                fluxPeak = np.linspace(round(flux_peaks[1]['left_ips'][fluxPeakIndex]) + sig_spaxel_pair_offset,
                                                       round(flux_peaks[1]['right_ips'][fluxPeakIndex]) + sig_spaxel_pair_offset,
                                                       round(flux_peaks[1]['right_ips'][fluxPeakIndex]) - round(flux_peaks[1]['left_ips'][fluxPeakIndex]) + 1)

                                SigmaPeak1Index = 0
                                SigmaPeak2Index = 1

                            if SigmaPeak1Index != None and SigmaPeak2Index != None:
                                # Construct list of all spaxels in the sigma peaks
                                sigmaPeak1 = np.linspace(round(found_sigma_peaks[1]['left_ips'][SigmaPeak1Index]) + sig_spaxel_pair_offset,
                                                         round(found_sigma_peaks[1]['right_ips'][SigmaPeak1Index]) + sig_spaxel_pair_offset,
                                                         round(found_sigma_peaks[1]['right_ips'][SigmaPeak1Index]) - round(found_sigma_peaks[1]['left_ips'][SigmaPeak1Index]) + 1)
                                sigmaPeak2 = np.linspace(round(found_sigma_peaks[1]['left_ips'][SigmaPeak2Index]) + sig_spaxel_pair_offset,
                                                         round(found_sigma_peaks[1]['right_ips'][SigmaPeak2Index]) + sig_spaxel_pair_offset,
                                                         round(found_sigma_peaks[1]['right_ips'][SigmaPeak2Index]) - round(found_sigma_peaks[1]['left_ips'][SigmaPeak2Index]) + 1)

                                # Get all of the unique sigma values in the sigma peaks. Also get the unique sigma
                                # values that are greater than the sigma threshold.
                                FSP_VP = [edited_sigma[int(element - sig_spaxel_pair_offset - 2)] for element in sigmaPeak1]
                                for i in range(len(FSP_VP)):
                                    if FSP_VP[i] not in FSP_unique_VP and FSP_VP[i] != 100.0:
                                        FSP_unique_VP.append(FSP_VP[i])
                                for i in range(len(FSP_unique_VP)):
                                    if FSP_unique_VP[i] > sigmaThreshold:
                                        FSP_above_sigmaThreshold.append(FSP_unique_VP[i])

                                SSP_VP = [edited_sigma[int(element - sig_spaxel_pair_offset - 2)] for element in sigmaPeak2]
                                for i in range(len(SSP_VP)):
                                    if SSP_VP[i] not in SSP_unique_VP and SSP_VP[i] != 100.0:
                                        SSP_unique_VP.append(SSP_VP[i])
                                for i in range(len(SSP_unique_VP)):
                                    if SSP_unique_VP[i] > sigmaThreshold:
                                        SSP_above_sigmaThreshold.append(SSP_unique_VP[i])

                                # Make sure there are at least 3 unique sigma values in both peaks, as well as
                                # at least 2 that are greater than the sigma threshold. This is to try
                                # and eliminate peaks that occur from a single bin, or only two bins, that are
                                # very large due to having low S/N from Voronoi binning procedure.
                                if (len(FSP_unique_VP) >= 3 and len(SSP_unique_VP) >= 3 and
                                        len(FSP_above_sigmaThreshold) >= 2 and len(SSP_above_sigmaThreshold) >= 2):
                                    # Get the peak locations and sort them
                                    sigma_peak_locations.append(found_sigma_peaks[0][SigmaPeak1Index] + sig_spaxel_pair_offset)
                                    sigma_peak_locations.append(found_sigma_peaks[0][SigmaPeak2Index] + sig_spaxel_pair_offset)
                                    sigma_peak_locations.sort()

                                    # Do not want to include sigma peaks that overlap with the flux peak (residing from
                                    # the central bulge of galaxy), so make sure recorded sigma peaks are not from this
                                    if len(sigmaPeak1) >= len(fluxPeak):
                                        for k in range(len(sigmaPeak1)):
                                            if sigmaPeak1[k] in fluxPeak:
                                                counter_sigmaPeak1 += 1
                                        if counter_sigmaPeak1 < (len(fluxPeak) * 0.75):
                                            sigma_peaks.append(sigmaPeak1)
                                    elif len(fluxPeak) > len(sigmaPeak1):
                                        for k in range(len(fluxPeak)):
                                            if fluxPeak[k] in sigmaPeak1:
                                                counter_sigmaPeak1 += 1
                                        if counter_sigmaPeak1 < (len(sigmaPeak1) * 0.75):
                                            sigma_peaks.append(sigmaPeak1)

                                    if len(sigmaPeak2) >= len(fluxPeak):
                                        for k in range(len(sigmaPeak2)):
                                            if sigmaPeak2[k] in fluxPeak:
                                                counter_sigmaPeak2 += 1
                                        if counter_sigmaPeak2 < (len(fluxPeak) * 0.75):
                                            sigma_peaks.append(sigmaPeak2)
                                    elif len(fluxPeak) > len(sigmaPeak2):
                                        for k in range(len(fluxPeak)):
                                            if fluxPeak[k] in sigmaPeak2:
                                                counter_sigmaPeak2 += 1
                                        if counter_sigmaPeak2 < (len(sigmaPeak2) * 0.75):
                                            sigma_peaks.append(sigmaPeak2)

                                    # Check if there are two sigma peaks that are on opposite sides of the galaxy.
                                    # If so, set Double_Sig to true.
                                    if (len(sigma_peaks) == 2 and sigma_peak_locations[0] < middle_spaxel_pair
                                            and sigma_peak_locations[1] > middle_spaxel_pair):
                                        Double_Sig = True

                            # Check if there are 3/4V peaks or 2 sigma peaks along this kinematic PA. If so,
                            # then move on!
                            if ThreeVPeak or FourVPeak or Double_Sig:
                                CRD = True
                                break

                    # If 3/4V peaks or 2 sigma peaks, break out of the re-checking center of galaxy loop and
                    # move on!
                    if CRD:
                        break

                #----------------------------------------------------
                # Report extrema info
                #----------------------------------------------------
                print('\n' + plate + '-' + ifu, str(j + 1) + '/' + str(len(plateifu)))
                print('ThreeVPeak: ', ThreeVPeak)
                print('FourVPeak: ', FourVPeak)
                print('Double_Sig: ', Double_Sig)
                print('--------------------')

                if len(unique_velocities) >= 60:
                    del stel_vel
                    del stel_sig
                    del bin_snr
                    del bin_flux
                    del firstHalf_positivePeaks
                    del firstHalf_negativePeaks
                    del secondHalf_positivePeaks
                    del secondHalf_negativePeaks
                    del edited_sigma
                    del found_sigma_peaks
                    del flux_peaks
                    del SigmaPeak1Index
                    del SigmaPeak2Index
                    del sigmaPeak1
                    del sigmaPeak2
                    del sigma_peaks
                    del sigma_peak_locations
                    del firstHalfSigmaPeaks
                    del secondHalfSigmaPeaks
                    del heights
                    del widths
                    del FSP_VP
                    del FSP_unique_VP
                    del FSP_above_sigmaThreshold
                    del SSP_VP
                    del SSP_unique_VP
                    del SSP_above_sigmaThreshold
                    del counter_sigmaPeak1
                    del counter_sigmaPeak2

                if writing:
                    analyzed_galaxies.append(plate + '-' + ifu)
                    name = os.path.join(OUTPUT_DIR, 'analyzed_galaxies.txt')
                    np.savetxt(name, analyzed_galaxies, fmt='%s')
                    print('Writing ' + name)

                    if int(ThreeVPeak) + int(FourVPeak) + int(Double_Sig) == 0:
                        no_CRD.append(plate + '-' + ifu)
                        name = os.path.join(OUTPUT_DIR, 'no_CRD', 'no_CRD.txt')
                        np.savetxt(name, no_CRD, fmt='%s')
                        print('Writing ' + name)
                    else:
                        data = [ThreeVPeak, FourVPeak, Double_Sig]
                        name = os.path.join(OUTPUT_DIR, 'maybe_CRD', plate + '-' + ifu + '.txt')
                        np.savetxt(name, data, fmt='%s')
                        print('Writing ' + name)

                        maybe_CRD.append(plateifu[j])
                        name = os.path.join(OUTPUT_DIR, 'maybe_CRD', 'maybe_CRD.txt')
                        np.savetxt(name, maybe_CRD, fmt='%s')
                        print('Writing ' + name)

            except ValueError:
                print('ValueError - Cannot find PA_Kin')
                print('--------------------')
                if writing:
                    analyzed_galaxies.append(plate + '-' + ifu)
                    name = os.path.join(OUTPUT_DIR, 'analyzed_galaxies.txt')
                    np.savetxt(name, analyzed_galaxies, fmt='%s')
                    print('Writing ' + name)

                    problems.append(plate + '-' + ifu)
                    name = os.path.join(OUTPUT_DIR, 'problems.txt')
                    np.savetxt(name, problems, fmt='%s')
                    print('Writing ' + name)

            except IndexError:
                print('IndexError - Cannot find PA_Kin')
                print('--------------------')
                if writing:
                    analyzed_galaxies.append(plate + '-' + ifu)
                    name = os.path.join(OUTPUT_DIR, 'analyzed_galaxies.txt')
                    np.savetxt(name, analyzed_galaxies, fmt='%s')
                    print('Writing ' + name)

                    problems.append(plate + '-' + ifu)
                    name = os.path.join(OUTPUT_DIR, 'problems.txt')
                    np.savetxt(name, problems, fmt='%s')
                    print('Writing ' + name)


#-------------------------------------------------
if __name__ == "__main__":
    do_the_thing()
