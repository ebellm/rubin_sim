import numpy as np
import numpy.lib.recfunctions as rfn
from .offsets import OffsetSNR
from scipy.spatial import cKDTree as kdtree
import warnings
from .star_tools import stars_project, assign_patches
import sys


__all__ = ["generate_catalog"]


def wrapRA(ra):
    """Wraps RA into 0-360 degrees."""
    ra = ra % 360.0
    return ra


def capDec(dec):
    """Terminates declination at +/- 90 degrees."""
    dec = np.where(dec > 90, 90, dec)
    dec = np.where(dec < -90, -90, dec)
    return dec


def treexyz(ra, dec):
    """Calculate x/y/z values for ra/dec points, ra/dec in radians."""
    # Note ra/dec can be arrays.
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def buildTree(simDataRa, simDataDec, leafsize=100):
    """Build KD tree on simDataRA/Dec and set radius (via setRad) for matching.

    simDataRA, simDataDec = RA and Dec values (in radians).
    leafsize = the number of Ra/Dec pointings in each leaf node."""
    if np.any(np.abs(simDataRa) > np.pi * 2.0) or np.any(
        np.abs(simDataDec) > np.pi * 2.0
    ):
        raise ValueError("Expecting RA and Dec values to be in radians.")
    x, y, z = treexyz(simDataRa, simDataDec)
    data = list(zip(x, y, z))
    if np.size(data) > 0:
        starTree = kdtree(data, leafsize=leafsize)
    else:
        raise ValueError("SimDataRA and Dec should have length greater than 0.")
    return starTree


def generate_catalog(
    visits,
    stars_array,
    offsets=None,
    lsst_filter="r",
    n_patches=16,
    radius_fov=1.8,
    seed=42,
    uncert_floor=0.005,
    verbose=True,
):
    """
    Generate a catalog of observed stellar magnitudes.

    visits:  A numpy array with the properties of the visits.  Expected to have Opsim-like values
    starsDbAddress:  a sqlAlchemy address pointing to a database that contains properties of stars used as input.
    offsets:  A list of instatiated classes that will apply offsets to the stars
    lsst_filter:  Which filter to use for the observed stars
    obs_file:  File to write the observed stellar magnitudes to
    truthFile:  File to write the true stellar magnitudes to
    n_patches:  Number of patches to divide the FoV into.  Must be an integer squared
    radius_fov: Radius of the telescope field of view in degrees
    seed: random number seed
    uncert_floor: value to add in quadrature to magnitude uncertainties
    """

    if offsets is None:
        # Maybe change this to just run with a default SNR offset
        warnings.warn("Warning, no offsets set, returning without running")
        return

    # For computing what the 'expected' uncertainty on the observation will be
    mag_uncert = OffsetSNR(lsst_filter=lsst_filter)

    # set the radius for the kdtree
    x0, y0, z0 = (1, 0, 0)
    x1, y1, z1 = treexyz(np.radians(radius_fov), 0)
    treeRadius = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)

    newcols = ["x", "y", "radius", "patch_id", "sub_patch", "observed_mag", "mag_uncert"]
    newtypes = [float, float, float, int, int, float, float]
    stars_new = np.zeros(stars_array.size, dtype=list(zip(newcols, newtypes)))

    # only need to output these columns (saving rmag+ for now for convienence)
    output_cols = ["id", "patch_id", "observed_mag", "mag_uncert", "%smag" % lsst_filter, "ra", "decl"]
    output_dtypes = [int, int, float, float, float, float, float]

    stars = rfn.merge_arrays([stars_array, stars_new], flatten=True)

    # Build a KDTree for the stars
    starTree = buildTree(np.radians(stars["ra"]), np.radians(stars["decl"]))

    # XXX--maybe update the way seeding is going on
    np.random.seed(seed)

    list_of_observed_arrays = []

    n_visits = np.size(visits)

    for i, visit in enumerate(visits):
        dmags = {}
        # Calc x,y, radius for each star, crop off stars outside the FoV
        # XXX - plan to replace with code to see where each star falls and get chipID.
        vx, vy, vz = treexyz(np.radians(visit["ra"]), np.radians(visit["dec"]))
        indices = starTree.query_ball_point((vx, vy, vz), treeRadius)
        stars_in = stars[indices]
        stars_in = stars_project(stars_in, visit)

        # Assign patchIDs
        stars_in = assign_patches(stars_in, visit, n_patches=n_patches)

        # Apply the offsets that have been configured
        for offset in offsets:
            dmags[offset.newkey] = offset(stars_in, visit, dmags=dmags)

        # Total up all the dmag's to make the observed magnitude
        keys = list(dmags.keys())
        obs_mag = stars_in["%smag" % lsst_filter].copy()
        for key in keys:
            obs_mag += dmags[key]

        # Calculate the uncertainty in the observed mag:
        mag_err = (
            mag_uncert.calc_mag_errors(obs_mag, errOnly=True, m5=visit["fiveSigmaDepth"])
            ** 2
            + uncert_floor**2
        ) ** 0.5

        # put values into the right columns
        stars_in["observed_mag"] = obs_mag
        stars_in["mag_uncert"] = mag_err
        # Should shrink this down so we only return the needed columns
        # observed_mag, mag_uncert, patchid, star_id
        sub_cols = np.empty(stars_in.size, dtype=list(zip(output_cols, output_dtypes)))
        for key in output_cols:
            sub_cols[key] = stars_in[key]
        list_of_observed_arrays.append(sub_cols)
        if verbose:
            progress = i / n_visits * 100
            text = "\rprogress = %.2f%%" % progress
            sys.stdout.write(text)
            sys.stdout.flush()

    result = np.concatenate(list_of_observed_arrays)
    return result
