__all__ = (
    "CvLc",
    "CVPopMetric",
    "generate_cv_pop_slicer",
    "sample_dwarf_nova_positions",
    "expected_number_of_dwarf_novae",
    "DN_SUBTYPE_FRACTIONS",
    "DN_SCALE_HEIGHTS",
)

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, Galactocentric
from rubin_scheduler.utils import SURVEY_START_MJD

from rubin_sim.maf.maps.ebv_3d_hp import ebv_3d_hp
from rubin_sim.maf.utils import m52snr
from rubin_sim.phot_utils import DustValues

from ..metrics import BaseMetric
from ..slicers import UserPointsSlicer

# ---------------------------------------------------------------------------
# Population constants.
#
# Dwarf nova subtype fractions, from the space densities of the three
# outbursting subtypes in Inight et al. 2023 (MNRAS 524, 4867; Table 4,
# rho_0(P2007) column, which adopts the Pretorius et al. 2007 scale heights):
# U Gem/Z Cam 0.63, SU UMa 1.88, WZ Sge 2.52 (x1e-6 pc^-3), i.e. a
# dwarf nova space density of ~5e-6 pc^-3.  The ratios are 0.125 : 0.375 :
# 0.50.  This is consistent with the Gaia DR2 150 pc volume-limited sample of
# Pala et al. 2020 (MNRAS 494, 3799; Table 1), which contains 2 U Gem,
# 9 SU UMa and 11 WZ Sge-type dwarf novae (22 of 42 CVs).
# ---------------------------------------------------------------------------
DN_SUBTYPE_FRACTIONS = {"ugem": 0.125, "suuma": 0.375, "wzsge": 0.50}

# Vertical exponential scale heights (pc) by subtype, following
# Pretorius, Knigge & Kolb 2007 (MNRAS 374, 1495), as used by Pala et al. 2020
# and Buckley et al. 2025 (arXiv:2509.07298): 120 pc for long-period systems
# (above the period gap), 260 pc for short-period systems, 450 pc for period
# bouncers.  WZ Sge stars are short-period systems at the period minimum
# and are assigned 260 pc (Pala et al. 2020).  Note that Canbay et al. 2023
# (AJ 165, 163) find a larger exponential scale height of 375 pc for all CVs
# from Gaia DR3, so these values may be on the low side.
DN_SCALE_HEIGHTS = {"ugem": 120.0, "suuma": 260.0, "wzsge": 260.0}

# Galactic structure parameters, from Bland-Hawthorn & Gerhard 2016
# (ARA&A 54, 529): R0 = 8.2 kpc, z_sun = 25 pc, thin disk radial scale length
# 2.6 kpc; boxy/peanut bulge exponential scale lengths 0.70 : 0.44 : 0.18 kpc
# with the bar major axis at 27 deg from the Sun-Galactic centre line, and
# a bulge stellar mass fraction of 0.3 +/- 0.06 of the total stellar mass.
GALACTIC_STRUCTURE = {
    "r_sun_kpc": 8.2,
    "z_sun_pc": 25.0,
    "disk_scale_length_kpc": 2.6,
    "disk_max_radius_kpc": 25.0,
    "bulge_scales_kpc": (0.70, 0.44, 0.18),
    "bar_angle_deg": 27.0,
}

# Fiducial single-visit 5-sigma depths, as used in XrbLc.detectable_duration.
LSST_SINGLE_EPOCH_DEPTH = {"u": 23.9, "g": 25.0, "r": 24.7, "i": 24.0, "z": 23.3, "y": 22.1}


class CvLc:
    """Synthesize dwarf nova (outbursting cataclysmic variable) lightcurves.

    The model treats each dwarf nova as a constant quiescent source plus a
    sequence of outbursts.  Each outburst is a piecewise-linear (in
    magnitude) template: a fast linear rise, a slowly declining plateau,
    and a final linear decline at the Bailey rate back to quiescence.
    Quiescent and outburst fluxes are added, so the system returns smoothly
    to quiescence.  Three subtypes are modelled, with parameters drawn from
    the literature as described in the methods below:

    ``wzsge``
        WZ Sge-type dwarf novae: rare (decade-scale), large-amplitude
        (~7-8 mag) superoutbursts only.
    ``suuma``
        SU UMa-type dwarf novae: normal outbursts (days) plus superoutbursts
        (~2 weeks) on a supercycle of several hundred days.
    ``ugem``
        U Gem / Z Cam-type (long-period) dwarf novae: normal outbursts of
        ~1-2 weeks every few weeks to months.  Z Cam standstills are
        not modelled.

    Parameters
    ----------
    seed : `int`, opt
        Seed for the random number generator.
    survey_length : `float`, opt
        Length of time (days) over which to generate outbursts.
    """

    def __init__(self, seed=42, survey_length=3652.0):
        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1
        self.rng = np.random.default_rng(seed)
        self.survey_length = survey_length

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------
    @staticmethod
    def quiescent_colors(subtype):
        """Quiescent colors (band - V) of dwarf novae per LSST filter.

        SDSS ugriz colors of dwarf novae in quiescence are taken from the
        medians of the Table 3 compilation of Kato, Maehara & Uemura 2012
        (PASJ 64, 63; arXiv:1111.4286).  For 160 systems below the period
        gap (P_orb < 2.16 h) the medians are u-g = -0.03, g-r = 0.09,
        r-i = 0.02, i-z = 0.19; for 78 systems above the gap (P_orb > 3 h),
        where the donor contributes more light, u-g = 0.09, g-r = 0.42,
        r-i = 0.37, i-z = 0.29.  These agree with the SDSS colors in the
        CRTS outburst catalogue of Coppejans et al. 2016 (MNRAS 456, 4441).
        z-y is not available and is assumed to be 0.1 (short period) and
        0.2 (long period).  V is derived from g and g-r using the Jester
        et al. 2005 (AJ 130, 873) transformation V = g - 0.59 (g-r) - 0.01.

        Parameters
        ----------
        subtype : `str`
            One of "wzsge", "suuma", "ugem".

        Returns
        -------
        colors : `dict` [`str`, `float`]
            Offset (band - V) for each LSST filter.
        """
        if subtype == "ugem":
            ug, gr, ri, iz, zy = 0.09, 0.42, 0.37, 0.29, 0.2
        else:
            ug, gr, ri, iz, zy = -0.03, 0.09, 0.02, 0.19, 0.1
        return CvLc._colors_to_offsets(ug, gr, ri, iz, zy)

    @staticmethod
    def outburst_colors():
        """Outburst colors (band - V) of dwarf novae per LSST filter.

        In outburst the optical light is dominated by the hot, optically
        thick accretion disk, which is approximated by a 12,000 K blackbody
        (as in Buckley et al. 2025, arXiv:2509.07298).  The LSST colors of
        a 12,000 K blackbody, computed with the rubin_sim baseline
        throughputs, are u-g = 0.01, g-r = -0.14, r-i = -0.19, i-z = -0.16,
        z-y = -0.15.  V is derived from g and g-r using the Jester et al.
        2005 (AJ 130, 873) transformation.

        Returns
        -------
        colors : `dict` [`str`, `float`]
            Offset (band - V) for each LSST filter.
        """
        return CvLc._colors_to_offsets(0.01, -0.14, -0.19, -0.16, -0.15)

    @staticmethod
    def _colors_to_offsets(ug, gr, ri, iz, zy):
        g_minus_v = 0.59 * gr + 0.01
        g = g_minus_v
        return {
            "u": g + ug,
            "g": g,
            "r": g - gr,
            "i": g - gr - ri,
            "z": g - gr - ri - iz,
            "y": g - gr - ri - iz - zy,
        }

    # ------------------------------------------------------------------
    # Population parameters
    # ------------------------------------------------------------------
    def orbital_periods(self, subtypes):
        """Draw orbital periods (hours) for each subtype.

        The distributions are set to match the percentiles of the orbital
        periods of the dwarf novae with each classification in the catalogue
        of Otulakowska-Hypka, Olech & Patterson 2016 (MNRAS 460, 2526;
        VizieR J/MNRAS/460/2526): WZ Sge-type stars cluster tightly at the
        period minimum (median 1.37 h, 10-90% range 1.30-1.56 h), SU UMa
        stars fill the range below the period gap (median 1.66 h, 10-90%
        range 1.37-2.10 h), and U Gem/Z Cam stars lie above the gap
        (median 5.2 h, 10-90% range 3.8-9.8 h).

        Parameters
        ----------
        subtypes : `np.ndarray` [`str`]
            Subtype of each system.

        Returns
        -------
        porb : `np.ndarray` [`float`]
            Orbital periods in hours.
        """
        n = len(subtypes)
        porb = np.zeros(n)
        w = subtypes == "wzsge"
        porb[w] = np.clip(self.rng.normal(1.38, 0.08, size=np.sum(w)), 1.28, 2.0)
        w = subtypes == "suuma"
        porb[w] = np.clip(self.rng.normal(1.70, 0.25, size=np.sum(w)), 1.30, 2.15)
        w = subtypes == "ugem"
        porb[w] = np.clip(10 ** self.rng.normal(np.log10(5.0), 0.18, size=np.sum(w)), 3.0, 15.0)
        return porb

    def quiescent_abs_mags(self, subtypes, porb):
        """Draw quiescent absolute V magnitudes.

        For the short-period subtypes we use the result of Kato 2015
        (PASJ 67, 108; Section 7.10) that the outburst "amplitude" measured
        from quiescence to the start of the superoutburst plateau (where
        ordinary superhumps appear) is 4.3 +/- 1.1 mag for SU UMa stars
        and 6.4 +/- 0.7 mag for WZ Sge stars, combined with the plateau
        absolute magnitude M_V = 5.5 +/- 0.2 for short-period dwarf novae
        from Patterson 2011 (MNRAS 411, 2695; eq. 6).  This gives quiescent
        M_V = 9.8 +/- 1.1 (SU UMa) and 11.9 +/- 0.7 (WZ Sge).  The WZ Sge
        value is consistent with the Gaia-based quiescent absolute
        magnitudes of the WZ Sge prototypes used by Buckley et al. 2025
        (M_r = 11.4-12.0) and with the mean M_g = 11.6 +/- 0.7 for SDSS
        CVs at the period minimum (Gaensicke et al. 2009, MNRAS 397, 2170).

        For long-period (U Gem/Z Cam) systems we use the quiescent
        M_V - P_orb relation from the CRTS outburst catalogue of Coppejans
        et al. 2016 (MNRAS 456, 4441; eq. 3), M_V = 12.7 - 6.6 log10(P_orb/h),
        with 0.8 mag of scatter, which gives M_V ~ 8 at P_orb = 5 h
        (cf. U Gem itself, M_V ~ 9.4, and SS Cyg, M_V ~ 6.7, using the Gaia
        distances of Pala et al. 2020).  See also Abrahams et al. 2022
        (ApJ 938, 46) for the Gaia period-color-absolute magnitude sequence
        of quiescent CVs.

        Parameters
        ----------
        subtypes : `np.ndarray` [`str`]
            Subtype of each system.
        porb : `np.ndarray` [`float`]
            Orbital periods in hours.

        Returns
        -------
        mv_quiescent : `np.ndarray` [`float`]
            Quiescent absolute V magnitudes.
        """
        n = len(subtypes)
        mv = np.zeros(n)
        w = subtypes == "wzsge"
        mv[w] = self.rng.normal(5.5 + 6.4, 0.7, size=np.sum(w))
        w = subtypes == "suuma"
        mv[w] = self.rng.normal(5.5 + 4.3, 1.1, size=np.sum(w))
        w = subtypes == "ugem"
        mv[w] = 12.7 - 6.6 * np.log10(porb[w]) + self.rng.normal(0, 0.8, size=np.sum(w))
        return mv

    def peak_abs_mag(self, porb, kind, size):
        """Draw absolute V magnitudes at outburst maximum.

        Uses the M_V(max) - P_orb relations of Patterson 2011
        (MNRAS 411, 2695), which update Warner 1987 (MNRAS 227, 23):
        for superoutbursts, M_V(max) = 4.95 - 0.199 P_orb(h) with an rms
        scatter of 0.41 mag (eq. 4), and for normal outbursts
        M_V(max) = 5.70 - 0.287 P_orb(h) with an rms scatter of 0.47 mag
        (eq. 3).  Patterson's relations are calibrated with an average
        binary inclination; the scatter here also absorbs the inclination
        dependence.

        Parameters
        ----------
        porb : `float`
            Orbital period in hours.
        kind : `str`
            "super" or "normal".
        size : `int`
            Number of outbursts.

        Returns
        -------
        mv_peak : `np.ndarray` [`float`]
            Absolute V magnitudes at maximum.
        """
        if kind == "super":
            return 4.95 - 0.199 * porb + self.rng.normal(0, 0.41, size=size)
        return 5.70 - 0.287 * porb + self.rng.normal(0, 0.47, size=size)

    @staticmethod
    def bailey_decline_rate(porb):
        """Decline rate (days per magnitude) at the end of an outburst.

        The Bailey (1975) relation as given by Warner 1995 (Cataclysmic
        Variable Stars, eq. 3.5): tau_d = 0.53 P_orb(h)^0.84 d/mag, i.e.
        ~0.8 d/mag below the period gap and ~2 d/mag at 5 h.  Otulakowska-
        Hypka et al. 2016 confirm a relation of this form for long outbursts
        above the period gap.

        Parameters
        ----------
        porb : `float`
            Orbital period in hours.

        Returns
        -------
        tau_d : `float`
            Decline timescale in days per magnitude.
        """
        return 0.53 * porb**0.84

    def outburst_schedule(self, subtype, porb):
        """Generate the sequence of outbursts for a single dwarf nova.

        Recurrence times and durations are drawn from log-normal
        distributions matched to the catalogue of Otulakowska-Hypka, Olech
        & Patterson 2016 (MNRAS 460, 2526; VizieR J/MNRAS/460/2526), and
        for WZ Sge stars to the review of Kato 2015 (PASJ 67, 108):

        * WZ Sge: superoutbursts only, with a median recurrence time of
          10 yr (Kato 2015 finds a median observed interval of 11.5 yr,
          biased towards shorter intervals; the shortest known is 450 d)
          and 0.35 dex scatter.  The superoutburst plateau lasts 20 d
          (interquartile range 13.5-22 d) before the final decline,
          consistent with the "weeks" durations in Kato 2015.
          Post-outburst rebrightenings are not modelled (cf. Buckley et al.
          2025).
        * SU UMa: supercycle length with median 280 d and 0.34 dex scatter;
          normal outburst cycle with median 35 d and 0.35 dex scatter (the
          catalogue gives minimal and maximal cycle lengths of 25 and 40 d).
          Superoutburst plateaus last 10 d (9-12 d) and normal outbursts
          last 4 d (3-5 d) in total.  Superoutbursts are ~0.8 mag brighter
          than normal outbursts (Patterson 2011; Otulakowska-Hypka et al.
          2016).
        * U Gem/Z Cam: normal outbursts with median cycle 45 d and 0.3 dex
          scatter (catalogue minimal/maximal cycle lengths 30 and 60 d) and
          total duration 9 d (6-13 d).

        Successive intervals are perturbed by 0.1 dex to make the outbursts
        quasi-periodic.  The implied duty cycles (~0.5% for WZ Sge over a
        decade, ~15% for SU UMa, ~20% for U Gem) bracket the CRTS duty
        cycles of Coppejans et al. 2016 (median 6-8%, rising with P_orb)
        and the measured duty cycles of nearby dwarf novae in Britt et al.
        2015 (MNRAS 448, 3455; 0.5% for GW Lib to 30% for SS Cyg).

        The rise to maximum is fast (1 d for normal outbursts, 2 d for
        superoutbursts; Otulakowska-Hypka et al. 2016 find rise durations
        of 1-2 d and Buckley et al. 2025 adopt <0.2 d/mag for WZ Sge
        stars).  Superoutburst plateaus decline at 0.12 mag/day
        (Patterson 2011, Section 9) and are followed by a decline at the
        Bailey rate.  For normal outbursts the catalogue duration is the
        total outburst length; the final decline is at the Bailey rate
        where that fits within the duration (long-period systems) and is
        compressed to fit otherwise (short-period systems, whose normal
        outbursts are short and spiky).

        Parameters
        ----------
        subtype : `str`
            One of "wzsge", "suuma", "ugem".
        porb : `float`
            Orbital period in hours.

        Returns
        -------
        schedule : `dict`
            Arrays describing each outburst: ``t_start`` (days relative to
            survey start), ``kind`` ("super"/"normal"), ``mv_peak``,
            ``t_rise``, ``duration`` (plateau length for superoutbursts,
            total length for normal outbursts) and ``plateau_slope``
            (mag/day).
        """
        rng = self.rng
        t_max = self.survey_length

        def lognormal(median, sigma_dex, size=1):
            return 10 ** rng.normal(np.log10(median), sigma_dex, size=size)

        starts = []
        kinds = []
        if subtype == "wzsge":
            # Random phase within the first cycle, then march forward.
            t = -rng.uniform() * lognormal(3650.0, 0.35)[0]
            while t < t_max:
                if t > -60:
                    starts.append(t)
                    kinds.append("super")
                t += lognormal(3650.0, 0.35)[0]
            dur_super = lognormal(20.0, 0.12, size=max(len(starts), 1))
            dur_normal = None
        elif subtype == "suuma":
            p_sc = lognormal(280.0, 0.34)[0]
            p_c = max(lognormal(35.0, 0.35)[0], 10.0)
            p_c = min(p_c, p_sc / 2.0)
            t = -rng.uniform() * p_sc
            while t < t_max:
                if t > -60:
                    starts.append(t)
                    kinds.append("super")
                t_next_super = t + p_sc * 10 ** rng.normal(0, 0.1)
                # normal outbursts between superoutbursts
                tn = t + 15.0 + p_c * 10 ** rng.normal(0, 0.1)
                while tn < t_next_super - 10.0:
                    if tn > -60:
                        starts.append(tn)
                        kinds.append("normal")
                    tn += p_c * 10 ** rng.normal(0, 0.1)
                t = t_next_super
            dur_super = lognormal(10.0, 0.08, size=max(len(starts), 1))
            dur_normal = lognormal(4.0, 0.12, size=max(len(starts), 1))
        elif subtype == "ugem":
            p_c = max(lognormal(45.0, 0.3)[0], 12.0)
            t = -rng.uniform() * p_c
            while t < t_max:
                if t > -60:
                    starts.append(t)
                    kinds.append("normal")
                t += p_c * 10 ** rng.normal(0, 0.1)
            dur_super = None
            dur_normal = lognormal(9.0, 0.2, size=max(len(starts), 1))
        else:
            raise ValueError(f"Unknown dwarf nova subtype {subtype}")

        starts = np.array(starts, dtype=float)
        kinds = np.array(kinds, dtype="U6")
        n_ob = len(starts)
        order = np.argsort(starts)
        starts = starts[order]
        kinds = kinds[order]

        mv_peak = np.zeros(n_ob)
        t_rise = np.zeros(n_ob)
        duration = np.zeros(n_ob)
        slope = np.zeros(n_ob)
        w_super = kinds == "super"
        w_normal = ~w_super
        if np.sum(w_super) > 0:
            mv_peak[w_super] = self.peak_abs_mag(porb, "super", np.sum(w_super))
            t_rise[w_super] = 2.0
            duration[w_super] = dur_super[: np.sum(w_super)]
            slope[w_super] = 0.12
        if np.sum(w_normal) > 0:
            mv_peak[w_normal] = self.peak_abs_mag(porb, "normal", np.sum(w_normal))
            t_rise[w_normal] = 1.0
            duration[w_normal] = dur_normal[: np.sum(w_normal)]
            slope[w_normal] = 0.0

        return {
            "t_start": starts,
            "kind": kinds,
            "mv_peak": mv_peak,
            "t_rise": t_rise,
            "duration": duration,
            "plateau_slope": slope,
        }

    def outburst_params(self, size=1, subtype_fractions=None):
        """Return parameters at random characterizing each dwarf nova.

        Parameters
        ----------
        size : `int`, opt
            Number of systems to generate.
        subtype_fractions : `dict` [`str`, `float`], opt
            Fraction of systems of each subtype ("wzsge", "suuma",
            "ugem").  Default `DN_SUBTYPE_FRACTIONS`.

        Returns
        -------
        params : `list` [`dict`]
            Per-system parameters: ``subtype``, ``orbital_period`` (hours),
            ``mv_quiescent``, ``abs_mag_quiescent`` and
            ``abs_mag_outburst_offset`` (per-filter dicts), the outburst
            schedule arrays (see `outburst_schedule`), plus per-outburst
            ``t_plateau`` (days), ``tau_decline`` (days/mag) and ``t_end``
            (time at which each outburst template returns to the quiescent
            level).
        """
        if subtype_fractions is None:
            subtype_fractions = DN_SUBTYPE_FRACTIONS
        names = list(subtype_fractions.keys())
        probs = np.array([subtype_fractions[k] for k in names], dtype=float)
        probs /= probs.sum()
        subtypes = self.rng.choice(np.array(names), size=size, p=probs)
        porbs = self.orbital_periods(subtypes)
        mv_q = self.quiescent_abs_mags(subtypes, porbs)
        outburst_offsets = self.outburst_colors()

        params = []
        for subtype, porb, mvq in zip(subtypes, porbs, mv_q):
            sched = self.outburst_schedule(subtype, porb)
            # Ensure every outburst is at least 1.5 mag brighter than
            # quiescence (the amplitude cut of outburst catalogues).
            sched["mv_peak"] = np.minimum(sched["mv_peak"], mvq - 1.5)
            amplitude = mvq - sched["mv_peak"]
            # Plateau length and final decline rate.
            # Superoutbursts: the catalogue duration is the plateau, and
            # the decline follows at the Bailey rate.
            # Normal outbursts: the catalogue duration is the total length;
            # fit the Bailey decline inside it if possible.
            tau_bailey = self.bailey_decline_rate(porb)
            w_super = sched["kind"] == "super"
            t_plateau = np.where(w_super, sched["duration"], 0.0)
            tau_decline = np.ones(len(amplitude)) * tau_bailey
            if np.any(~w_super):
                t_avail = sched["duration"][~w_super] - sched["t_rise"][~w_super]
                t_decline = amplitude[~w_super] * tau_bailey
                plat = np.maximum(t_avail - t_decline, 0.5)
                t_plateau[~w_super] = plat
                tau_decline[~w_super] = np.maximum(t_avail - plat, 0.2) / amplitude[~w_super]
            sched["t_plateau"] = t_plateau
            sched["tau_decline"] = tau_decline
            t_end = (
                sched["t_start"]
                + sched["t_rise"]
                + sched["t_plateau"]
                + np.maximum(amplitude - sched["plateau_slope"] * sched["t_plateau"], 0.0)
                * sched["tau_decline"]
            )
            qcol = self.quiescent_colors(subtype)
            p = {
                "subtype": subtype,
                "orbital_period": porb,
                "mv_quiescent": mvq,
                "abs_mag_quiescent": {f: mvq + qcol[f] for f in "ugrizy"},
                "abs_mag_outburst_offset": outburst_offsets,
                "t_end": t_end,
            }
            p.update(sched)
            params.append(p)
        return params

    # ------------------------------------------------------------------
    # Light curves
    # ------------------------------------------------------------------
    def outburst_template(self, t, params):
        """Evaluate the outburst-only V-band absolute magnitude at times t.

        Parameters
        ----------
        t : `np.ndarray` [`float`]
            Times in days relative to the survey start.
        params : `dict`
            Parameters for one system from `outburst_params`.

        Returns
        -------
        mv : `np.ndarray` [`float`]
            Absolute V magnitude of the outburst component; 99 when
            not in outburst.
        """
        mv = np.ones(len(t)) * 99.0
        t_start = params["t_start"]
        if len(t_start) == 0:
            return mv
        # Index of the most recent outburst to have started before t
        idx = np.searchsorted(t_start, t, side="right") - 1
        valid = idx >= 0
        if not np.any(valid):
            return mv
        idx = idx[valid]
        tt = t[valid] - t_start[idx]
        peak = params["mv_peak"][idx]
        t_rise = params["t_rise"][idx]
        t_plat = params["t_plateau"][idx]
        slope = params["plateau_slope"][idx]
        tau = params["tau_decline"][idx]
        mq = params["mv_quiescent"]
        amp = mq - peak

        m = np.ones(len(tt)) * 99.0
        # rise: linear in magnitude from quiescence to peak
        w = (tt >= 0) & (tt < t_rise)
        m[w] = mq - amp[w] * tt[w] / t_rise[w]
        # plateau: slow linear decline
        w = (tt >= t_rise) & (tt < t_rise + t_plat)
        m[w] = peak[w] + slope[w] * (tt[w] - t_rise[w])
        # final decline at the Bailey rate
        w = tt >= t_rise + t_plat
        m_end_plat = peak[w] + slope[w] * t_plat[w]
        m[w] = m_end_plat + (tt[w] - t_rise[w] - t_plat[w]) / tau[w]
        m = np.where(m > mq, 99.0, m)
        mv[valid] = m
        return mv

    def in_outburst(self, t, params):
        """Return the index of the outburst in progress at each time.

        Parameters
        ----------
        t : `np.ndarray` [`float`]
            Times in days relative to the survey start.
        params : `dict`
            Parameters for one system from `outburst_params`.

        Returns
        -------
        idx : `np.ndarray` [`int`]
            Index into the outburst arrays, or -1 when in quiescence.
        """
        idx = np.ones(len(t), dtype=int) * -1
        t_start = params["t_start"]
        if len(t_start) == 0:
            return idx
        i = np.searchsorted(t_start, t, side="right") - 1
        valid = i >= 0
        active = np.zeros(len(t), dtype=bool)
        active[valid] = t[valid] <= params["t_end"][i[valid]]
        idx[active] = i[active]
        return idx

    def lightcurve(self, t, filtername, params):
        """Generate a dwarf nova lightcurve for given times and a filter.

        Parameters
        ----------
        t : `np.ndarray` [`float`]
            Times in days relative to the survey start.
        filtername : `str`
            The filter, one of ugrizy.
        params : `dict`
            Parameters for one system from `outburst_params`.

        Returns
        -------
        lc : `np.ndarray` [`float`]
            Absolute magnitudes in the given filter at the specified times
            (quiescent plus outburst flux).
        """
        mq = params["abs_mag_quiescent"][filtername]
        mob = self.outburst_template(t, params) + params["abs_mag_outburst_offset"][filtername]
        flux = 10 ** (-0.4 * mq) + 10 ** (-0.4 * mob)
        return -2.5 * np.log10(flux)

    def detectable_duration(self, params, ebv, distance, t_start=0.0, t_end=None):
        """Determine the time an object spends above the fiducial LSST
        single-visit depth in outburst, with perfect sampling.

        Does not consider visibility constraints.

        Parameters
        ----------
        params : `dict`
            Parameters for one system from `outburst_params`.
        ebv : `float`
            E(B-V)
        distance : `float`
            Distance in kpc
        t_start, t_end : `float`, opt
            Survey window (days) within which to count outbursts.

        Returns
        -------
        visible_start_time : `float`
            First time (days, relative to survey start) at which an outburst
            could be detected; nan if never.
        visible_duration : `float`
            Total time (days) spent above the single-visit depth in any
            filter during the survey window; 0 if never.
        n_outbursts : `int`
            Number of outbursts starting within the survey window.
        """
        if t_end is None:
            t_end = self.survey_length
        distmod = 5 * np.log10(distance * 1.0e3) - 5.0
        w = (params["t_start"] >= t_start) & (params["t_start"] <= t_end)
        n_outbursts = int(np.sum(w))
        if n_outbursts == 0:
            return np.nan, 0.0, 0
        dt = 0.1
        visible_start = np.inf
        visible_duration = 0.0
        for i in np.where(w)[0]:
            t = np.arange(params["t_start"][i], params["t_end"][i] + dt, dt)
            detectable = np.zeros(len(t), dtype=bool)
            for filtername in "ugrizy":
                mags = self.lightcurve(t, filtername, params) + self.ax1[filtername] * ebv + distmod
                detectable |= mags <= LSST_SINGLE_EPOCH_DEPTH[filtername]
            if np.any(detectable):
                visible_duration += np.sum(detectable) * dt
                visible_start = min(visible_start, t[np.where(detectable)[0][0]])
        if np.isinf(visible_start):
            return np.nan, 0.0, n_outbursts
        return visible_start, visible_duration, n_outbursts


class CVPopMetric(BaseMetric):
    """Evaluate whether a given dwarf nova (outbursting cataclysmic
    variable) would be detectable, in analogy to `XRBPopMetric`.

    Includes a variety of detection criteria options: whether an outburst
    is ever bright enough to detect, whether it is detected at least
    pts_needed times, whether it is detected pts_early times within
    t_early days of the start of an outburst, how many of the outbursts in
    the survey are detected, and whether the quiescent counterpart is also
    detected (allowing the outburst amplitude to be measured).

    Parameters
    ----------
    pts_needed : `int`, opt
        Minimum number of detections in outburst, for the `ever_detect`
        and `n_outbursts_detected` options.
    pts_early : `int`, opt
        Number of detections required within t_early days of the start of
        an outburst for the `early_detect` option.
    t_early : `float`, opt
        Window (days) after the start of an outburst for `early_detect`.
    pts_quiescent : `int`, opt
        Minimum number of detections in quiescence for the
        `quiescence_detected` option.
    mjd0 : `float`, opt
        Start of survey.
    output_lc : `bool`, opt
        If True, output lightcurve points.
        If False, just return metric values.
    """

    def __init__(
        self,
        metric_name="CVPopMetric",
        mjd_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        filter_col="band",
        night_col="night",
        pts_needed=2,
        pts_early=2,
        t_early=7,
        pts_quiescent=2,
        mjd0=SURVEY_START_MJD,
        output_lc=False,
        badval=-666,
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.night_col = night_col
        self.pts_needed = pts_needed
        self.pts_early = pts_early
        self.t_early = t_early
        self.pts_quiescent = pts_quiescent
        self.output_lc = output_lc

        self.lightcurves = CvLc()
        self.mjd0 = mjd0

        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1

        cols = [self.mjd_col, self.m5_col, self.filter_col, self.night_col]
        super(CVPopMetric, self).__init__(
            col=cols,
            units="Detected, 0 or 1",
            metric_name=metric_name,
            badval=badval,
            **kwargs,
        )
        self.comment = "Number or characterization of dwarf novae."

    def _ever_detect(self, where_detected):
        """Simple detection criteria: detect at least a certain number
        of times in outburst.
        """
        return np.size(where_detected) >= self.pts_needed

    def _number_of_detections(self, where_detected):
        """Count total number of detections in outburst."""
        return len(where_detected)

    def _early_detect(self, t_since_start, outburst_idx, early_window_days=7.0, n_early_detections=2):
        """Detection near the start of any outburst.

        Parameters
        ----------
        t_since_start : `np.ndarray`
            Time since the start of the outburst in progress, for each
            detected point in outburst.
        outburst_idx : `np.ndarray`
            Index of the outburst in progress for each detected point.
        early_window_days : `float`
            Time since start of outburst.
        n_early_detections : `int`
            Number of required early detections.
        """
        if len(outburst_idx) == 0:
            return False
        early = t_since_start <= early_window_days
        if np.sum(early) < n_early_detections:
            return False
        _, counts = np.unique(outburst_idx[early], return_counts=True)
        return bool(np.max(counts) >= n_early_detections)

    def _n_outbursts_detected(self, outburst_idx):
        """Count the number of distinct outbursts with at least
        pts_needed detections.
        """
        if len(outburst_idx) == 0:
            return 0
        _, counts = np.unique(outburst_idx, return_counts=True)
        return int(np.sum(counts >= self.pts_needed))

    def _possible_to_detect(self, visible_duration):
        """Return True if an outburst during the survey is ever bright
        enough for LSST to detect.
        """
        return visible_duration > 0

    def run(self, data_slice, slice_point=None):
        result = {}
        params = slice_point["outburst_params"]
        t = data_slice[self.mjd_col] - self.mjd0
        mags = np.zeros(t.size, dtype=float)

        distmod = 5 * np.log10(slice_point["distance"] * 1.0e3) - 5.0
        for filtername in np.unique(data_slice[self.filter_col]):
            infilt = np.where(data_slice[self.filter_col] == filtername)
            mags[infilt] = self.lightcurves.lightcurve(t[infilt], filtername, params)
            # Apply dust extinction on the light curve
            a_x = self.ax1[filtername] * slice_point["ebv"]
            mags[infilt] += a_x
            mags[infilt] += distmod

        outburst_idx = self.lightcurves.in_outburst(t, params)
        in_outburst = outburst_idx >= 0

        detected = mags < data_slice[self.m5_col]
        where_detected = np.where(detected & in_outburst)[0]
        where_quiescent = np.where(detected & ~in_outburst)[0]
        # Magnitude uncertainties with Gaussian approximation
        snr = m52snr(mags, data_slice[self.m5_col])
        mags_unc = 2.5 * np.log10(1.0 + 1.0 / snr)

        det_idx = outburst_idx[where_detected]
        t_since_start = t[where_detected] - params["t_start"][det_idx] if len(det_idx) > 0 else np.array([])

        result["possible_to_detect"] = self._possible_to_detect(slice_point["visible_duration"])
        result["ever_detect"] = self._ever_detect(where_detected)
        result["early_detect"] = self._early_detect(t_since_start, det_idx, self.t_early, self.pts_early)
        result["number_of_detections"] = self._number_of_detections(where_detected)
        result["n_outbursts"] = int(slice_point["n_outbursts"])
        result["n_outbursts_detected"] = self._n_outbursts_detected(det_idx)
        result["quiescence_detected"] = len(where_quiescent) >= self.pts_quiescent
        result["outburst_and_quiescence"] = result["ever_detect"] and result["quiescence_detected"]

        # Export the light curve
        if self.output_lc is True:
            mags_out = mags.copy()
            mags_out[~detected] = 99.0
            result["lc"] = [
                data_slice[self.mjd_col],
                mags_out,
                mags_unc,
                data_slice[self.m5_col],
                data_slice[self.filter_col],
                in_outburst,
            ]
            result["lc_colnames"] = ("t", "mag", "mag_unc", "maglim", "filter", "in_outburst")

        return result

    def reduce_possible_to_detect(self, metric):
        return metric["possible_to_detect"]

    def reduce_ever_detect(self, metric):
        return metric["ever_detect"]

    def reduce_early_detect(self, metric):
        return metric["early_detect"]

    def reduce_number_of_detections(self, metric):
        return metric["number_of_detections"]

    def reduce_n_outbursts(self, metric):
        return metric["n_outbursts"]

    def reduce_n_outbursts_detected(self, metric):
        return metric["n_outbursts_detected"]

    def reduce_fraction_outbursts_detected(self, metric):
        if metric["n_outbursts"] == 0:
            return self.badval
        return metric["n_outbursts_detected"] / metric["n_outbursts"]

    def reduce_quiescence_detected(self, metric):
        return metric["quiescence_detected"]

    def reduce_outburst_and_quiescence(self, metric):
        return metric["outburst_and_quiescence"]


def sample_dwarf_nova_positions(scale_heights_pc, bulge_fraction=0.3, seed=42, galactic_structure=None):
    """Sample Galactic positions for a population of dwarf novae.

    The population is a mix of an exponential thin disk and a boxy bulge.
    The disk has an exponential radial profile with the thin disk scale
    length of Bland-Hawthorn & Gerhard 2016 (ARA&A 54, 529; 2.6 kpc) and an
    exponential vertical profile with a per-object scale height (set by
    subtype, see `DN_SCALE_HEIGHTS`), as in the CV Galactic models of
    Pretorius et al. 2007, Pala et al. 2020 and Buckley et al. 2025.
    No radial dependence of the scale height is included.

    The bulge is modelled as a triaxial exponential with the scale lengths
    (0.70, 0.44, 0.18 kpc) and bar angle (27 deg) of Bland-Hawthorn &
    Gerhard 2016.  The Galactic Bulge Survey found a specific frequency of
    outbursting dwarf novae towards the bulge consistent with the local
    space density per unit stellar mass (Britt et al. 2015, MNRAS 448,
    3455), so the default bulge fraction is the bulge's share of the
    Galactic stellar mass, 0.3 (Bland-Hawthorn & Gerhard 2016).  The thick
    disk and halo are not modelled.

    Parameters
    ----------
    scale_heights_pc : `np.ndarray` [`float`]
        Vertical scale height (pc) for each object; the length sets the
        number of objects.
    bulge_fraction : `float`, opt
        Fraction of the population placed in the bulge.
    seed : `int`, opt
        Random seed.
    galactic_structure : `dict`, opt
        Override values of `GALACTIC_STRUCTURE`.

    Returns
    -------
    ra, dec : `np.ndarray` [`float`]
        ICRS coordinates in degrees.
    distance : `np.ndarray` [`float`]
        Heliocentric distance in kpc.
    in_bulge : `np.ndarray` [`bool`]
        True for bulge members.
    """
    gs = dict(GALACTIC_STRUCTURE)
    if galactic_structure is not None:
        gs.update(galactic_structure)
    rng = np.random.default_rng(seed)
    n = len(scale_heights_pc)
    in_bulge = rng.uniform(size=n) < bulge_fraction

    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)

    # Disk: surface density ~ R exp(-R/h) dR -> Gamma(2, h)
    n_disk = int(np.sum(~in_bulge))
    h_r = gs["disk_scale_length_kpc"]
    r = rng.gamma(2.0, h_r, size=n_disk)
    too_far = r > gs["disk_max_radius_kpc"]
    while np.any(too_far):
        r[too_far] = rng.gamma(2.0, h_r, size=np.sum(too_far))
        too_far = r > gs["disk_max_radius_kpc"]
    phi = rng.uniform(0, 2 * np.pi, size=n_disk)
    x[~in_bulge] = r * np.cos(phi)
    y[~in_bulge] = r * np.sin(phi)
    z[~in_bulge] = rng.laplace(0.0, scale_heights_pc[~in_bulge] / 1000.0)

    # Bulge: triaxial exponential, major axis rotated by the bar angle
    n_bulge = int(np.sum(in_bulge))
    if n_bulge > 0:
        sx, sy, sz = gs["bulge_scales_kpc"]
        bx = rng.laplace(0.0, sx, size=n_bulge)
        by = rng.laplace(0.0, sy, size=n_bulge)
        bz = rng.laplace(0.0, sz, size=n_bulge)
        alpha = np.radians(gs["bar_angle_deg"])
        # The near end of the bar lies at positive Galactic longitude (+y)
        # and on the Sun's side of the Galactic centre (-x).
        x[in_bulge] = -bx * np.cos(alpha) - by * np.sin(alpha)
        y[in_bulge] = bx * np.sin(alpha) - by * np.cos(alpha)
        z[in_bulge] = bz

    gc = Galactocentric(
        x=x * u.kpc,
        y=y * u.kpc,
        z=z * u.kpc,
        galcen_distance=gs["r_sun_kpc"] * u.kpc,
        z_sun=gs["z_sun_pc"] * u.pc,
    )
    icrs = gc.transform_to(ICRS())
    return icrs.ra.deg, icrs.dec.deg, icrs.distance.kpc, in_bulge


def expected_number_of_dwarf_novae(
    rho0=5.0e-6,
    subtype_fractions=None,
    scale_heights=None,
    bulge_fraction=0.3,
    galactic_structure=None,
):
    """Estimate the total number of dwarf novae in the Galaxy for the
    population model used by `generate_cv_pop_slicer`.

    The disk count is the analytic integral of the exponential disk,
    N = sum_subtype f_s rho0 (2 h_s) 2 pi h_R^2 exp(R_sun / h_R), where
    rho0 is the local mid-plane space density.  The bulge adds
    N_disk bulge_fraction / (1 - bulge_fraction).  This can be used to
    scale the detection fractions from the metric to absolute numbers.

    The default local dwarf nova space density of 5e-6 pc^-3 is the sum of
    the U Gem, SU UMa and WZ Sge space densities in Inight et al. 2023
    (MNRAS 524, 4867; Table 4).  The Gaia DR2 150 pc volume-limited sample
    of Pala et al. 2020 (MNRAS 494, 3799) gives a lower total CV density of
    4.8e-6 pc^-3, of which ~52 per cent are dwarf novae, so the true value
    may be a factor of ~2 lower.  Both are well below the 1e-5 - 1e-4 pc^-3
    predicted by population synthesis (see Canbay et al. 2023, AJ 165, 163).

    Parameters
    ----------
    rho0 : `float`, opt
        Local mid-plane space density of dwarf novae (pc^-3).
    subtype_fractions : `dict`, opt
        See `DN_SUBTYPE_FRACTIONS`.
    scale_heights : `dict`, opt
        See `DN_SCALE_HEIGHTS`.
    bulge_fraction : `float`, opt
        Fraction of the population in the bulge.
    galactic_structure : `dict`, opt
        Override values of `GALACTIC_STRUCTURE`.

    Returns
    -------
    n_total : `float`
        Expected number of dwarf novae in the Galaxy.
    """
    if subtype_fractions is None:
        subtype_fractions = DN_SUBTYPE_FRACTIONS
    if scale_heights is None:
        scale_heights = DN_SCALE_HEIGHTS
    gs = dict(GALACTIC_STRUCTURE)
    if galactic_structure is not None:
        gs.update(galactic_structure)
    h_r = gs["disk_scale_length_kpc"] * 1000.0
    r_sun = gs["r_sun_kpc"] * 1000.0
    norm = sum(subtype_fractions.values())
    n_disk = 0.0
    for key, frac in subtype_fractions.items():
        n_disk += (frac / norm) * rho0 * 2 * scale_heights[key] * 2 * np.pi * h_r**2 * np.exp(r_sun / h_r)
    return n_disk / (1.0 - bulge_fraction)


def generate_cv_pop_slicer(
    t_start=1,
    t_end=3652,
    n_events=10000,
    seed=42,
    subtype_fractions=None,
    scale_heights=None,
    bulge_fraction=0.3,
    nside=64,
):
    """Generate a population of dwarf novae, and put the info about them
    into a UserPointSlicer object.

    Positions are drawn with `sample_dwarf_nova_positions`, E(B-V) at
    each position and distance is interpolated from the rubin_sim 3-d dust
    map, and the outburst history of each system over the survey is
    generated with `CvLc.outburst_params`.  Unlike `generate_xrb_pop_slicer`,
    each system may have zero, one or many outbursts during the survey.

    Parameters
    ----------
    t_start : `float`, opt
        The first night of the survey window (days; default 1).
    t_end : `float`, opt
        The final night of the survey window (days; default 3652).
    n_events : `int`, opt
        The number of dwarf novae to generate (default 10000).
    seed : `float`, opt
        The seed passed to np.random (default 42).
    subtype_fractions : `dict`, opt
        See `DN_SUBTYPE_FRACTIONS`.
    scale_heights : `dict`, opt
        See `DN_SCALE_HEIGHTS`.
    bulge_fraction : `float`, opt
        Fraction of the population in the Galactic bulge (default 0.3).
    nside : `int`, opt
        Healpix nside of the 3-d dust map to use (64 or 128).

    Returns
    -------
    slicer : `UserPointsSlicer`
        Slicer with slice_point entries ``distance`` (kpc), ``ebv``,
        ``subtype``, ``orbital_period`` (h), ``in_bulge``,
        ``outburst_params``, ``visible_start_time``, ``visible_duration``
        and ``n_outbursts``.
    """
    if subtype_fractions is None:
        subtype_fractions = DN_SUBTYPE_FRACTIONS
    if scale_heights is None:
        scale_heights = DN_SCALE_HEIGHTS

    cv_lc_gen = CvLc(seed=seed, survey_length=t_end)
    params = cv_lc_gen.outburst_params(size=n_events, subtype_fractions=subtype_fractions)
    subtypes = np.array([p["subtype"] for p in params])
    porbs = np.array([p["orbital_period"] for p in params])

    heights = np.array([scale_heights[s] for s in subtypes])
    ra, dec, distance_kpc, in_bulge = sample_dwarf_nova_positions(
        heights, bulge_fraction=bulge_fraction, seed=seed
    )

    # 3-d dust extinction at each position and distance
    dists, ebvs = ebv_3d_hp(nside, ra=np.radians(ra), dec=np.radians(dec), interp=False)
    ebv = np.array(
        [np.interp(d * 1000.0, dists[i], ebvs[i]) for i, d in enumerate(distance_kpc)],
        dtype=float,
    )

    # Set up the slicer to evaluate the catalog we just made
    slicer = UserPointsSlicer(ra, dec, lat_lon_deg=True, badval=0)
    slicer.slice_points["distance"] = distance_kpc
    slicer.slice_points["ebv"] = ebv
    slicer.slice_points["subtype"] = subtypes
    slicer.slice_points["orbital_period"] = porbs
    slicer.slice_points["in_bulge"] = in_bulge
    slicer.slice_points["outburst_params"] = params

    # determine detectable durations
    visible_starts = []
    visible_durations = []
    n_outbursts = []
    for idx, param in enumerate(params):
        start, duration, n_ob = cv_lc_gen.detectable_duration(
            param, ebv[idx], distance_kpc[idx], t_start=t_start, t_end=t_end
        )
        visible_starts.append(start)
        visible_durations.append(duration)
        n_outbursts.append(n_ob)

    slicer.slice_points["visible_start_time"] = np.array(visible_starts)
    slicer.slice_points["visible_duration"] = np.array(visible_durations)
    slicer.slice_points["n_outbursts"] = np.array(n_outbursts)

    return slicer
