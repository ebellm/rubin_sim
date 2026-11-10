import os
import unittest

import numpy as np
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import SURVEY_START_MJD

from rubin_sim.maf.maf_contrib import (
    DN_SCALE_HEIGHTS,
    DN_SUBTYPE_FRACTIONS,
    CvLc,
    CVPopMetric,
    expected_number_of_dwarf_novae,
    generate_cv_pop_slicer,
    sample_dwarf_nova_positions,
)


class TestCvLc(unittest.TestCase):
    def setUp(self):
        self.lc = CvLc(seed=7, survey_length=3652.0)
        self.params = self.lc.outburst_params(size=300)

    def test_subtypes_and_periods(self):
        subtypes = np.array([p["subtype"] for p in self.params])
        for key in DN_SUBTYPE_FRACTIONS:
            self.assertIn(key, subtypes)
        porb = np.array([p["orbital_period"] for p in self.params])
        self.assertTrue(np.all(porb[subtypes == "wzsge"] < 2.1))
        self.assertTrue(np.all(porb[subtypes == "suuma"] < 2.2))
        self.assertTrue(np.all(porb[subtypes == "ugem"] >= 3.0))

    def test_quiescent_and_peak_magnitudes(self):
        for p in self.params:
            # Every outburst is at least 1.5 mag brighter than quiescence
            self.assertTrue(np.all(p["mv_quiescent"] - p["mv_peak"] >= 1.5))
            # Outbursts are time ordered and end after they start
            self.assertTrue(np.all(np.diff(p["t_start"]) > 0))
            self.assertTrue(np.all(p["t_end"] > p["t_start"]))
        mq_wz = np.array([p["mv_quiescent"] for p in self.params if p["subtype"] == "wzsge"])
        mq_ug = np.array([p["mv_quiescent"] for p in self.params if p["subtype"] == "ugem"])
        # WZ Sge stars are much fainter in quiescence than U Gem stars
        self.assertGreater(np.median(mq_wz), np.median(mq_ug) + 2)

    def test_recurrence(self):
        n_wz = [np.sum(p["t_start"] >= 0) for p in self.params if p["subtype"] == "wzsge"]
        n_su = [np.sum(p["t_start"] >= 0) for p in self.params if p["subtype"] == "suuma"]
        # WZ Sge stars have ~1 outburst per decade; SU UMa stars have many
        self.assertLess(np.mean(n_wz), 3)
        self.assertGreater(np.mean(n_su), 20)
        for p in self.params:
            if p["subtype"] == "wzsge":
                self.assertTrue(np.all(p["kind"] == "super"))
            if p["subtype"] == "ugem":
                self.assertTrue(np.all(p["kind"] == "normal"))

    def test_lightcurve(self):
        p = [p for p in self.params if p["subtype"] == "suuma" and len(p["t_start"]) > 2][0]
        i = 1
        t0 = p["t_start"][i]
        t = np.array([t0 - 1.0, t0 + p["t_rise"][i], p["t_end"][i] + 5.0])
        for f in "ugrizy":
            mags = self.lc.lightcurve(t, f, p)
            quiescent = p["abs_mag_quiescent"][f]
            # before the outburst and well after it: quiescent
            self.assertAlmostEqual(mags[0], quiescent, places=3)
            self.assertAlmostEqual(mags[2], quiescent, places=3)
            # at peak: brighter than quiescence by roughly the amplitude
            expected_peak = p["mv_peak"][i] + p["abs_mag_outburst_offset"][f]
            self.assertLess(mags[1], quiescent - 1.0)
            self.assertAlmostEqual(mags[1], expected_peak, delta=0.3)
        idx = self.lc.in_outburst(t, p)
        self.assertEqual(idx[0], i - 1 if t[0] <= p["t_end"][i - 1] else -1)
        self.assertEqual(idx[1], i)
        self.assertEqual(idx[2], -1 if t[2] < p["t_start"][i + 1] else i + 1)

    def test_detectable_duration(self):
        p = [p for p in self.params if p["subtype"] == "wzsge" and np.sum(p["t_start"] >= 0) > 0][0]
        start, duration, n_ob = self.lc.detectable_duration(p, 0.05, 1.0, t_start=0, t_end=3652)
        self.assertGreater(n_ob, 0)
        self.assertGreater(duration, 1.0)
        self.assertTrue(np.isfinite(start))
        # Far too distant to be seen
        start, duration, n_ob = self.lc.detectable_duration(p, 0.05, 1000.0, t_start=0, t_end=3652)
        self.assertEqual(duration, 0.0)
        self.assertTrue(np.isnan(start))


class TestCvPopulation(unittest.TestCase):
    def test_positions(self):
        heights = np.ones(5000) * DN_SCALE_HEIGHTS["suuma"]
        ra, dec, dist, in_bulge = sample_dwarf_nova_positions(heights, bulge_fraction=0.3, seed=3)
        self.assertEqual(len(ra), 5000)
        self.assertTrue(np.all((ra >= 0) & (ra < 360)))
        self.assertTrue(np.all((dec >= -90) & (dec <= 90)))
        self.assertTrue(np.all(dist > 0))
        self.assertAlmostEqual(in_bulge.mean(), 0.3, delta=0.05)
        # bulge objects are concentrated around the Galactic centre
        self.assertAlmostEqual(np.median(dist[in_bulge]), 8.2, delta=1.0)

    def test_expected_number(self):
        n = expected_number_of_dwarf_novae()
        self.assertGreater(n, 1e6)
        self.assertLess(n, 1e7)
        self.assertGreater(expected_number_of_dwarf_novae(bulge_fraction=0.5), n)


class TestCVPopMetric(unittest.TestCase):
    def _data_slice(self, n_visits=2000, seed=11):
        rng = np.random.default_rng(seed)
        names = ["observationStartMJD", "fiveSigmaDepth", "band", "night"]
        types = [float, float, "U1", int]
        data = np.zeros(n_visits, dtype=list(zip(names, types)))
        nights = np.sort(rng.integers(0, 3650, size=n_visits))
        data["night"] = nights
        data["observationStartMJD"] = SURVEY_START_MJD + nights + rng.uniform(0.1, 0.5, size=n_visits)
        data["band"] = rng.choice(list("ugrizy"), size=n_visits)
        data["fiveSigmaDepth"] = rng.normal(24.0, 0.5, size=n_visits)
        return data

    def _slice_point(self, subtype, distance, ebv=0.02):
        lc = CvLc(seed=5)
        params = lc.outburst_params(size=50)
        param = [p for p in params if p["subtype"] == subtype and np.sum(p["t_start"] >= 0) > 0][0]
        start, duration, n_ob = lc.detectable_duration(param, ebv, distance)
        return {
            "outburst_params": param,
            "distance": distance,
            "ebv": ebv,
            "visible_start_time": start,
            "visible_duration": duration,
            "n_outbursts": n_ob,
        }

    def test_run_nearby(self):
        metric = CVPopMetric(output_lc=True)
        data = self._data_slice()
        sp = self._slice_point("suuma", 0.3)
        result = metric.run(data, sp)
        for key in [
            "possible_to_detect",
            "ever_detect",
            "early_detect",
            "number_of_detections",
            "n_outbursts",
            "n_outbursts_detected",
            "quiescence_detected",
            "outburst_and_quiescence",
        ]:
            self.assertIn(key, result)
        self.assertTrue(result["possible_to_detect"])
        self.assertTrue(result["ever_detect"])
        self.assertTrue(result["quiescence_detected"])
        self.assertTrue(result["outburst_and_quiescence"])
        self.assertGreater(result["n_outbursts_detected"], 0)
        self.assertLessEqual(result["n_outbursts_detected"], result["n_outbursts"])
        self.assertEqual(len(result["lc"]), len(result["lc_colnames"]))
        # reduce functions
        self.assertEqual(metric.reduce_ever_detect(result), True)
        frac = metric.reduce_fraction_outbursts_detected(result)
        self.assertGreater(frac, 0)
        self.assertLessEqual(frac, 1)

    def test_run_distant(self):
        metric = CVPopMetric()
        data = self._data_slice()
        sp = self._slice_point("wzsge", 500.0)
        result = metric.run(data, sp)
        self.assertFalse(result["possible_to_detect"])
        self.assertFalse(result["ever_detect"])
        self.assertEqual(result["number_of_detections"], 0)
        self.assertFalse(result["quiescence_detected"])
        self.assertEqual(metric.reduce_fraction_outbursts_detected(result), 0)

    @unittest.skipUnless(
        os.path.isdir(os.path.join(get_data_dir(), "maps", "DustMaps3D")),
        "Skipping slicer test because 3-d dust maps are not available.",
    )
    def test_slicer(self):
        slicer = generate_cv_pop_slicer(n_events=100, seed=2)
        self.assertEqual(slicer.nslice, 100)
        for key in [
            "distance",
            "ebv",
            "subtype",
            "orbital_period",
            "in_bulge",
            "outburst_params",
            "visible_start_time",
            "visible_duration",
            "n_outbursts",
        ]:
            self.assertIn(key, slicer.slice_points)
            self.assertEqual(len(slicer.slice_points[key]), 100)
        self.assertTrue(np.all(slicer.slice_points["ebv"] >= 0))
        self.assertTrue(np.all(slicer.slice_points["visible_duration"] >= 0))


if __name__ == "__main__":
    unittest.main()
