"""Unit tests for the constraint analysis module."""
import unittest

import numpy as np

from ADRpy import constraintanalysis as ca
from ADRpy import unitconversions as uc

from ADRpy._sampleac import Cirrus_SR22


class TestFunctions(unittest.TestCase):

    def test_drag_models(self):
        """
        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014. Example 19.1.

        """
        # Step 1
        e0 = 0.8294
        AR = 7.33
        k = 1 / np.pi / AR / e0

        # Step 2
        CL = 0.5
        # CDmin 0.02, k, CLmax 1.5, CLminD 0.0
        fsimple = ca.make_modified_drag_model(0.02, k, 1.5, 0.0)
        self.assertAlmostEqual(fsimple(CL), 0.0331, places=4)

        fadjusted = ca.make_modified_drag_model(0.02, k, 1.3, 0.2)
        self.assertAlmostEqual(fadjusted(CL), 0.0247, places=4)

        return

    def test_lift_curve_slopes(self):
        """
        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        """
        # Table 9-7.
        sweptback_tr05_sweep30 = ca.AircraftConcept(
            design={"aspectratio": 10, "taperratio": 0.5, "sweep_le_deg": 30})
        sweptfwrd_tr20_sweep30 = ca.AircraftConcept(
            design={"aspectratio": 10, "taperratio": 2.0, "sweep_le_deg": -30})
        sweptback_tr10_sweep30 = ca.AircraftConcept(
            design={"aspectratio": 10, "taperratio": 1.0, "sweep_le_deg": 30})
        sweptback_tr20_sweep30 = ca.AircraftConcept(
            design={"aspectratio": 10, "taperratio": 2.0, "sweep_le_deg": 30})
        delta = ca.AircraftConcept(
            design={"aspectratio": 2.27, "taperratio": 0, "sweep_le_deg": 60})
        print(delta.design.sweep_m_deg(delta.design.sweep_le_deg, 0, 0.5))

        # Example 9.8
        hershey20 = ca.AircraftConcept(
            design={"aspectratio": 20, "taperratio": 1, "sweep_le_deg": 0}
        )

        def case_checker(cases2test, methodname, delta):
            for i, (concept, gold) in enumerate(cases2test.items()):
                self.assertAlmostEqual(
                    concept.CLslope(method=methodname), gold, delta=delta,
                    msg=f"Failed on case: {i}"
                )
            return

        # Check Helmbold
        cases = {
            sweptback_tr05_sweep30: 5.15, sweptfwrd_tr20_sweep30: 5.15,
            sweptback_tr10_sweep30: 5.15, sweptback_tr20_sweep30: 5.15
        }
        case_checker(cases, "Helmbold", delta=1e-2)

        # Check DATCOM (use weaker tolerance because of assumption fuzziness)
        cases = {
            sweptback_tr05_sweep30: 4.40, sweptfwrd_tr20_sweep30: 4.24,
            sweptback_tr10_sweep30: 4.35, sweptback_tr20_sweep30: 4.45
        }
        case_checker(cases, "DATCOM", delta=0.3)

        cases = {hershey20: 5.472}
        case_checker(cases, "DATCOM", delta=0.1)

        return


class TestConceptCirrusSR22(unittest.TestCase):
    # Aircraft and MTOW wing loading
    ac = Cirrus_SR22()
    ws = uc.kg_N(uc.lb_kg(3600)) / 13.5

    def test_wingshape(self):
        self.assertEqual(self.ac.design.aspectratio, 10.12)
        self.assertEqual(self.ac.design.taperratio, 0.5)
        self.assertAlmostEqual(self.ac.design.sweep_le_deg, 2, places=0)
        return

    def test_get_bestspeeds(self):
        # Carson's speed
        VCAR = self.ac.get_bestV_CAR(self.ws)
        # VCAR_ktas = uc.mps_kts(VCAR)

        # Best glide speed (and range for propeller aircraft)
        VBG = self.ac.get_bestV_BG(self.ws)
        # VBG_ktas = uc.mps_kts(VBG)

        # Best climb rate speed
        VY = self.ac.get_bestV_Y(self.ws)
        # VY_ktas = uc.mps_kts(VY)

        self.assertGreater(VCAR, VBG)  # Carson's speed is always faster than BG
        self.assertGreater(VBG, VY)  # I don't know if this is always true

        return

    def test_get_bestCLs(self):
        # CL for range, constant speed (variable altitude)
        # Optimise CL / CD
        CL_r1, LD_r1 = self.ac.get_bestCL_range(constantspeed=True)

        # CL for range, variable speed (constant altitude)
        # Optimise sqrt(CL) / CD
        CL_r2, LD_r2 = self.ac.get_bestCL_range(constantspeed=False)

        # CL for endurance, constant speed (variable altitude)
        # Optimise CL / CD
        CL_e1, LD_e1 = self.ac.get_bestCL_endurance(constantspeed=True)

        # CL for endurance, variable speed (constant altitude)
        # Optimise CL^1.5 / CD (for propeller aircraft only, otherwise CL / CD)
        CL_e2, LD_e2 = self.ac.get_bestCL_endurance(constantspeed=False)

        # Each optimal result should be the best at its own CL^power / CD
        self.assertGreater(LD_r1, LD_r2)
        self.assertEqual(LD_r1, LD_e1)  # Same optimisation!
        self.assertGreater(LD_r1, LD_e2)

        self.assertGreater(LD_r2 / CL_r2 ** 0.5, LD_r1 / CL_r1 ** 0.5)
        self.assertGreater(LD_r2 / CL_r2 ** 0.5, LD_e1 / CL_e1 ** 0.5)
        self.assertGreater(LD_r2 / CL_r2 ** 0.5, LD_e2 / CL_e2 ** 0.5)

        # self.assertEqual(LD_e1, LD_r1)  # Don't bother, we already checked
        self.assertGreater(LD_e1, LD_r2)
        self.assertGreater(LD_e1, LD_e2)

        self.assertGreater(LD_e2 * CL_e2 ** 0.5, LD_r1 * CL_r1 ** 0.5)
        self.assertGreater(LD_e2 * CL_e2 ** 0.5, LD_r2 * CL_r2 ** 0.5)
        self.assertGreater(LD_e2 * CL_e2 ** 0.5, LD_e1 * CL_e1 ** 0.5)

        return

    def test_ground_influence(self):
        """
        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014. Section 9.5.8.

        """
        bref_m = 11.68
        href_m = 1.5 * bref_m
        altitudes = np.linspace(0, href_m, num=7)

        phis = {"Wieselberger": None, "McCormick": None, "Asselin": None}

        for methodname in phis:
            phis[methodname] = self.ac.ground_influence_coefficient(
                wingloading_pa=self.ws,
                h_m=altitudes,
                method=methodname
            )

        # Compare to reference (computed nums, but they looked right with fig.)
        self.assertTrue(np.isclose(
            phis["Asselin"],
            np.array([0., 0.80857224, 0.88247553, 0.9146219, 0.93283462,
                      0.94460361, 0.95284734])
        ).all())
        self.assertTrue(np.isclose(
            phis["Wieselberger"],
            np.array([0.04761905, 0.76877641, 0.92828006, 0.9983753, 1.03778071,
                      1.06303183, 1.08059378])
        ).all())
        self.assertTrue(np.isclose(
            phis["McCormick"],
            np.array([0., 0.94109626, 0.98459344, 0.99309352, 0.99610333,
                      0.99750263, 0.99826439])
        ).all())

        return

    def test_lift_curve_slopes(self):
        """
        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014. Example 9.11.

        """
        self.assertAlmostEqual(self.ac.CLslope(method="DATCOM"), 5.06, delta=.1)
        return


if __name__ == '__main__':
    unittest.main()
