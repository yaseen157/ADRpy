"""
This module contains tools for the analysis of an aircraft concept, from the
point of view of meeting initial airworthiness requirements.
"""
from functools import partial
from itertools import product
import typing
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants, optimize as sopt

from ADRpy import constraintanalysis as ca
# Don't believe PyCharm 2023's lies, we are v. much using ADRpy.unitconversions!
from ADRpy import unitconversions as uc
from ADRpy import mtools4acdc as actools

__author__ = "Yaseen Reza"


class CS23Amendment4:
    """
    A class for evaluating the conformity of an aircraft concept to the EASA
    Airworthiness Certification and Regulation rules for Part 23 aircraft.
    Methods contained within are built on EASA CS-23 (Amendment 4),
    specifications for "Normal", "Utility", "Aerobatic", and "Commuter"
    aeroplanes.

    Presently, the class' main attraction is supporting the construction of
    V-n diagrams for aircraft concepts using the rules.

    """

    def __init__(self,
                 concept: ca.AircraftConcept,
                 categories: typing.Union[list[str], str],
                 wingarea_m2: float):
        """
        Args:
            concept: Aircraft concept, as per the constraintsanalysis module.
            categories: A string, or list of strings describing desired CS-23
                certification categories.
            wingarea_m2: The concept's wing area, in metres squared.
        """

        self.concept = concept

        # Make sure categories is recorded as a list
        valid_categories = list(self._new_categories_dict())
        if isinstance(categories, str):
            categories = [categories]
        else:
            categories = list(categories)  # flatten tuples, dicts, etc.

        # Verify categories have been chosen correctly
        invalid_subset = set(categories) - set(valid_categories)
        if invalid_subset:
            errormsg = (
                f"'{categories=}' contained {invalid_subset=}. "
                f"Please ensure choices are all {valid_categories=}"
            )
            raise ValueError(errormsg)
        self.categories = categories

        self.wingarea_m2 = wingarea_m2

        return

    def _make_Vn_data(self,
                      altitude_m=None,
                      weightfraction=None,
                      N: int = None
                      ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Create data for plotting V-n diagrams.

        Args:
            altitude_m: Altitude of operation.
            weightfraction: Fraction of MTOW at which V-n should be considered.
            N: The number of x-coordinates to generate (discretising speeds from
             V=0 to dive speed, VD). Optional, defaults to 100 points.

        Returns:
            A tuple (altitude_m, weightfraction, Vndata), where the input arrays
            have been broadcasted against each other, and with Vndata, for
            convenience. Vndata is a class that itself contains a speeds class
            'V' (in KEAS), and load factor classes 'n' and 'ng' for manoeuvre
            and gust loads, respectively.

        """
        # Recast as necessary
        if altitude_m is None:
            altitude_m = 0
        if weightfraction is None:
            weightfraction = 1.0
        N = 100 if N is None else int(N)

        altitude_m = actools.recastasnpfloatarray(altitude_m)
        weightfraction = actools.recastasnpfloatarray(weightfraction)

        # Broadcast altitudes against weightfraction and then flatten
        altitude_m, weightfraction \
            = np.broadcast_arrays(altitude_m, weightfraction)
        output_Vndata = self._new_categories_dict()
        output_Vndata = {category: np.empty(altitude_m.shape, dtype=object)
                         for (category, _) in output_Vndata.items()}

        # Refactoring
        CLmax = self.concept.performance.CLmax
        CLmaxHL = np.nan
        if hasattr(self.concept.performance, "CLmaxHL"):
            CLmaxHL = self.concept.performance.CLmaxHL
        CLmin = self.concept.performance.CLmin
        designatm = self.concept.designatm
        funcs_ngminus1 = self.paragraph341_c_ngminus1
        wslim_pa = self.concept.design.weight_n / self.wingarea_m2  # MTOW (W/S)

        for i in range(altitude_m.size):
            alt_m = altitude_m.flat[i]
            wfrac = weightfraction.flat[i]
            ws_pa = wfrac * wslim_pa  # Current loading frac.

            # Compute rho - even though CS23 requires ISA, we allow other atms.
            rho = designatm.airdens_kgpm3(altitude_m=alt_m)

            def keas2mpstas(keas):
                """Convert knots EAS to m/s TAS."""
                ktas = designatm.eas2tas(eas=keas, altitude_m=alt_m)
                mpstas = uc.kts2mps(speed_kts=ktas)
                return mpstas

            # def mpstas2keas(mpstas):
            #     """Convert m/s TAS to knots EAS."""
            #     ktas = co.mps2kts(speed_mps=mpstas)
            #     keas = designatm.tas2eas(tas=ktas, altitude_m=alt_m)
            #     return keas

            def f_npos(keas):
                """Compute positively loaded stall curve."""
                V = keas2mpstas(keas)
                npos = 0.5 * rho * V ** 2 * CLmax / ws_pa
                return npos

            def f_npos_flap(keas):
                """Compute positively loaded stall curve, with full flaps."""
                V = keas2mpstas(keas)
                npos = 0.5 * rho * V ** 2 * CLmaxHL / ws_pa
                return npos

            def f_nneg(keas):
                """Compute negatively loaded stall curve."""
                V = keas2mpstas(keas)
                nneg = 0.5 * rho * V ** 2 * CLmin / ws_pa
                return nneg

            for category in output_Vndata:

                # ------------------------------------------------------------ #
                # Equivalent Airspeeds (initial guesses based on minimums)
                VC = self.paragraph335_a_VCmin[category]
                VD = self.paragraph335_b_VDmin[category]
                VAmano = self.paragraph335_c_VAmin[category](wfrac)
                VGmano = self.paragraph335_c_VGmin[category](wfrac)
                VF = self.paragraph345_b_VFmin(wfrac)
                # Stall (and inverted stall) speed
                VS = self.paragraph49_b_VS1 * (wfrac ** 0.5)
                VSi = VS * (CLmax / abs(CLmin)) ** 0.5

                # Find VA gust point (where stall curve meets the VC gust line)
                def f_opt(V, condition, npos=True):
                    """Optimisation func. for finding stall x gust crossing."""
                    ng_m1 = funcs_ngminus1[category][condition](V, ws_pa, alt_m)
                    if npos is True:
                        n = f_npos(V)
                        ng = 1 + ng_m1
                    else:
                        n = f_nneg(V)
                        ng = 1 - ng_m1
                    return n - ng

                try:
                    VAgust = sopt.newton(f_opt, VAmano, args=("C", True))
                except RuntimeError:
                    VAgust = np.nan
                VA = np.nanmax((VAmano, VAgust))
                # CS 23.335 Design airspeeds. Sub-paragraph (c). Item (2):
                VA = min(VA, VC)  # VA need not exceed VC

                # Find VB gust point (where stall curve meets the VB gust line)

                if category == "commuter":
                    try:
                        VB = sopt.newton(f_opt, VAmano, args=("B", True))
                    except RuntimeError:
                        VB = np.nan
                else:
                    VB = np.nan
                # CS 23.335 Design airspeeds. Sub-paragraph (d). Item (2):
                VB = min(VB, VC)  # VB need not exceed VC
                # Using the built-in min() preserves the np.nan, if it is there

                # Find VG gust point (inverted stall version of VA)
                # Intersection point isn't guaranteed, like VA or VB is!
                try:
                    VGgust = sopt.newton(f_opt, VGmano, args=("C", False))
                except RuntimeError:
                    VGgust = np.nan
                VG = np.nanmax((VGmano, VGgust))
                # CS 23.335 Design airspeeds. Sub-paragraph (c). Item (2):
                # Technically this was for VA, but I think it applies anyway
                VG = min(VG, VC)  # VG need not exceed VC

                class Speeds:
                    """V-n diagram speeds. Unless specified, units of KEAS."""
                    xs = np.linspace(0, VD, num=N)
                    A, B, C, D = VA, VB, VC, VD
                    F = VF
                    G = VG
                    S, Si = VS, VSi

                # ------------------------------------------------------------ #
                # Load factor landmarks + curves (manoeuvre, gust, and combined)

                # ... manoeuvring
                nAmano = f_npos(keas=VA)
                nGmano = f_nneg(keas=VG)
                nEmano = 0 if category in ["normal", "commuter"] else -1
                nFmano = self.paragraph345_a_nF

                class ManoeuvreLoads:
                    """V-n diagram manoeuvre load curves."""
                    S_A = f_npos(keas=Speeds.xs)
                    A_D = nAmano * np.ones(Speeds.xs.shape)
                    Si_G = f_nneg(keas=Speeds.xs)
                    G_F = nGmano * np.ones(Speeds.xs.shape)
                    F_E = np.interp(Speeds.xs, [VC, VD], [nGmano, nEmano])
                    flaps = np.clip(f_npos_flap(keas=Speeds.xs), None, nFmano)

                # ... gusting
                nCgust_m1 = funcs_ngminus1[category]["C"](VC, ws_pa, alt_m)
                nDgust_m1 = funcs_ngminus1[category]["D"](VD, ws_pa, alt_m)

                gust_args = (Speeds.xs, ws_pa, alt_m)
                gust_m1C = funcs_ngminus1[category]["C"](*gust_args)
                gust_m1D = funcs_ngminus1[category]["D"](*gust_args)

                if category == "commuter":
                    gust_m1B = funcs_ngminus1[category]["B"](*gust_args)
                    nBgust_m1 = funcs_ngminus1[category]["B"](VB, ws_pa, alt_m)
                else:
                    gust_m1B = np.nan * np.ones(Speeds.xs.shape)
                    nBgust_m1 = np.nan

                class GustLoads:
                    """V-n diagram gust loads."""
                    O_B = np.vstack((1 + gust_m1B, 1 - gust_m1B)).T
                    O_C = np.vstack((1 + gust_m1C, 1 - gust_m1C)).T
                    O_D = np.vstack((1 + gust_m1D, 1 - gust_m1D)).T
                    B_m1 = nBgust_m1
                    C_m1 = nCgust_m1
                    D_m1 = nDgust_m1

                class Vndata:
                    """Data for plotting a V-n diagram."""
                    V = Speeds
                    n = ManoeuvreLoads
                    ng = GustLoads

                output_Vndata[category].flat[i] = Vndata

        return altitude_m, weightfraction, output_Vndata

    @staticmethod
    def _parse_Vn_data(Vndata):
        """
        Given some computed V-n data objects from the _make_Vn_data() method,
        extract coordinates for the V-n diagram's manoeuvre limit, and combined
        manoeuvre and gust limit envelope.

        Args:
            Vndata: A V-n data object, as per the ones found in
                "self._make_Vn_data()[2][category].flat".

        Returns:
            A tuple of coordinates (xs, ys_combined, ys_manoeuvre). Plotting
            xs against ys will either reveal the combined gust and manoeuvre
            plot, or just the manoeuvre plot.

        """
        # Create masks for the upper half of the V-n diagram
        mask_O_B = (0.0 <= Vndata.V.xs) & (Vndata.V.xs < Vndata.V.B)
        mask_O_C = (0.0 <= Vndata.V.xs) & (Vndata.V.xs < Vndata.V.C)
        mask_O_S = (0.0 <= Vndata.V.xs) & (Vndata.V.xs < Vndata.V.S)
        mask_S_A = (Vndata.V.S <= Vndata.V.xs) & (Vndata.V.xs < Vndata.V.A)
        mask_A_C = (Vndata.V.A <= Vndata.V.xs) & (Vndata.V.xs < Vndata.V.C)
        mask_C_D = (Vndata.V.C <= Vndata.V.xs) & (Vndata.V.xs <= Vndata.V.D)

        # Create masks for the lower half of the V-n diagram
        mask_O_Si = (0.0 <= Vndata.V.xs) & (Vndata.V.xs < Vndata.V.Si)
        mask_Si_G = (Vndata.V.Si <= Vndata.V.xs) & (Vndata.V.xs < Vndata.V.G)
        mask_G_F = (Vndata.V.G <= Vndata.V.xs) & (Vndata.V.xs < Vndata.V.C)
        mask_F_E = (Vndata.V.C <= Vndata.V.xs) & (Vndata.V.xs <= Vndata.V.D)

        # Extract manoeuvre ONLY data using masks
        ys_u_mano = np.hstack((
            Vndata.n.S_A[mask_O_S] * np.nan,
            Vndata.n.S_A[mask_S_A],
            Vndata.n.A_D[mask_A_C | mask_C_D])
        )
        ys_l_mano = np.hstack((
            Vndata.n.Si_G[mask_O_Si] * np.nan,
            Vndata.n.Si_G[mask_Si_G],
            Vndata.n.G_F[mask_G_F],
            Vndata.n.F_E[mask_F_E]
        ))

        # Extract gust ONLY data using masks
        w1 = (Vndata.V.xs - Vndata.V.C) / (Vndata.V.D - Vndata.V.C)
        if np.isnan(Vndata.V.B):
            ys_u_gust = np.hstack((
                Vndata.ng.O_C[:, 0][mask_O_C],
                1 + (Vndata.ng.C_m1 * (1 - w1) + Vndata.ng.D_m1 * w1)[mask_C_D]
            ))
        elif Vndata.V.B < Vndata.V.C:
            # mask_B_C bridges the gap in the diagram between VB and VC;
            mask_B_C = (mask_O_B != mask_O_C)

            w2 = (Vndata.V.xs - Vndata.V.B) / (Vndata.V.C - Vndata.V.B)
            ys_u_gust = np.hstack((
                Vndata.ng.O_B[:, 0][mask_O_B],
                1 + (Vndata.ng.B_m1 * (1 - w2) + Vndata.ng.C_m1 * w2)[mask_B_C],
                1 + (Vndata.ng.C_m1 * (1 - w1) + Vndata.ng.D_m1 * w1)[mask_C_D]
            ))
        else:
            # VC <= VB, but we still need to penetrate gust
            mask_B_D = ~mask_O_B
            w2 = (Vndata.V.xs - Vndata.V.B) / (Vndata.V.D - Vndata.V.B)
            Bgustline = Vndata.ng.O_B[:, 0]  # Original 66~38 fps gust line
            capped = Vndata.n.S_A[mask_O_B]  # ... cap it to the stall curve
            Bgustline[mask_O_B] = capped  # <- ... like this
            ys_u_gust = np.hstack((
                Vndata.ng.O_B[:, 0][mask_O_B],
                (capped[-1] * (1 - w2) + (1 + Vndata.ng.D_m1) * w2)[mask_B_D]
            ))

        # Don't even bother looking for a -ve gust intercept with B-type gusts
        ys_l_gust = np.hstack((
            Vndata.ng.O_C[:, 1][mask_O_C],
            1 - (Vndata.ng.C_m1 * (1 - w1) + Vndata.ng.D_m1 * w1)[mask_C_D]))

        # Stack/combine plot data
        xs = np.hstack((Vndata.V.xs, Vndata.V.xs[::-1]))
        ys_u_comb = np.vstack((ys_u_mano, ys_u_gust))
        ys_u_comb[1, mask_O_S | mask_S_A] = np.nan
        ys_l_comb = np.vstack((ys_l_mano, ys_l_gust))
        ys_l_comb[1, mask_O_Si | mask_Si_G] = np.nan
        with warnings.catch_warnings():  # From V=0 to VS, ignore all-NaN slices
            warnings.simplefilter("ignore")
            ys_combined = np.hstack((
                np.nanmax(ys_u_comb, axis=0),
                np.nanmin(ys_l_comb, axis=0)[::-1]
            ))
        ys_manoeuvre = np.hstack((ys_u_mano, ys_l_mano[::-1]))

        return xs, ys_combined, ys_manoeuvre

    def plot_Vn(self, altitude_m=None, weightfraction=None, N=None):
        """
        Make a pretty figure with the limit manoeuvre and the limit gust
        envelopes.

        Args:
            altitude_m: Altitudes at which to consider the V-n diagram.
            weightfraction: Weight fractions at which to consider calculations.
            N: The number of coordinates to discretise the velocity axis of the
                V-n diagram.

        Returns:
            A tuple of the matplotlib (Figure, Axes) objects used to plot all
            the data.

        Notes:
            The "w/ flaps 100%" limit envelope is based on the manoeuvre limit
            rules, and does not include the EASA CS-23 regulatory rules for how
            the envelope should account for +/- 25 feet per second gusts.

        """
        # Recast as necessary
        altitude_m = 0 if altitude_m is None else altitude_m
        weightfraction = 1.0 if weightfraction is None else weightfraction
        N = 100 if N is None else int(2 * np.ceil(N / 2))  # Multiple of 2

        # Get V-n plot data, and then extract categories relevant to the concept
        altitude_m, weightfraction, data \
            = self._make_Vn_data(altitude_m, weightfraction, N=N)
        data = {k: v for (k, v) in data.items() if k in self.categories}

        Vncases = []
        for i in range(altitude_m.size):
            # Choose correct altitude/weightfraction (loading) case to consider
            Vncases.append([x[i] for x in data.values()])
        else:
            # Recast into an array where dimension 0 spans each cert. category
            Vn_by_category = np.array(list(zip(*Vncases)))

        # Plotting time!
        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=140)
        fig.suptitle(f"{type(self).__name__} V-n Diagram")
        fig.subplots_adjust(left=0.165, bottom=0.2)
        axtitle = (f"Type(s): {', '.join(data)} | "
                   f"MTOW: {uc.n2kg(self.concept.design.weight_n):.0f} [kg]")
        ax.set_title(axtitle, fontsize="small")

        # ... manoeuvre and combined plots
        # for each dataset belonging to a specific certification category
        for cat_data in Vn_by_category:
            # Map x and y coordinates of limit manoeuvre and gust envelopes
            # Each dimension zero should be a different altitude/weightfraction
            xdata, ydata, ydata_mano = zip(*map(self._parse_Vn_data, cat_data))
            xdata = np.array(xdata)
            ydata = np.vstack(ydata)
            ydata_mano = np.vstack(ydata_mano)
            # Warning: ydata_flap has N data points, not 2 * N like the others!
            ydata_flap = np.array(list(map(lambda x: x.n.flaps, cat_data)))

            for i, wfrac in enumerate(weightfraction.flat):
                # The coordinate arrays are missing stall lines. Let's fix it
                not_nan = ~np.isnan(ydata[i])
                xdatai = xdata[i][not_nan]  # <-- Only for manoeuvre and gust!!!
                ydatai, ydatai_mano = ydata[i][not_nan], ydata_mano[i][not_nan]
                xdatai = np.hstack((xdatai[0], xdatai, xdatai[-1]))
                ydatai = np.hstack((0, ydatai, 0))
                ydatai_mano = np.hstack((0, ydatai_mano, 0))
                # The same, but flaps have a different stall condition
                not_stalled = ydata_flap[i] >= 1
                xdatai_flap = xdata[i][:N][not_stalled]  # See above, not 2 * N
                ydatai_flap = ydata_flap[i][not_stalled]
                # Now plot
                ax.fill(xdatai, ydatai_mano, c="paleturquoise", zorder=-5,
                        label="Manoeuvre\n Envelope")
                ax.fill(xdatai, ydatai, c="thistle", zorder=-10,
                        label="w/ Gust Limits")
                if not_stalled.any():
                    ax.fill_between(xdatai_flap, ydatai_flap, fc="lightgreen",
                                    zorder=-15, label="w/ Flaps 100%")

        # ... gust lines
        gustcases = [x for sublist in data.values() for x in sublist]
        commuters = [~np.isnan(case.V.B) for case in gustcases]
        if any(commuters):
            gustcases = np.array(gustcases)[commuters]
        big_gusto = sorted(gustcases, key=lambda x: x.ng.C_m1)[-1]  # V. gusty!

        origin = np.array([0, 1])
        gustB = np.array(((0, 0), (big_gusto.V.B, big_gusto.ng.B_m1)))
        gustC = np.array(((0, 0), (big_gusto.V.C, big_gusto.ng.C_m1)))
        gustD = np.array(((0, 0), (big_gusto.V.D, big_gusto.ng.D_m1)))
        style = {"ls": "dashdot", "c": "k", "lw": 0.7, "alpha": 0.4}
        for (gust, sign) in product([gustB, gustC, gustD], [1, -1]):
            ax.axline(*(origin + np.array([1, sign]) * gust), **style)

        # ... plot labelling
        ax.set_xlim(0, None)
        ax.spines[['left', 'bottom']].set_position('zero')
        ax.spines[['top', 'right']].set_visible(False)
        secax_x = ax.secondary_xaxis("bottom")
        secax_x.set_xlabel("Airspeed [KEAS]")
        secax_x.set_xticks([])
        secax_x.spines["bottom"].set_visible(False)
        ax.set_ylabel("Load factor, $n$")

        # The x-axis origin tick "0" intersects the y-axis. Here's a fix! But,
        # ... annoyingly, this works locally and not in jupyter notebooks (???)
        # ax.set_xticks(ax.get_xticks(), [""] + ax.get_xticklabels()[1:])

        # need to squash legend duplicates
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*dict(zip(labels, handles)).items())  # squash!
        # ph = [plt.plot([], marker="", ls="")[0]]  # Canvas
        # handles, labels = ph + handles, ["Envelope type:"] + labels
        ax.legend(handles=handles, labels=labels, ncol=len(handles),
                  bbox_to_anchor=(0.5, -0.1), loc="upper center")

        return fig, ax

    @staticmethod
    def _new_categories_dict() -> dict[str, typing.Any]:
        """Return a dictionary of type:None for each certifiable type."""
        types = ["normal", "utility", "commuter", "aerobatic"]
        categories = dict(zip(types, [None for _ in types]))
        return categories

    @property
    def paragraph1_a(self) -> dict[str, bool]:
        """
        CS 23.1 Applicability.
        Sub-paragraph (a).
        """
        output = self._new_categories_dict()
        # Item (1)
        weight_n = self.concept.design.weight_n
        weight_lb = uc.n2lbf(weight_n)
        output["normal"] = weight_lb <= 12_500
        output["utility"] = weight_lb <= 12_500
        output["aerobatic"] = weight_lb <= 12_500

        # Item (2)
        propulsion_type = self.concept.propulsion.type
        has_prop = propulsion_type in ["piston", "turboprop", "electric"]
        output["commuter"] = (weight_lb <= 19_000 and has_prop)

        return output

    @property
    def paragraph3_d(self) -> dict[str, bool]:
        """
        CS 23.3 Aeroplane categories.
        Sub-paragraph (d).
        """
        output = self._new_categories_dict()

        turning_loadfactor = self.concept.brief.stloadfactor
        turning_bank_angle = np.degrees(np.arccos(1 / turning_loadfactor))
        output["commuter"] = turning_bank_angle <= 60

        return output

    @property
    def paragraph3_e(self) -> dict[str, bool]:
        """
        CS 23.3 Aeroplane categories.
        Sub-paragraph (e).
        """
        output = self._new_categories_dict()

        output = {k: True for (k, _) in output.items()}
        if len(self.categories) > 1:
            output["commuter"] = False

        return output

    @property
    def paragraph45_a(self) -> bool:
        """
        CS 23.45 General.
        Sub-paragraph (a).
        """
        # Item (1)
        if self.concept.designatm.is_isa is False:
            return False

        # Item (2)
        propulsion_type = self.concept.propulsion.type
        if "commuter" in self.categories:
            return True
        elif propulsion_type == "piston":
            weight_n = self.concept.design.weight_n
            weight_lb = uc.n2lbf(weight_n)
            return weight_lb > 6_000
        elif propulsion_type in ["turbofan", "turbojet", "turboprop"]:
            return True
        return False

    @property
    def paragraph49_b_VS1(self) -> float:
        """
        CS 23.49 Stalling speed.
        Sub-paragraph (b).

        Returns:
            Aircraft MTOW clean-configuration stalling speed, in knots CAS.

        """
        # Clean, level flight stall speed VS from flight tests (use the brief)
        if hasattr(self.concept.brief, "vstallclean_kcas"):
            VS = self.concept.brief.vstallclean_kcas
        else:
            raise ValueError("'vstallclean_kcas' was not set!")

        return VS

    @property
    def paragraph49_b_VS0(self) -> float:
        """
        CS 23.49 Stalling speed.
        Sub-paragraph (b).

        Returns:
            Aircraft MTOW landing-configuration stalling speed, in knots CAS.

        """
        # 0.5 * rho * VS0^2 * S_HL * CLmaxHL == 0.5 * rho * VS1^2 * S * CLmax
        # --> VS0^2 * S_HL * CLmaxHL == VS1^2 * S * CLmax
        # --> VS0^2 == VS1^2 * (S / S_HL) * (CLmax / CLmaxHL)

        if not hasattr(self.concept.performance, "CLmax"):
            return np.nan
        CLmax = self.concept.performance.CLmax

        if not hasattr(self.concept.performance, "CLmaxHL"):
            return np.nan
        CLmaxHL = self.concept.performance.CLmaxHL

        # Assume that ratio of wing area to high-lift wing area (S / S_HL) is 1
        # for CS-23 aircraft --> (S / S_HL) == 1.0
        VS1 = self.paragraph49_b_VS1
        VS0 = (VS1 ** 2 * 1.0 * (CLmax / CLmaxHL)) ** 0.5

        return VS0

    @property
    def paragraph333_c_Ude(self) -> dict[str, dict[str, typing.Callable]]:
        """
        CS 23.333 Flight envelope.
        Sub-paragraph (c).

        Returns:
            Gust velocity as functions of altitude, given for VC, VD, and VB
            (where applicable). Gust velocities are in metres per second, and a
            function of the altitude given in metres above mean sea level.

        """
        output = self._new_categories_dict()

        # Item (1)
        kws = {"xp": (uc.feet2m(20e3), uc.feet2m(50e3)), "right": 0}
        u50fps, u25fps, u13fps = uc.feet2m(50), uc.feet2m(25), uc.feet2m(12.5)

        for (category, _) in output.items():
            output[category] = dict([
                ("C", partial(np.interp, **kws, fp=[u50fps, u25fps])),
                ("D", partial(np.interp, **kws, fp=[u25fps, u13fps]))
            ])

        u66fps, u38fps = uc.feet2m(66), uc.feet2m(38)
        output["commuter"]["B"] = partial(np.interp, **kws, fp=[u66fps, u38fps])

        return output

    @property
    def paragraph335_a_VCmin(self) -> dict[str, float]:
        """
        CS 23.335 Design airspeeds.
        Sub-paragraph (a).

        Returns:
            Minimum design cruising speed, VCmin, in knots EAS.

        Notes:
            The limit is computed, per instruction, to depend on MTOW.

        """
        output = self._new_categories_dict()

        # Items (1) and (2)
        W = self.concept.design.weight_n
        S = self.wingarea_m2
        ws_lbfpft2 = uc.pa2lbfft2(W / S)

        for (category, _) in output.items():
            factor = 36 if category == "aerobatic" else 33
            factor = np.interp(ws_lbfpft2, [20, 100], [factor, 28.6])
            output[category] = factor * ws_lbfpft2 ** 0.5

        return output

    @property
    def paragraph335_a_VCmax(self):
        """
        CS 23.335 Design airspeeds.
        Sub-paragraph (a).

        Returns:
            Maximum design cruising speed, VCmax, in knots EAS.

        """
        raise NotImplementedError("Haven't found a way to get VH yet")

    @property
    def paragraph335_a(self) -> dict[str, bool]:
        """
        CS 23.335 Design airspeeds.
        Sub-paragraph (a).
        """
        output = self._new_categories_dict()

        # Item (1)
        alt_m = self.concept.brief.cruisealt_m
        VC_ktas = self.concept.brief.cruisespeed_ktas
        VC_keas = self.concept.designatm.tas2eas(tas=VC_ktas, altitude_m=alt_m)

        for (category, VCmin_keas) in self.paragraph335_a_VCmin.items():
            output[category] = VCmin_keas <= VC_keas

        return output

    @property
    def paragraph335_b_VDmin(self) -> dict[str, float]:
        """
        CS 23.335 Design airspeeds.
        Sub-paragraph (b).

        Returns:
            Minimum design dive speed, VDmin, in knots EAS.

        Notes:
            The limit is computed, per instruction, to depend on MTOW.

        """
        output = self._new_categories_dict()

        # Items (2) and (3)
        VCmin_keas = self.paragraph335_a_VCmin
        W = self.concept.design.weight_n
        S = self.wingarea_m2
        ws_lbfpft2 = uc.pa2lbfft2(W / S)

        factors = {"utility": 1.50, "aerobatic": 1.55}

        for (category, _) in output.items():
            factor = factors.get(category, 1.40)
            factor = np.interp(ws_lbfpft2, [20, 100], [factor, 1.35])
            output[category] = factor * VCmin_keas[category]

        return output

    @property
    def paragraph335_c_VAmin(self) -> dict[str, typing.Callable]:
        """
        CS 23.335 Design airspeeds.
        Sub-paragraph (c).

        Returns:
            Function for minimum design manoeuvring speed, VAmin, in knots EAS.
            Accepts an optional argument 'weightfraction', defaults to 1.0.

        """
        # warnmsg = f"No method exists for kcas -> keas, assuming CAS ~ EAS"
        # warnings.warn(warnmsg, RuntimeWarning)
        # Item (1)
        n1 = self.paragraph337_a_n1
        VS = self.paragraph49_b_VS1

        def get_VAmin(n, weightfraction=None):
            """Given manoeuvre load factor (and weight fraction), get VAmin."""
            weightfraction = 1.0 if weightfraction is None else weightfraction
            VAmin = VS * (n / weightfraction) ** 0.5
            return VAmin

        output = {
            category: partial(get_VAmin, n)
            for (category, n) in n1.items()
        }

        return output

    @property
    def paragraph335_c_VGmin(self) -> dict[str, typing.Callable]:
        """
        CS 23.335 Design airspeeds.
        Sub-paragraph (c).

        Returns:
            Function for minimum design inverted manoeuvring speed, VGmin, in
            knots EAS. Accepts an optional argument 'weightfraction', defaults
            to 1.0.

        """
        # warnmsg = f"No method exists for kcas -> keas, assuming CAS ~ EAS"
        # warnings.warn(warnmsg, RuntimeWarning)
        # (W/S) = 0.5 * rho * V^2 * CL
        # ==> 2 * (W/S) / rho = V^2 * CL
        VS = self.paragraph49_b_VS1
        CLmax = self.concept.performance.CLmax
        CLmin = self.concept.performance.CLmin
        VSi = (VS ** 2 * CLmax / abs(CLmin)) ** 0.5

        # Item (1), technically this is for VA, but I think it applies to VG too
        def get_VGmin(n, weightfraction=None):
            """Given manoeuvre load factor (and weight fraction), get VGmin."""
            weightfraction = 1.0 if weightfraction is None else weightfraction
            VGmin = VSi * (np.abs(n) / weightfraction) ** 0.5
            return VGmin

        n2 = self.paragraph337_b_n2
        output = {
            category: partial(get_VGmin, n)
            for (category, n) in n2.items()
        }

        return output

    @property
    def paragraph335_c_VAmax(self) -> dict[str, float]:
        """
        CS 23.335 Design airspeeds.
        Sub-paragraph (c).

        Returns:
            Maximum design manoeuvring speed, VAmax.

        """
        output = self._new_categories_dict()

        # Item (2)
        alt_m = self.concept.brief.cruisealt_m
        VC_ktas = self.concept.brief.cruisespeed_ktas
        VC_keas = self.concept.designatm.tas2eas(tas=VC_ktas, altitude_m=alt_m)

        output = {category: VC_keas for (category, _) in output.items()}

        return output

    @property
    def paragraph335_d_VBmax(self) -> typing.Callable:
        """
        CS 23.335 Design airspeeds.
        Sub-paragraph (d).

        Raises:
            NotImplementedError.

        """
        raise NotImplementedError("sorry!")

    @property
    def paragraph335_d_VBmin(self) -> typing.Callable:
        """
        CS 23.335 Design airspeeds.
        Sub-paragraph (d).

        Raises:
            NotImplementedError.

        Notes:
            Assumes VS == VS1. This is true for MTOW, but otherwise the output
            must be corrected using: Vcorr = V_output / (weightfraction ** 0.5).

        """
        raise NotImplementedError("sorry!")

    @property
    def paragraph337_a_n1(self) -> dict[str, float]:
        """
        CS 23.337 Limit manoeuvring load factors.
        Sub-paragraph (a).

        Returns:
            Positive manoeuvring limit load factors.

        """
        output = self._new_categories_dict()

        # Item (1)
        weight_n = self.concept.design.weight_n
        weight_lb = uc.n2lbf(weight_n)
        n1 = np.clip(2.1 + 24_000 / (weight_lb + 10_000), 0, 3.8)
        output["normal"] = n1
        output["commuter"] = n1

        # Item (2)
        output["utility"] = 4.4
        output["aerobatic"] = 6.0

        return output

    @property
    def paragraph337_b_n2(self) -> dict[str, float]:
        """
        CS 23.337 Limit manoeuvring load factors.
        Sub-paragraph (b).

        Returns:
            Negative manoeuvring limit load factors.

        """
        # Items (1) and (2)
        output = {
            category: -0.5 * n1 if category == "aerobatic" else -0.4 * n1
            for (category, n1) in self.paragraph337_a_n1.items()
        }

        return output

    @property
    def paragraph341_c_ngminus1(self) -> dict[str, dict[str, typing.Callable]]:
        """
        CS 23.341 Gust load factors.
        Sub-paragraph (c).

        Returns:
            A dictionary of gust loading functions, that take the speed of the
            aeroplane in knots EAS, the altitude of consideration in metres
            above mean sea level, and the applicable wing loading in Pascal.

        """
        output = self._new_categories_dict()

        # Aeroplane mean geometric chord
        Cbar = (self.wingarea_m2 / self.concept.design.aspectratio) ** 0.5

        CLalpha = self.concept.performance.CLalpha
        rho0 = self.concept.designatm.airdens_kgpm3(altitude_m=0.0)

        def mu_g(wingloading_pa, altitude_m):
            """Aeroplane mass ratio, = function(wingloading, altitude)."""
            rho = self.concept.designatm.airdens_kgpm3(altitude_m)
            num = 2 * wingloading_pa
            den = rho * Cbar * CLalpha * constants.g
            return num / den

        def kg(wingloading_pa, altitude_m):
            """Gust alleviation factor, = function(wingloading, altitude)."""
            massratio = mu_g(wingloading_pa, altitude_m)
            num = 0.88 * massratio
            den = 5.3 + massratio
            return num / den

        for category, speedsdict in self.paragraph333_c_Ude.items():

            def one_pm_this(keas, wingloading_pa, altitude_m, f_Ude):
                """Gust load factor: n = 1 +/- 'one_pm_this(...)'."""
                mpseas = uc.kts2mps(speed_kts=keas)
                gustfactor = kg(wingloading_pa, altitude_m)
                num = gustfactor * rho0 * f_Ude(altitude_m) * mpseas * CLalpha
                den = 2 * wingloading_pa
                return num / den

            output[category] = dict()
            for condition, Ude in speedsdict.items():
                output[category][condition] = partial(one_pm_this, f_Ude=Ude)

        return output

    @property
    def paragraph345_a_nF(self) -> float:
        """
        CS 23.345 High lift devices.
        Sub-paragraph (a).

        Returns:
            Positive manoeuvring limit load factor for flight with high-lift
            devices deployed.

        """
        return 2.0

    @property
    def paragraph345_b_VFmin(self) -> typing.Callable:
        """
        CS 23.345 High lift devices.
        Sub-paragraph (b).

        Returns:
            Function for minimum design flap speed, VFmin, in knots EAS.
            Accepts an optional argument 'weightfraction', defaults to 1.0.

        """
        VS1 = self.paragraph49_b_VS1  # MTOW stall, clean-config
        VS0 = self.paragraph49_b_VS0  # MTOW stall, landing-config

        def get_VFmin(weightfraction=None):
            """Given optional parameter of weight fraction, get VFmin."""
            weightfraction = 1.0 if weightfraction is None else weightfraction
            VFmin = weightfraction ** 0.5 * max(1.4 * VS1, 1.8 * VS0)
            return VFmin

        return get_VFmin


if __name__ == "__main__":
    from ADRpy import atmospheres as at
    from ADRpy import constraintanalysis as ca
    from ADRpy import unitconversions as uc

    designbrief = {
        "vstallclean_kcas": 101
    }

    designdef = {
        "aspectratio": 10.48,
        "weight_n": uc.lbf2n(17120)
    }

    designperf = {
        "CLmax": 1.6, "CLmin": -1, "CLalpha": 6.28
    }

    designatm = at.Atmosphere()  # set the design atmosphere to a zero-offset ISA

    designpropulsion = "Turboprop"

    Beech1900Dconcept = ca.AircraftConcept(
        brief=designbrief,
        design=designdef,
        performance=designperf,
        atmosphere=designatm,
        propulsion=designpropulsion
    )

    Beech1900Dcs23spec = CS23Amendment4(Beech1900Dconcept, "commuter",
                                        uc.feet22m2(310))

    fig, ax = Beech1900Dcs23spec.plot_Vn(weightfraction=0.7)

    plt.show()
