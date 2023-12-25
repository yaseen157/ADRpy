"""
This module contains tools for the constraint analysis of fixed wing aircraft.
"""
import typing
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants, optimize

from ADRpy import atmospheres as at
from ADRpy import unitconversions as co
from ADRpy import mtools4acdc as actools
from ADRpy import propulsion as pdecks

__author__ = "Yaseen Reza"


class AircraftConcept:
    """
    Definition of a basic aircraft concept. An object of this class defines an
    aircraft design in terms of the *brief* it is aiming to meet, high level
    *design* variables that specify it, key parameters that describe its
    *performance*, the *atmosphere* it operates in, as well as the type of
    propulsion system. These are the five arguments that define an object of the
    AircraftConcept class. The first three are dictionaries, as described below,
    and the last two are instances of `Atmosphere <https://adrpy.readthedocs.io/en/latest/#atmospheres.Atmosphere>`_
    and engine decks in the `propulsion module <https://adrpy.readthedocs.io/en/latest/#propulsion>`_.

    """

    def __init__(self, brief: dict = None, design: dict = None,
                 performance: dict = None,
                 atmosphere: at.Atmosphere = None,
                 propulsion: typing.Union[
                     pdecks.EngineDeck, str, typing.Any] = None):
        """
        Args:
            brief: Definition of the design brief, that is, the requirements the
                design seeks to meet. Contains the following key names:

                climbalt_m
                    Float. The altitude (in metres) where the climb rate
                    requirement is specified. Optional, defaults to zero
                    (sea level).

                climbspeed_kias
                    Float. The airspeed (in knots, indicated) at which the
                    required climb rate has to be achieved.

                climbrate_fpm
                    Float. Required climb rate (in feet per minute) at the
                    altitude specified in the *climbalt_m* entry (above).

                cruisealt_m
                    Float. The altitude at which the cruise speed requirement
                    will be defined.

                cruisespeed_ktas
                    Float. The required cruise speed (in knots, true) at the
                    altitude specified in the *cruisealt_m* entry (above).

                cruisethrustfact
                    Float. The fraction of thrust (throttle setting) that should
                    be used to satisfy the cruise constraint.

                servceil_m
                    Float. The required service ceiling in meters (that is, the
                    altitude at which the maximum rate of climb drops to 100
                    feet per minute).

                secclimbspd_kias
                    Float. The speed (knots indicated airspeed) at which the
                    service ceiling must be reached. This should be an estimate
                    of the best rate of climb speed.

                vstallclean_kcas
                    Float. The maximum acceptable stall speed (in knots,
                    indicated/calibrated).

                groundrun_m
                    Float. Length (in metres) of take-off ground run in meters
                    at the elevation defined by the *rwyelevation_m* entry of
                    the dictionary. This is a basic, 100% N1, no wind, zero
                    runway gradient ground run.

                rwyelevation_m
                    Float. The elevation (in metres) of the runway againts which
                    the take-off constraint is defined. Optional, defaults to
                    zero (sea level).

                stloadfactor
                    Float. Load factor to be sustained by the aircraft in a
                    steady, level turn.

                turnalt_m
                    Float. Altitude (in metres) where the turn requirement is
                    defined. Optional, defaults to zero (sea level).

                turnspeed_ktas
                    Float. True airspeed (in knots) at which the turn
                    requirement (above) has to be met. Since the dynamics of
                    turning flight is dominated by inertia, which depends
                    on ground speed, the turn speed is specified here as TAS (on
                    the zero wind assumption). If you'd rather specify this as
                    IAS/CAS/EAS, use `eas2tas <https://adrpy.readthedocs.io/en/latest/#atmospheres.Atmosphere.eas2tas>`_
                    first to obtain the TAS value.

                Example design brief::

                    brief = {'rwyelevation_m':0, 'groundrun_m':313,
                             'stloadfactor': 1.5, 'turnalt_m': 1000, 'turnspeed_ktas': 100,
                             'climbalt_m': 0, 'climbspeed_kias': 101, 'climbrate_fpm': 1398,
                             'cruisealt_m': 3048, 'cruisespeed_ktas': 182, 'cruisethrustfact': 1.0,
                             'servceil_m': 6580, 'secclimbspd_kias': 92,
                             'vstallclean_kcas': 69}

            design: Definition of key, high level design variables that define
                the future design.

                aspectratio
                    Float. Wing aspect ratio. Optional, defaults to 8.

                sweep_le_deg
                    Float. Main wing leading edge sweep angle (in degrees).
                    Optional, defaults to zero (no sweep).

                sweep_mt_deg
                    Float. Main wing sweep angle measured at the maximum
                    thickness point. Optional, defaults to value of
                    'sweep_le_deg'.

                sweep_25_deg
                    Float. Main wing sweep angle measured at the quarter chord
                    point. Optional, defaults to ~29% sweep_le_deg,
                    ~71% sweep_mt_deg.

                taperratio
                    Float. Standard definition of wing tip chord to root chord
                    ratio, zero for sharp, pointed wing-tip delta wings.
                    Optional, defaults to the theoretical optimal value as a
                    function of the quarter-chord sweep angle.

                weight_n
                    Float. Specifies the maximum take-off weight of the
                    aircraft.

                weightfractions
                    Dictionary. Specifies at what fraction of the maximum
                    take-off weight do various constraints have to be met. It
                    should contain the following keys: *take-off*, *climb*,
                    *cruise*, *turn*, *servceil*. Optional, each defaults to
                    1.0 if not specified.

            performance: Definition of key, high level design performance
                estimates.

                CD0TO
                    Float. Zero-lift drag coefficient in the take-off
                    configuration.

                CDTO
                    Float. Take-off drag coefficient. Optional, defaults to
                    0.09.

                CDmin
                    Float. Zero lift drag coefficient in clean configuration.
                    Optional, defaults to 0.03.

                mu_R
                    Float. Coefficient of rolling resistance on the wheels.
                    Optional, defaults to 0.03.

                CLTO
                    Float. Take-off lift coefficient. Optional, defaults to
                    0.95.

                CLmaxTO
                    Float. Maximum lift coefficient in take-off conditions.
                    Optional, defaults to 1.5.

                CLmax
                    Float. Maximum lift coefficient in flight, in clean
                    configuration.

                eta_prop
                    Dictionary. Propeller efficiency in various phases of the
                    mission. It should contain the following keys: *take-off*,
                    *climb*, *cruise*, *turn*, *servceil*. Optional, unspecified
                    entries in the dictionary default to the following values:

                    :code: `etap = {'take-off': 0.45, 'climb': 0.75, 'cruise': 0.85, 'turn': 0.85, 'servceil': 0.65}`

            atmosphere: `Atmosphere <https://adrpy.readthedocs.io/en/latest/#atmospheres.Atmosphere>`_
                class object. Specifies the virtual atmosphere in which all the
                design calculations within the *AircraftConcept* class will be
                performed. Optional, defaults to the International Standard
                Atmosphere.

            propulsion: `Engine deck <https://adrpy.readthedocs.io/en/latest/#atmospheres.Atmosphere>`_
                object, or a case-sensitive string specifying the propulsion
                system (or its type) for performance lapse calculations. Options
                are described in the EngineDeck class documentation, which
                includes (but is not limited to) *TurbofanHiBPR*,
                *TurbofanLoBPR*, *Turboprop*, *Turbojet*, *Piston*,
                *SuperchargedPiston*. Optional, defaults to *Piston*.

        """
        # Parse the input arguments
        brief = dict() if brief is None else brief
        design = dict() if design is None else design
        performance = dict() if performance is None else performance
        atmosphere = at.Atmosphere() if atmosphere is None else atmosphere
        # propulsion comes later, it's complicated

        # ----- DESIGN BRIEF HANDLING -----
        # Climb constraint
        brief.setdefault("climbalt_m", 0.0)
        brief.setdefault("climbspeed_kias")
        brief.setdefault("climbrate_fpm")
        # Cruise constraint
        brief.setdefault("cruisealt_m")
        brief.setdefault("cruisespeed_ktas")
        brief.setdefault("cruisethrustfact", 1.0)
        # Service ceiling constraint
        brief.setdefault("servceil_m")
        brief.setdefault("secclimbspd_kias")
        # Stall constraint
        brief.setdefault("vstallclean_kcas")
        # Take-off constraint
        brief.setdefault("groundrun_m")
        brief.setdefault("rwyelevation_m", 0.0)
        # Sustained turn constraint
        brief.setdefault("stloadfactor")
        brief.setdefault("turnalt_m", 0.0)
        brief.setdefault("turnspeed_ktas")

        # ----- CONCEPT DESIGN HANDLING -----
        # Geometry definitions
        design.setdefault("aspectratio", 8.0)
        design.setdefault("sweep_le_deg", 0.0)
        design.setdefault("sweep_mt_deg", design["sweep_le_deg"])
        design.setdefault(
            "sweep_25_deg",
            (2 / 7) * design["sweep_le_deg"]
            + (5 / 7) * design["sweep_mt_deg"]
        )
        design.setdefault(
            "taperratio",
            # Optimal root taper ratio (if one is not provided)
            # https://www.fzt.haw-hamburg.de/pers/Scholz/OPerA/OPerA_PRE_DLRK_12-09-10_MethodOnly.pdf
            0.45 * np.exp(-0.0375 * design["sweep_25_deg"])
        )
        # Weight and loading
        design.setdefault("weight_n")
        design.setdefault("weightfractions", dict())
        design["weightfractions"].setdefault("climb", 1.0)
        design["weightfractions"].setdefault("cruise", 1.0)
        design["weightfractions"].setdefault("servceil", 1.0)
        design["weightfractions"].setdefault("take-off", 1.0)
        design["weightfractions"].setdefault("turn", 1.0)

        # ----- CONCEPT PERFORMANCE HANDLING -----
        # Drag/resistance coefficients
        performance.setdefault("CD0TO")
        performance.setdefault("CDTO")
        performance.setdefault("CDmin")
        performance.setdefault("mu_R")
        # Lift coefficients
        performance.setdefault("CLTO", 0.95)
        performance.setdefault("CLmaxTO", 1.5)
        performance.setdefault("CLmax")
        # Propulsive efficiencies
        performance.setdefault("eta_prop", dict())
        performance["eta_prop"].setdefault("climb", 0.75)
        performance["eta_prop"].setdefault("cruise", 0.85)
        performance["eta_prop"].setdefault("servceil", 0.65)
        performance["eta_prop"].setdefault("take-off", 0.45)
        performance["eta_prop"].setdefault("turn", 0.85)

        # ----- PROPULSION HANDLING -----
        propulsion = "Piston" if propulsion is None else propulsion
        if isinstance(
                propulsion,
                (pdecks.EngineDeck, pdecks.TurbofanHiBPR, pdecks.TurbofanLoBPR,
                 pdecks.Turbojet, pdecks.Turboprop, pdecks.Piston,
                 pdecks.SuperchargedPiston)
        ):
            pass
        elif isinstance(propulsion, str):
            propulsion = pdecks.EngineDeck(f"class:{propulsion}")
        else:
            raise ValueError(f"{propulsion=} is an invalid choice")

        # Make parameters accessible to the public
        self.brief = type("brief", (object,), brief)
        self.design = type("design", (object,), design)
        self.performance = type("performance", (object,), performance)
        self.designatm = atmosphere
        self.propulsion = propulsion

        return

    def cdi_factor(self, **kwargs):
        """
        Estimate the induced drag factor (as in CD = CD_0 + k * CL^2).

        Keyword Args:
            mach: Freestream Mach number.
            method: Users may select any of:
                ``"Cavallo"``, for Oswald span efficiency, e0;
                ``"Brandt"``, for inviscid span efficiency, e;
                ``"Nita-Scholz"``, for Oswald span efficiency, e0=f(M);
                ``"Obert"``, for Oswald span efficiency, e0;
                ``"Kroo"``, for Oswald span efficiency, e0;

        Returns:
            Estimate for induced drag factor, k.

        Notes:
            Brandt's method returns an estimate for the inviscid spanwise
            effiency factor e.

        References:
            -   Cavallo, B., "Subsonic drag estimation methods", Technical
                Report NADC-AW-6604, US Naval Air Development Center, 1966.
            -   Brandt, S. et al., "Introduction to Aeronautics: A Design
                Perspective", 2nd ed., American Institute of Aeronautics and
                Astronautics, 2004, p. 163.
            -   Nita, M., Scholz, D., "Estimating the Oswald Factor from Basic
                Aircraft Geometrical Parameters," Hamburg University of Applied
                Sciences, Hamburg, Germany, DocumentID: 281424, 2012. Accessed:
                24/12/2023. [Online]. Available: https://www.fzt.haw-hamburg.de/pers/Scholz/OPerA/OPerA_PUB_DLRK_12-09-10.pdf
            -   Obert E. et al., "Aerodynamic Design of Transport Aircraft",
                IOS Press BV, 2009, p. 542.
            -   Kroo, I., "Aircraft Design: Synthesis and Analysis", Stanford
                Desktop Aeronautics, 2001 [Online], ch. Lift-Dependent Drag
                Items. Available: https://www.academia.edu/43367454/Aircraft_Design_Synthesis_and_Analysis

        """
        # Recast as necessary
        mach = kwargs.get("mach", 0.3)
        method = kwargs.get("method", "Nita-Scholz")

        if method == "Cavallo":
            # Find leading edge sweep and wing aspect ratio
            aspectratio = getattr(self.design, "aspectratio")
            sweep = np.radians(getattr(self.design, "sweep_le_deg"))

            # Compute Oswald span efficiency estimate
            ARfactor = (1 - 0.045 * aspectratio ** 0.68)
            e0_straight = 1.78 * ARfactor - 0.64
            e0_swept = 4.61 * ARfactor * np.cos(sweep) ** 0.15 - 3.1
            wt = np.interp(sweep, [0, np.radians(30)], [1, 0])
            e0_mixed = wt * e0_straight + (1 - wt) * e0_swept

            cdi_factor = 1 / (np.pi * e0_mixed * aspectratio)

        elif method == "Brandt":
            # THIS IS ACTUALLY THE SPAN EFFICIENCY FACTOR!!!
            # Find maximum thickness sweep and wing aspect ratio
            aspectratio = getattr(self.design, "aspectratio")
            sweep = np.radians(getattr(self.design, "sweep_mt_deg"))

            # Compute Oswald span efficiency estimate
            sqrtterm = 4 + aspectratio ** 2 * (1 + (np.tan(sweep)) ** 2)
            e0_estimate = 2 / (2 - aspectratio + np.sqrt(sqrtterm))

            cdi_factor = 1 / (np.pi * e0_estimate * aspectratio)

        elif method == "Nita-Scholz":
            # Find quarter chord sweep, wing taper ratio, and aspect ratio
            sweep = getattr(self.design, "sweep_25_deg")
            taperratio = getattr(self.design, "taperratio")
            aspectratio = getattr(self.design, "aspectratio")

            # Calculate Hoerner's delta/AR factor for unswept wings (with NASA's
            # swept wing study, fitted for c=25% sweep)
            dtaperratio = -0.357 + 0.45 * np.exp(-0.0375 * sweep)
            tcorr = taperratio - dtaperratio
            k_hoernerfactor = (
                    (0.0524 * tcorr ** 4)
                    - (0.15 * tcorr ** 3)
                    + (0.1659 * tcorr ** 2)
                    - (0.0706 * tcorr)
                    + 0.0119

            )
            # Theoretical Oswald efficiency, before corrections
            e_theo = 1 / (1 + k_hoernerfactor * aspectratio)

            # CORRECTION FACTOR F: Kroo's correction factor due to reduced lift
            # from fuselage presence (assumed fuselage diam. / span = 11.4%)
            dfoverb_all = 0.114
            ke_fuse = 1 - 2 * (dfoverb_all ** 2)

            # CORRECTION FACTOR D0: Correction factor due to viscous drag from
            # generated lift
            ke_d0 = 0.85

            # CORRECTION FACTOR M: Correction factor due to compressibility
            # effects on induced drag - constants from statistical analysis
            if mach > 0.3:
                ke_mach = -0.001521 * (((mach / 0.3) - 1) ** 10.82) + 1
            else:
                ke_mach = 1

            e0_estimate = np.clip(e_theo * ke_fuse * ke_d0 * ke_mach, 0, None)

            cdi_factor = 1 / (np.pi * e0_estimate * aspectratio)

        elif method == "Obert":
            # Find the aspect ratio
            aspectratio = getattr(self.design, "aspectratio")

            e0_estimate = 1 / (1.05 + 0.007 * np.pi * aspectratio)

            cdi_factor = 1 / (np.pi * e0_estimate * aspectratio)

        elif method == "Kroo":
            # Find the inviscid induced drag factor, aspect ratio, and CDmin
            cdi_inviscid = self.cdi_factor(method="Brandt")
            aspectratio = getattr(self.design, "aspectratio")
            CDmin = getattr(self.performance, "CDmin")

            K = 0.38
            piAR = np.pi * aspectratio
            e_inviscid = 1 / piAR / cdi_inviscid
            e0_estimate = 1 / (1 / e_inviscid + K * CDmin * piAR)

            cdi_factor = 1 / e0_estimate / piAR

        elif method == "Raymer":
            raise NotImplementedError("Awaiting implementation")

        else:
            raise ValueError(f"Invalid selection {method=}. Try 'Nita-Scholz'?")

        return cdi_factor

    def get_LDmax(self, **kwargs):
        # Recast as necessary
        CDmin = getattr(self.performance, "CDmin")
        CLmax = getattr(self.performance, "CLmax")

        # Differentiated simple drag model, solve CD0 - k CL^2 = 0.0
        k = self.cdi_factor(method="Nita-Scholz", **kwargs)
        CL = (CDmin / k) ** 0.5
        if CL > CLmax:
            warnmsg = f"Couldn't achieve (L/D)max, needed {CL=} ({CLmax=})"
            warnings.warn(warnmsg, RuntimeWarning)
            CL = np.clip(CL, None, CLmax)

        LDmax = CL / (CDmin + k * CL ** 2)
        return LDmax

    def get_bestclimbangle_Vx(self, wingloading_pa, **kwargs):
        """
        Estimate the speed, Vx, which results in the best angle of climb.

        Args:
            wingloading_pa: Aircraft wing loading, in Pascal.

        Keyword Args:
            altitude_m: Geopotential altitude, in metres.

        Returns:
            Best climb angle speed Vx, in metres per second.

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        """
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        altitude_m = actools.recastasnpfloatarray(kwargs.get("altitude_m", 0.0))
        wingloading_pa, altitude_m \
            = np.broadcast_arrays(wingloading_pa, altitude_m)
        CDmin = getattr(self.performance, "CDmin")
        weight_n = getattr(self.design, "weight_n")

        rho_kgpm3 = self.designatm.airdens_kgpm3(altitude_m)
        vsound_mps = self.designatm.vsound_mps(altitude_m)

        if self.propulsion.type in ["turbofan", "turbojet"]:

            densfactor = 2 / rho_kgpm3

            def f_obj(Vx, i):
                """Helper func: Objective function. Accepts guess of Vx."""

                # Gudmundsson, eq. (18-21)
                mach = Vx / vsound_mps.flat[i]
                k = self.cdi_factor(mach=mach, method="Nita-Scholz")
                thrust = self.propulsion.thrust(
                    mach, altitude_m.flat[i], norm=False
                )
                LDmax = self.get_LDmax(mach=mach)
                theta_max = np.arcsin(np.clip(
                    thrust / weight_n - 1 / LDmax,
                    None, 1  # Limit theta_max to 90 degrees climb angle
                ))

                # Gudmundsson, eq. (18-22)
                bestspeed_guess_mps = (densfactor.flat[i]
                                       * wingloading_pa.flat[i]
                                       * (k / CDmin) ** 0.5  # == CL ** -.5
                                       * np.cos(theta_max)) ** 0.5

                return bestspeed_guess_mps

            def f_opt(Vx, i):
                """Helper func: Solve for Vx."""
                return f_obj(Vx, i) - Vx

        elif self.propulsion.type in ["turboprop", "piston"]:

            eta_prop = getattr(self.performance, "eta_prop")["climb"]
            weight_n = getattr(self.design, "weight_n")

            def f_opt(Vx, i):
                """Helper func: Solve for Vx, for each coefficient index i."""

                # Gudmundsson, eq. (18-26)
                mach = Vx / vsound_mps.flat[i]
                shaftpower = self.propulsion.shaftpower(
                    mach, altitude_m.flat[i], norm=False)
                power = eta_prop * shaftpower
                c1 = power / rho_kgpm3.flat[i] / (
                        weight_n / wingloading_pa.flat[i]) / CDmin
                k = self.cdi_factor(mach=mach, method="Nita-Scholz")
                c2 = (wingloading_pa.flat[i] ** 2 * 4 * k
                      / rho_kgpm3.flat[i] ** 2 / CDmin)

                return Vx ** 4 + c1 * Vx - c2

        else:
            raise NotImplementedError("Unsupported propulsion system type")

        bestspeed_mps = np.array([
            optimize.newton(f_opt, x0=100.0, args=(i,))
            for i, _ in enumerate(wingloading_pa.flat)
        ]).reshape(wingloading_pa.shape)

        return bestspeed_mps

    def get_bestclimbrate_Vy(self, wingloading_pa, **kwargs):
        """
        Estimate the speed, Vy, which results in the best rate of climb.

        Args:
            wingloading_pa: Aircraft wing loading, in Pascal.

        Keyword Args:
            altitude_m: Geopotential altitude, in metres.

        Returns:
            Best rate of climb speed Vy, in metres per second.

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        """
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        altitude_m = actools.recastasnpfloatarray(kwargs.get("altitude_m", 0.0))
        wingloading_pa, altitude_m \
            = np.broadcast_arrays(wingloading_pa, altitude_m)
        CDmin = getattr(self.performance, "CDmin")

        rho_kgpm3 = self.designatm.airdens_kgpm3(altitude_m)
        vsound_mps = self.designatm.vsound_mps(altitude_m)

        if self.propulsion.type in ["turbofan", "turbojet"]:

            wsfactor = wingloading_pa / 3 / rho_kgpm3 / CDmin

            def f_obj(Vy, i):
                """Helper func: Objective function. Accepts guess of Vy."""

                # Gudmundsson, eq. (18-24)
                mach = Vy / vsound_mps.flat[i]
                thrust_n = self.propulsion.thrust(
                    mach, altitude_m.flat[i], norm=False)
                weight_n = getattr(self.design, "weight_n")
                tw = thrust_n / weight_n
                LDmax = self.get_LDmax(mach=mach)
                ldfactor = 1 + (1 + 3 / LDmax ** 2 / tw ** 2) ** 0.5
                bestspeed_guess_mps = (tw * wsfactor.flat[i] * ldfactor) ** 0.5

                return bestspeed_guess_mps

        elif self.propulsion.type in ["turboprop", "piston"]:

            densfactor = 2 / rho_kgpm3

            def f_obj(Vy, i):
                """Helper func: Objective function. Accepts guess of Vy."""

                # Gudmundsson, eq. (18-27)
                mach = Vy / vsound_mps.flat[i]
                k = self.cdi_factor(mach=mach, method="Nita-Scholz")
                dragfactor = (k / (3 * CDmin)) ** 0.5
                bestspeed_guess_mps = (densfactor.flat[i]
                                       * wingloading_pa.flat[i]
                                       * dragfactor) ** 0.5

                return bestspeed_guess_mps

        else:
            raise NotImplementedError("Unsupported propulsion system type")

        def f_opt(Vy, i):
            """Helper func: Solve for Vy."""
            return f_obj(Vy, i) - Vy

        bestspeed_mps = np.array([
            optimize.newton(f_opt, x0=100.0, args=(i,))
            for i, _ in enumerate(wingloading_pa.flat)
        ]).reshape(wingloading_pa.shape)

        return bestspeed_mps

    def constrain_climb(self, wingloading_pa):
        """
        Compute the thrust-to-weight and the power-to-weight (if applicable)
        requirements to climb as prescribed in the design brief for the concept.

        Args:
            wingloading_pa: Wing loading, in Pascal.

        Returns:
            A tuple of (T/W_required, P/W_required). If the brief is completely
            defined for this constraint, T/W_required is always calculated and
            is dimensionless. P/W_required is calculated if it makes sense for
            a propulsion system to be characterised by its output shaft power,
            and has units of metres per second. If any wing loading produces an
            unattainable lift coefficient or is otherwise not computed, NaN
            values are returned. Values are always mapped back to sea-level
            static performance of the propulsion system, and to MTOW conditions.

        """
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        climbalt_m = getattr(self.brief, "climbalt_m")
        climbspeed_kias = getattr(self.brief, "climbspeed_kias")
        climbrate_fpm = getattr(self.brief, "climbrate_fpm")
        CDmin = getattr(self.performance, "CDmin")
        CLmax = getattr(self.performance, "CLmax")

        # Determine the thrust and power lapse corrections
        climbspeed_mpsias = co.kts2mps(climbspeed_kias)
        climbspeed_mpstas = self.designatm.eas2tas(
            eas=climbspeed_mpsias,
            altitude_m=climbalt_m
        )
        mach = climbspeed_mpstas / self.designatm.vsound_mps(climbalt_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude=climbalt_m, norm=True,
            eta_prop=getattr(self.performance, "eta_prop")["climb"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=climbalt_m, norm=True
            ) * getattr(self.performance, "eta_prop")["climb"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = getattr(self.design, "weightfractions")["climb"]

        # Compute climb constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=climbspeed_mpstas, altitude_m=climbalt_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... induced drag factor
        k = self.cdi_factor(mach=mach, method="Nita-Scholz")

        # ... rate of climb penalty
        climbrate_mps = co.fpm2mps(climbrate_fpm)
        climbrate_mpstroc = self.designatm.eas2tas(
            eas=climbrate_mps, altitude_m=climbalt_m)
        cos_sq_theta = (1 - (climbrate_mpstroc / climbspeed_mpstas) ** 2)

        # ... "acceleration factor"
        Ka = 1.0

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CDmin / ws_pa
                     + k / q_pa * ws_pa * cos_sq_theta
                     + Ka * climbrate_mpstroc / climbspeed_mpstas
             ) / tcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * climbspeed_mpstas / pcorr

        return tw, pw

    def constrain_cruise(self, wingloading_pa):
        """
        Compute the thrust-to-weight and the power-to-weight (if applicable)
        requirements to cruise as prescribed in the design brief for the
        concept.

        Args:
            wingloading_pa: Wing loading, in Pascal.

        Returns:
            A tuple of (T/W_required, P/W_required). If the brief is completely
            defined for this constraint, T/W_required is always calculated and
            is dimensionless. P/W_required is calculated if it makes sense for
            a propulsion system to be characterised by its output shaft power,
            and has units of metres per second. If any wing loading produces an
            unattainable lift coefficient or is otherwise not computed, NaN
            values are returned. Values are always mapped back to sea-level
            static performance of the propulsion system, and to MTOW conditions.

        """
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        cruisealt_m = getattr(self.brief, "cruisealt_m")
        cruisespeed_ktas = getattr(self.brief, "cruisespeed_ktas")
        cruisethrustfact = getattr(self.brief, "cruisethrustfact")
        CDmin = getattr(self.performance, "CDmin")
        CLmax = getattr(self.performance, "CLmax")

        # Determine the thrust and power lapse corrections
        cruisespeed_mpstas = co.kts2mps(cruisespeed_ktas)
        mach = cruisespeed_mpstas / self.designatm.vsound_mps(cruisealt_m)
        tcorr = cruisethrustfact * self.propulsion.thrust(
            mach=mach, altitude=cruisealt_m, norm=True,
            eta_prop=getattr(self.performance, "eta_prop")["cruise"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=cruisealt_m, norm=True
            ) * getattr(self.performance, "eta_prop")["cruise"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = getattr(self.design, "weightfractions")["cruise"]

        # Compute cruise constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=cruisespeed_mpstas, altitude_m=cruisealt_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... induced drag factor
        k = self.cdi_factor(mach=mach, method="Nita-Scholz")

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CDmin / ws_pa
                     + k / q_pa * ws_pa
             ) / tcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * cruisespeed_mpstas / pcorr

        return tw, pw

    def constrain_servceil(self, wingloading_pa):
        """
        Compute the thrust-to-weight and the power-to-weight (if applicable)
        requirements to fly at the service ceiling as prescribed in the design
        brief for the concept.

        Args:
            wingloading_pa: Wing loading, in Pascal.

        Returns:
            A tuple of (T/W_required, P/W_required). If the brief is completely
            defined for this constraint, T/W_required is always calculated and
            is dimensionless. P/W_required is calculated if it makes sense for
            a propulsion system to be characterised by its output shaft power,
            and has units of metres per second. If any wing loading produces an
            unattainable lift coefficient or is otherwise not computed, NaN
            values are returned. Values are always mapped back to sea-level
            static performance of the propulsion system, and to MTOW conditions.

        """
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        servceil_m = getattr(self.brief, "servceil_m")
        secclimbspd_kias = getattr(self.brief, "secclimbspd_kias")
        CDmin = getattr(self.performance, "CDmin")
        CLmax = getattr(self.performance, "CLmax")

        # Determine the thrust and power lapse corrections
        secclimbspd_mpsias = co.kts2mps(secclimbspd_kias)
        secclimbspd_mpstas = self.designatm.eas2tas(
            eas=secclimbspd_mpsias,
            altitude_m=servceil_m
        )
        mach = secclimbspd_mpstas / self.designatm.vsound_mps(servceil_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude=servceil_m, norm=True,
            eta_prop=getattr(self.performance, "eta_prop")["servceil"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=servceil_m, norm=True
            ) * getattr(self.performance, "eta_prop")["servceil"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = getattr(self.design, "weightfractions")["servceil"]

        # Compute climb constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=secclimbspd_mpstas, altitude_m=servceil_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... induced drag factor
        k = self.cdi_factor(mach=mach, method="Nita-Scholz")

        # Service ceiling typically defined in terms of climb rate (at best
        # climb speed) dropping to 100feet/min ~ 0.508m/s
        climbrate_mps = co.fpm2mps(100)
        # What true climb rate does 100 feet/minute correspond to?
        climbrate_mpstroc = self.designatm.eas2tas(
            eas=climbrate_mps, altitude_m=servceil_m)

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CDmin / ws_pa
                     + k / q_pa * ws_pa
                     + climbrate_mpstroc / secclimbspd_mpstas
             ) / tcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * secclimbspd_mpstas / pcorr

        return tw, pw

    def constrain_takeoff(self, wingloading_pa):
        """
        Compute the thrust-to-weight and the power-to-weight (if applicable)
        requirements to perform take-off as prescribed in the design brief for
        the concept.

        Args:
            wingloading_pa: Wing loading, in Pascal.

        Returns:
            A tuple of (T/W_required, P/W_required). If the brief is completely
            defined for this constraint, T/W_required is always calculated and
            is dimensionless. P/W_required is calculated if it makes sense for
            a propulsion system to be characterised by its output shaft power,
            and has units of metres per second. If any wing loading produces an
            unattainable lift coefficient or is otherwise not computed, NaN
            values are returned. Values are always mapped back to sea-level
            static performance of the propulsion system, and to MTOW conditions.

        """
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        groundrun_m = getattr(self.brief, "groundrun_m")
        rwyelevation_m = getattr(self.brief, "rwyelevation_m")
        # to_headwind_kts = getattr(self.brief, "to_headwind_kts")
        # to_slope_perc = getattr(self.brief, "to_slope_perc")
        CDTO = getattr(self.performance, "CDTO")
        mu_R = getattr(self.performance, "mu_R")
        CLTO = getattr(self.performance, "CLTO")
        CLmaxTO = getattr(self.performance, "CLmaxTO")

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = getattr(self.design, "weightfractions")["take-off"]

        # Approximation for lift-off speed
        ws_pa = wingloading_pa * wcorr
        airdensity_kgpm3 = self.designatm.airdens_kgpm3(rwyelevation_m)
        vs1to_mps = (2 * ws_pa / airdensity_kgpm3 / CLmaxTO) ** 0.5
        vrotate_mps = vs1to_mps
        vliftoff_mps = vrotate_mps

        # Determine the thrust and power lapse corrections
        vbar_mpstas = 0.75 * vliftoff_mps  # Repr. speed from CoM of integral
        machbar = vbar_mpstas / self.designatm.vsound_mps(rwyelevation_m)
        tcorr = self.propulsion.thrust(
            mach=machbar, altitude=rwyelevation_m, norm=True,
            eta_prop=getattr(self.performance, "eta_prop")["take-off"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=machbar, altitude=rwyelevation_m, norm=True
            ) * getattr(self.performance, "eta_prop")["take-off"]
        else:
            pcorr = np.nan

        # Compute take-off constraint
        # ... T/W (mapped to sea-level static)
        acctakeoffbar = vliftoff_mps ** 2 / 2 / groundrun_m
        tw = (
                     acctakeoffbar / constants.g
                     + 0.5 * CDTO / CLTO
                     + 0.5 * mu_R
             ) / tcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * vbar_mpstas / pcorr

        return tw, pw

    def constrain_turn(self, wingloading_pa):
        """
        Compute the thrust-to-weight and the power-to-weight (if applicable)
        requirements to carry out the constrained turn as prescribed in the
        design brief for the concept.

        Args:
            wingloading_pa: Wing loading, in Pascal.

        Returns:
            A tuple of (T/W_required, P/W_required). If the brief is completely
            defined for this constraint, T/W_required is always calculated and
            is dimensionless. P/W_required is calculated if it makes sense for
            a propulsion system to be characterised by its output shaft power,
            and has units of metres per second. If any wing loading produces an
            unattainable lift coefficient or is otherwise not computed, NaN
            values are returned. Values are always mapped back to sea-level
            static performance of the propulsion system, and to MTOW conditions.

        """
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        stloadfactor = getattr(self.brief, "stloadfactor")
        turnalt_m = getattr(self.brief, "turnalt_m")
        turnspeed_ktas = getattr(self.brief, "turnspeed_ktas")
        CDmin = getattr(self.performance, "CDmin")
        CLmax = getattr(self.performance, "CLmax")

        # Determine the thrust and power lapse corrections
        turnspeed_mpstas = co.kts2mps(turnspeed_ktas)
        mach = turnspeed_mpstas / self.designatm.vsound_mps(turnalt_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude=turnalt_m, norm=True,
            eta_prop=getattr(self.performance, "eta_prop")["turn"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=turnalt_m, norm=True
            ) * getattr(self.performance, "eta_prop")["turn"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = getattr(self.design, "weightfractions")["turn"]

        # Compute climb constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=turnspeed_mpstas, altitude_m=turnalt_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... induced drag factor
        k = self.cdi_factor(mach=mach, method="Nita-Scholz")

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CDmin / ws_pa
                     + k / q_pa * ws_pa * stloadfactor ** 2
             ) / tcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * turnspeed_mpstas / pcorr

        return tw, pw

    @property
    def cleanstall_WSmax(self):
        """
        The maximum viable wing loading before stalling at sea-level, when the
        aircraft is in its clean configuration.

        Returns:
            Wing loading W/S, in Pascal.

        """
        # (W/S)_max = q_vstall * CLmax
        # Recast as necessary
        vstallclean_kcas = getattr(self.brief, "vstallclean_kcas")
        CLmax = getattr(self.performance, "CLmax")
        if vstallclean_kcas is None or CLmax is None:
            return np.nan

        # We do the q calculation at SL conditions, TAS ~= EAS ~= CAS
        # (on the basis that the stall Mach number is likely very small)
        stallspeed_mpstas = co.kts2mps(vstallclean_kcas)

        # Dynamic pressure at sea level
        q_pa = self.designatm.dynamicpressure_pa(stallspeed_mpstas, 0.0)
        ws_pa = q_pa * CLmax
        return ws_pa

    @property
    def cleanstall_Smin(self):
        """
        The minimum viable wing area before stalling at sea-level, when the
        aircraft is in its clean configuration.

        Returns:
            Wing area S, in m^2.

        """
        # (S)_min = W / (W/S)_max
        # Recast as necessary
        weight_n = getattr(self.design, "weight_n")
        s_m2 = weight_n / self.cleanstall_WSmax
        return s_m2

    def plot_constraints(self, wingloading_pa):
        """
        Make a pretty figure with all the constraints outlined in the design
        brief. Depending on the contents of the aircraft concept's brief,
        design, and performance arguments, thrust-to-weight, power-to-weight,
        thrust, and power graphs are intelligently selected from.

        Args:
            wingloading_pa: An ordered array of wing loading values across which
                the concept's constraints should be evaluated.

        Returns:
            A tuple of the matplotlib (Figure, Axes) objects used to plot all
            the data.

        """
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        # Compute constraints
        constraint_fs = {
            "climb": self.constrain_climb,
            "cruise": self.constrain_cruise,
            "service ceiling": self.constrain_servceil,
            "sustained turn": self.constrain_turn,
            "take-off": self.constrain_takeoff
        }
        tws, pws = dict(), dict()
        for label, function in constraint_fs.items():
            try:
                # noinspection PyArgumentList
                tws[label], pws[label] = function(wingloading_pa)
            except (TypeError, RuntimeError) as _:
                # tws[label] = np.nan * wingloading_pa
                # pws[label] = np.nan * wingloading_pa
                pass

        # Choose between S and W/S, depending on if the MTOW is known
        weight_n = getattr(self.design, "weight_n")

        if weight_n:
            xs = weight_n / wingloading_pa
            xstall = weight_n / self.cleanstall_WSmax
        else:
            xs = wingloading_pa
            xstall = self.cleanstall_WSmax

        # Choose between T/W and P/W, depending on propulsion system
        type_is_power = self.propulsion.type in ["piston", "turboprop"]
        if type_is_power:
            if weight_n:
                ys = {k: v * weight_n for (k, v) in pws.items()}  # P [W]
            else:
                ys = pws  # P/W  [W/N], a.k.a [m/s]
        else:
            if weight_n:
                ys = {k: v * weight_n for (k, v) in tws.items()}  # T [N]
            else:
                ys = tws  # T/W
        ymed = np.nanmax(np.nanmedian(np.array(list(ys.values())), axis=1))
        ylim = ymed / 0.45  # Median of constraints is the bot. 45 % of the fig.
        if np.isnan(ylim):
            errormsg = (
                f"{self.plot_constraints.__name__} found no values"
            )
            raise RuntimeError(errormsg)

        fig, ax = plt.subplots(dpi=140, figsize=(7, 4))
        fig.subplots_adjust(right=0.7, left=0.19, bottom=0.21)

        # Plot the main constraints
        cmap = plt.get_cmap("jet")
        ncolours = len(constraint_fs) + 1
        colours = cmap(np.arange(ncolours) / ncolours)
        for i, (label, values) in enumerate(ys.items()):
            clr = colours[i]
            ax.plot(xs, values, label=label, c=clr, zorder=10, lw=2)
            ax.fill_between(xs, values, fc=clr, alpha=0.2, zorder=5)

        # Plot the clean stall
        if np.isfinite(xstall):
            yx1x2 = [0, ylim], xstall, xs.min() if weight_n else xs.max()
            clr = colours[len(constraint_fs)]
            l2d = ax.axvline(xstall, label="clean stall, 1g", c=clr, zorder=10)
            ax.fill_betweenx(*yx1x2, fc=l2d.get_color(), alpha=0.2, zorder=5)

        # Plot the stall limit due to all constraints
        nanindex = np.isnan(np.array(list(ys.values()))).sum(axis=1).max()
        yx1x2 = [0, ylim], xs[-nanindex], xs.min() if weight_n else xs.max()
        if nanindex > 0:
            kkka = (0, 0, 0, 0.2)
            ax.fill_betweenx(
                *yx1x2, fc=kkka, ec="r", zorder=15, hatch="x",
                label=r"above $C_{L,max}$"
            )

        # Limits and positioning
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(0, ylim)
        ax.grid()

        # Custom legend behaviour - allow user to redraw legend w/default posns!
        def custom_legend_maker(*args, **kwargs):
            """A matplotlib legend but with default position parameters."""
            default_legend_kwargs = dict([
                ("bbox_to_anchor", (1.05, 0.618)),
                ("loc", "center left")
            ])
            return ax.legend(*args, **{**default_legend_kwargs, **kwargs})

        ax.remake_legend = custom_legend_maker  # Assign the method to our ax
        ax.remake_legend()  # Use the method

        # Default labelling - set defaults (quantities normalised by weight)
        ax.set_xlabel("Wing Loading [Pa]")
        if type_is_power:
            ax.set_ylabel("Power-to-Weight")
        else:
            ax.set_ylabel("Thrust-to-Weight")

        if weight_n is None:
            return fig, ax

        # Advanced labelling - Multiple unit labelling
        # All the magic below allows the Axes object to have primary axes for
        # the actual plot data to be in SI units, but display in other units
        # x, metric
        ax.set_xlabel("[m$^2$]")
        ax.xaxis.set_label_coords(1.09, -0.02)
        # x, imperial
        yloc = -0.12  # Shared y-location of secondary and tertiary axes
        secax_x = ax.secondary_xaxis(yloc, functions=(co.m22feet2, co.feet22m2))
        secax_x.set_xlabel("[ft$^2$]")
        secax_x.xaxis.set_label_coords(1.09, -0.02)
        secax_x.set_xticks([])
        # x, label
        terax_x = ax.secondary_xaxis(yloc, functions=(co.m22feet2, co.feet22m2))
        terax_x.set_xlabel("Wing Area")
        if type_is_power:
            # y, metric
            ax.set_yticks([])
            ax.set_ylabel("[W]")
            ax.yaxis.label.set_color((1.0, 0, 0, 0))
            # y, metric & imperial
            secax_y = ax.secondary_yaxis(0, functions=(co.w2kw, co.kw2w))
            secax_y.set_ylabel("[hp] | [kW]" + " " * 28, rotation=0)
            secax_y.yaxis.set_label_coords(0.0, -.1)  # x-transform is useless
            # y, label
            terax_y = ax.secondary_yaxis(-.14, functions=(co.w2hp, co.hp2w))
            terax_y.set_ylabel("Shaft Power")
        else:
            # y, metric
            ax.set_yticks([])
            ax.set_ylabel("[N]")
            ax.yaxis.label.set_color((1.0, 0, 0, 0))
            # y, metric & imperial
            secax_y = ax.secondary_yaxis(0, functions=(co.n2kn, co.kn2n))
            secax_y.set_ylabel("[lbf] | [kN]" + " " * 22, rotation=0)
            secax_y.yaxis.set_label_coords(0.0, -.1)  # x-transform is useless
            # y, label
            terax_y = ax.secondary_yaxis(-.14, functions=(co.n2lbf, co.lbf2n))
            terax_y.set_ylabel("Thrust")

        return fig, ax
