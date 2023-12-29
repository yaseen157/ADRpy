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


def make_modified_drag_model(CDmin, k, CLmax, CLminD):
    """
    Return a function f, which computes the coefficient of drag as a function of
    the coefficient of lift, i.e. CD=f(CL).

    Args:
        CDmin: The minimum drag coefficient to use in the drag model.
        k: The induced drag factor, as seen in CD = CDmin + k * CL ** 2.
        CLmax: The maximum allowable coefficient of lift of the drag model.
        CLminD: The coefficient of lift at which the drag model is minimised.

    Returns:
        A function, CD = f(CL).

    References:
        Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
        and Procedures," 1st ed., Elselvier, 2014.
    """

    def quadratic(CL, _CDmin, _k):
        """Standard, classical quadratic model for drag."""
        return _CDmin + _k * CL ** 2

    def quadratic_adjusted(CL, _CDmin, _k, _CLminD):
        """Quadratic model adjusted for CL of minimum CD."""
        return quadratic(CL=(CL - _CLminD), _CDmin=_CDmin, _k=_k)

    def get_quadratic_modified(_CDmin, _k, _CLminD, _CLmax):
        """Make a quadratic drag model with spline modification."""

        # Estimate point of switching to quadratic spline
        CLm = 0.5 * (_CLminD + _CLmax)

        # Estimate the CDstall about 180% of the quadratic model
        CDstall = 1.8 * quadratic_adjusted(
            _CDmin=_CDmin, _k=_k, CL=_CLmax, _CLminD=_CLminD)

        # As per [1], fit a spline to produce a modified drag model
        matA = np.array([
            [CLm ** 2, CLm, 1],
            [2 * CLm, 1, 0],
            [_CLmax ** 2, _CLmax, 1]
        ])
        matB = np.array([
            quadratic_adjusted(_CDmin=_CDmin, _k=_k, CL=CLm, _CLminD=_CLminD),
            2 * _k * (CLm - _CLminD),
            CDstall
        ])
        A, B, C = np.linalg.solve(matA, matB)

        def drag_model(CL):
            """
            A modified, adjusted drag model.

            Args:
                CL: Coefficent of lift.

            Returns:
                The coefficient of drag, CD.

            """
            # Recast as necessary
            CL = actools.recastasnpfloatarray(CL)

            # Switch between models as we need to
            CDmod = A * CL ** 2 + B * CL + C
            CDquad = quadratic_adjusted(
                _CDmin=_CDmin, _k=_k, CL=CL, _CLminD=_CLminD)
            CD = np.where(CL <= CLm, CDquad, CDmod)

            if (CL > _CLmax).any():
                warnmsg = f"Coefficient of lift exceeded CLmax={_CLmax}"
                # warnings.warn(warnmsg, category=RuntimeWarning)
                CD[CL > _CLmax] = np.nan

            return CD

        return drag_model

    model2return = get_quadratic_modified(
        _CDmin=CDmin, _k=k, _CLminD=CLminD, _CLmax=CLmax)

    return model2return


def get_default_concept_design_objects():
    """
    Create classes for storing and easy access to attributes of an aircraft
    concept.

    Returns:
        A tuple of classes for storing design brief, design definition, and
        design performance, respectively.

    """

    class BaseMethods:
        """Default methods to run on instantiation."""

        def __init__(self, dictionary: dict = None):
            """
            Args:
                dictionary: key-value pairs with which to update default args.
            """
            if dict is None:
                return

            for key, value in dictionary.items():
                # If we already have a default value, overwrite it
                if hasattr(self, key):
                    # A value that is type dict, should update the original dict
                    if isinstance(value, dict):
                        currentdict = getattr(self, key)
                        setattr(self, key, {**currentdict, **value})
                    else:
                        setattr(self, key, value)
                # There is no default value, but the parameter exists, so set it
                elif hasattr(self, "__annotations__") and (
                        key in self.__annotations__):
                    setattr(self, key, value)
                else:
                    errormsg = f"Unknown {key=} for {type(self).__name__}"
                    raise KeyError(errormsg)
            return

    class DesignBrief(BaseMethods):
        """Parameters of the aircraft design brief."""
        # Climb constraint
        climbalt_m = 0.0
        climbspeed_kias: float
        climbrate_fpm: float
        # Cruise constraint
        cruisealt_m: float
        cruisespeed_ktas: float
        cruisethrustfact = 1.0
        # Service ceiling constraint
        servceil_m: float
        secclimbspd_kias: float
        # Stall constraint
        vstallclean_kcas: float
        # Take-off constraint
        groundrun_m: float
        rwyelevation_m = 0.0
        # Sustained turn constraint
        stloadfactor: float
        turnalt_m = 0.0
        turnspeed_ktas: float

    class DesignDefinition(BaseMethods):
        """Parameters of the aircraft design definition."""
        # Geometry definitions
        aspectratio = 8.0
        sweep_le_deg = 0.0
        sweep_mt_deg: float
        sweep_25_deg: float
        taperratio: float
        # Weight and loading
        weight_n: float = None
        weightfractions = {
            x: 1.0
            for x in ["climb", "cruise", "servceil", "take-off", "turn"]
        }

        def __init__(self, definition: dict):
            super().__init__(dictionary=definition)

            if not hasattr(self, "sweep_mt_deg"):
                self.sweep_mt_deg = self.sweep_le_deg

            if not hasattr(self, "sweep_25_deg"):
                self.sweep_25_deg = (
                        (2 / 7) * self.sweep_le_deg + (
                        5 / 7) * self.sweep_mt_deg)

            if not hasattr(self, "taperratio"):
                # Optimal root taper ratio (if one is not provided)
                # https://www.fzt.haw-hamburg.de/pers/Scholz/OPerA/OPerA_PRE_DLRK_12-09-10_MethodOnly.pdf
                self.taperratio = 0.45 * np.exp(-0.0375 * self.sweep_25_deg)

    class DesignPerformance(BaseMethods):

        # Drag/resistance coefficients
        CD0TO: float
        CDTO: float
        CDmin = 0.03
        mu_R = 0.03
        # Lift coefficients
        CL0 = 0.0
        CLTO = 0.95
        CLalpha = 5.2
        CLmax: float
        CLmaxHL: float
        CLmaxTO = 1.5
        CLmin: float
        CLminD = 0.2
        CLminHL: float
        # Propulsive efficiencies
        eta_prop = dict([
            ("climb", 0.75), ("cruise", 0.85), ("servceil", 0.65),
            ("take-off", 0.45), ("turn", 0.85)
        ])

    return DesignBrief, DesignDefinition, DesignPerformance


_class_brief, _class_definition, _class_performance = (
    get_default_concept_design_objects())


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
    brief: _class_brief
    design: _class_definition
    performance: _class_performance

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

                CLalpha
                    Float. The three-dimensional lift curve slope of the
                    aircraft, per radian. Optional, defaults to CLalpha=5.2.

                CLmax
                    Float. Maximum lift coefficient in flight, with the aircraft
                    in a clean configuration.

                CLmaxHL
                    Float. Maximum lift coefficient in flight, with the aircraft
                    in a high-lift configuration.

                CLmaxTO
                    Float. Maximum lift coefficient in take-off conditions.
                    Optional, defaults to 1.5.

                CLmin
                    Float. Minimum lift coefficient in flight, with the aircraft
                    in a clean configuration.

                CLminD
                    Float. The coefficient of lift at which the drag coefficient
                    is minimised, in clean configuration. Optional, defaults
                    to 0.2.

                CLminHL
                    Float. Minimum lift coefficient in flight, with the aircraft
                    in a high-lift configuration.

                eta_prop
                    Dictionary. Propeller efficiency in various phases of the
                    mission. It should contain the following keys: *take-off*,
                    *climb*, *cruise*, *turn*, *servceil*. Optional, efficiency
                    defaults to 0.45, 0.75, 0.85, 0.85, and 0.65, respectively.

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

        # Recast as necessary
        self.brief = _class_brief(brief)
        self.design = _class_definition(design)
        self.performance = _class_performance(performance)
        self.designatm = at.Atmosphere() if atmosphere is None else atmosphere

        # Propulsion handling
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
            aspectratio = self.design.aspectratio
            sweep = np.radians(self.design.sweep_le_deg)

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
            aspectratio = self.design.aspectratio
            sweep = np.radians(self.design.sweep_mt_deg)

            # Compute Oswald span efficiency estimate
            sqrtterm = 4 + aspectratio ** 2 * (1 + (np.tan(sweep)) ** 2)
            e0_estimate = 2 / (2 - aspectratio + np.sqrt(sqrtterm))

            cdi_factor = 1 / (np.pi * e0_estimate * aspectratio)

        elif method == "Nita-Scholz":
            # Find quarter chord sweep, wing taper ratio, and aspect ratio
            sweep = self.design.sweep_25_deg
            taperratio = self.design.taperratio
            aspectratio = self.design.aspectratio

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
            aspectratio = self.design.aspectratio

            e0_estimate = 1 / (1.05 + 0.007 * np.pi * aspectratio)

            cdi_factor = 1 / (np.pi * e0_estimate * aspectratio)

        elif method == "Kroo":
            # Find the inviscid induced drag factor, aspect ratio, and CDmin
            cdi_inviscid = self.cdi_factor(method="Brandt")
            aspectratio = self.design.aspectratio
            CDmin = self.performance.CDmin

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
        CDmin = self.performance.CDmin
        CLmax = self.performance.CLmax

        # Differentiated simple drag model, solve CD0 - k CL^2 = 0.0
        k = self.cdi_factor(method="Nita-Scholz", **kwargs)
        CL = (CDmin / k) ** 0.5
        if CL > CLmax:
            warnmsg = f"Couldn't achieve (L/D)max, needed {CL=} ({CLmax=})"
            warnings.warn(warnmsg, RuntimeWarning)
            CL = np.clip(CL, None, CLmax)

        LDmax = CL / (CDmin + k * CL ** 2)
        return LDmax

    def get_bestclimbangle_Vx(self, wingloading_pa, altitude_m=None):
        """
        Estimate the speed, Vx, which results in the best angle of climb.

        Args:
            wingloading_pa: Aircraft wing loading, in Pascal.
            altitude_m: Geopotential altitude, in metres.

        Returns:
            Best climb angle speed Vx, in metres per second.

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        """
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        altitude_m = 0 if altitude_m is None else altitude_m
        altitude_m = actools.recastasnpfloatarray(altitude_m)
        wingloading_pa, altitude_m \
            = np.broadcast_arrays(wingloading_pa, altitude_m)
        CDmin = self.performance.CDmin
        weight_n = self.design.weight_n

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

            eta_prop = self.performance.eta_prop["climb"]
            weight_n = self.design.weight_n

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

    def get_bestclimbrate_Vy(self, wingloading_pa, altitude_m=None):
        """
        Estimate the speed, Vy, which results in the best rate of climb.

        Args:
            wingloading_pa: Aircraft wing loading, in Pascal.
            altitude_m: Geopotential altitude, in metres. Optional, defaults to
                sea-level (0 metres).

        Returns:
            Best rate of climb speed Vy, in metres per second.

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        """
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        altitude_m = 0 if altitude_m is None else altitude_m
        altitude_m = actools.recastasnpfloatarray(altitude_m)

        wingloading_pa, altitude_m \
            = np.broadcast_arrays(wingloading_pa, altitude_m)
        CDmin = self.performance.CDmin

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
                weight_n = self.design.weight_n
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
        climbalt_m = self.brief.climbalt_m
        climbspeed_kias = self.brief.climbspeed_kias
        climbrate_fpm = self.brief.climbrate_fpm
        CDmin = self.performance.CDmin
        CLmax = self.performance.CLmax
        CLminD = self.performance.CLminD

        # Determine the thrust and power lapse corrections
        climbspeed_mpsias = co.kts2mps(climbspeed_kias)
        climbspeed_mpstas = self.designatm.eas2tas(
            eas=climbspeed_mpsias,
            altitude_m=climbalt_m
        )
        mach = climbspeed_mpstas / self.designatm.vsound_mps(climbalt_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude=climbalt_m, norm=True,
            eta_prop=self.performance.eta_prop["climb"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=climbalt_m, norm=True
            ) * self.performance.eta_prop["climb"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = self.design.weightfractions["climb"]

        # Compute climb constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=climbspeed_mpstas, altitude_m=climbalt_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = np.inf if CLmax is None else CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            # warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... load factor due to climb
        climbrate_mps = co.fpm2mps(climbrate_fpm)
        climbrate_mpstroc = self.designatm.eas2tas(
            eas=climbrate_mps, altitude_m=climbalt_m)
        cos_theta = (1 - (climbrate_mpstroc / climbspeed_mpstas) ** 2) ** 0.5

        # ... coefficient of drag
        k = self.cdi_factor(mach=mach, method="Nita-Scholz")
        f_CD = make_modified_drag_model(CDmin, k, CLmax, CLminD)
        CL = ws_pa * cos_theta / q_pa
        CD = f_CD(CL)

        # ... "acceleration factor"
        Ka = 1.0  # Small climb angle approximation! dV/dh ~ 0...

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CD / ws_pa
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
        cruisealt_m = self.brief.cruisealt_m
        cruisespeed_ktas = self.brief.cruisespeed_ktas
        cruisethrustfact = self.brief.cruisethrustfact
        CDmin = self.performance.CDmin
        CLmax = self.performance.CLmax
        CLminD = self.performance.CLminD

        # Determine the thrust and power lapse corrections
        cruisespeed_mpstas = co.kts2mps(cruisespeed_ktas)
        mach = cruisespeed_mpstas / self.designatm.vsound_mps(cruisealt_m)
        tcorr = cruisethrustfact * self.propulsion.thrust(
            mach=mach, altitude=cruisealt_m, norm=True,
            eta_prop=self.performance.eta_prop["cruise"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=cruisealt_m, norm=True
            ) * self.performance.eta_prop["cruise"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = self.design.weightfractions["cruise"]

        # Compute cruise constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=cruisespeed_mpstas, altitude_m=cruisealt_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = np.inf if CLmax is None else CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            # warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... coefficient of drag
        k = self.cdi_factor(mach=mach, method="Nita-Scholz")
        f_CD = make_modified_drag_model(CDmin, k, CLmax, CLminD)
        CL = ws_pa / q_pa
        CD = f_CD(CL)

        # ... T/W (mapped to sea-level static)
        tw = (q_pa * CD / ws_pa) / tcorr

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
        servceil_m = self.brief.servceil_m
        secclimbspd_kias = self.brief.secclimbspd_kias
        CDmin = self.performance.CDmin
        CLmax = self.performance.CLmax
        CLminD = self.performance.CLminD

        # Determine the thrust and power lapse corrections
        secclimbspd_mpsias = co.kts2mps(secclimbspd_kias)
        secclimbspd_mpstas = self.designatm.eas2tas(
            eas=secclimbspd_mpsias,
            altitude_m=servceil_m
        )
        mach = secclimbspd_mpstas / self.designatm.vsound_mps(servceil_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude=servceil_m, norm=True,
            eta_prop=self.performance.eta_prop["servceil"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=servceil_m, norm=True
            ) * self.performance.eta_prop["servceil"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = self.design.weightfractions["servceil"]

        # Compute climb constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=secclimbspd_mpstas, altitude_m=servceil_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = np.inf if CLmax is None else CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            # warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... load factor due to climb
        # Service ceiling typically defined in terms of climb rate (at best
        # climb speed) dropping to 100feet/min ~ 0.508m/s
        climbrate_mps = co.fpm2mps(100)
        # What true climb rate does 100 feet/minute correspond to?
        climbrate_mpstroc = self.designatm.eas2tas(
            eas=climbrate_mps, altitude_m=servceil_m)
        cos_theta = (1 - (climbrate_mpstroc / secclimbspd_mpstas) ** 2) ** 0.5

        # ... coefficient of drag
        k = self.cdi_factor(mach=mach, method="Nita-Scholz")
        f_CD = make_modified_drag_model(CDmin, k, CLmax, CLminD)
        CL = ws_pa * cos_theta / q_pa
        CD = f_CD(CL)

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CD / ws_pa
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
        groundrun_m = self.brief.groundrun_m
        rwyelevation_m = self.brief.rwyelevation_m
        # to_headwind_kts = self.brief.to_headwind_kts
        # to_slope_perc = self.brief.to_slope_perc
        CDTO = self.performance.CDTO
        mu_R = self.performance.mu_R
        CLTO = self.performance.CLTO
        CLmaxTO = self.performance.CLmaxTO

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = self.design.weightfractions["take-off"]

        # Approximation for lift-off speed
        ws_pa = wingloading_pa * wcorr
        airdensity_kgpm3 = self.designatm.airdens_kgpm3(rwyelevation_m)
        vs1to_mps = (2 * ws_pa / airdensity_kgpm3 / CLmaxTO) ** 0.5
        vrotate_mps = vs1to_mps
        vliftoff_mps = vrotate_mps

        # Determine the thrust and power lapse corrections
        # (use a representative speed from centre of mass of the integral)
        vbar_mpstas = 0.75 * vliftoff_mps
        machbar = vbar_mpstas / self.designatm.vsound_mps(rwyelevation_m)
        # (maybe we should use the altitude=0 sea-level take-off method below?)
        tcorr = self.propulsion.thrust(
            mach=machbar, altitude=rwyelevation_m, norm=True,
            eta_prop=self.performance.eta_prop["take-off"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=machbar, altitude=rwyelevation_m, norm=True
            ) * self.performance.eta_prop["take-off"]
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
        stloadfactor = self.brief.stloadfactor
        turnalt_m = self.brief.turnalt_m
        turnspeed_ktas = self.brief.turnspeed_ktas
        CDmin = self.performance.CDmin
        CLmax = self.performance.CLmax
        CLminD = self.performance.CLminD

        # Determine the thrust and power lapse corrections
        turnspeed_mpstas = co.kts2mps(turnspeed_ktas)
        mach = turnspeed_mpstas / self.designatm.vsound_mps(turnalt_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude=turnalt_m, norm=True,
            eta_prop=self.performance.eta_prop["turn"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=turnalt_m, norm=True
            ) * self.performance.eta_prop["turn"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = self.design.weightfractions["turn"]

        # Compute climb constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=turnspeed_mpstas, altitude_m=turnalt_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = np.inf if CLmax is None else CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            # warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... coefficient of drag
        k = self.cdi_factor(mach=mach, method="Nita-Scholz")
        f_CD = make_modified_drag_model(CDmin, k, CLmax, CLminD)
        CL = ws_pa * stloadfactor / q_pa
        CD = f_CD(CL)

        # ... T/W (mapped to sea-level static)
        tw = (q_pa * CD / ws_pa) / tcorr

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
        vstallclean_kcas = self.brief.vstallclean_kcas
        CLmax = self.performance.CLmax
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
        weight_n = self.design.weight_n
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
        weight_n = self.design.weight_n

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
            l2d = ax.axvline(
                xstall, label="clean stall, 1g", c=clr, zorder=10, lw=2)
            ax.fill_betweenx(*yx1x2, fc=l2d.get_color(), alpha=0.05, zorder=5)

        # Plot the stall limit due to all constraints
        nanindex = np.isnan(np.array(list(ys.values()))).sum(axis=1).max()
        yx1x2 = [0, ylim], xs[-nanindex], xs.min() if weight_n else xs.max()
        if nanindex > 0:
            kkka = (0.98039215686, 0.50196078431, 0.44705882352, 0.01)
            ax.fill_betweenx(
                *yx1x2, fc=kkka, ec="r", zorder=5, hatch="x", alpha=0.1,
                label=r"above $C_{L,max}$")
            ax.fill_betweenx(*yx1x2, fc=(0, 0, 0, 0), ec="fuchsia", lw=2,
                             label=r"above $C_{L,max}$")

        # Limits and positioning
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(0, ylim)
        ax.grid()

        # Custom legend behaviour - allow user to redraw legend w/default posns!
        ax.legend()
        # Squash the handles of legend handles shared by more than one label
        handles, labels = ax.get_legend_handles_labels()
        legenddict = {l: tuple(np.array(handles)[np.array(labels) == l])
                      for l in labels}
        legenddict = dict(zip(("labels", "handles"), zip(*legenddict.items())))

        def custom_legend_maker(**kwargs):
            """A matplotlib legend but with default position parameters."""
            default_legend_kwargs = dict([
                ("bbox_to_anchor", (1.05, 0.618)),
                ("loc", "center left")
            ])
            default_legend_kwargs.update(legenddict)
            return ax.legend(**{**default_legend_kwargs, **kwargs})

        ax.remake_legend = custom_legend_maker  # Assign the method to our ax
        ax.remake_legend()  # Use the method

        # Advanced labelling - Multiple unit labelling
        # All the magic below allows the Axes object to have primary axes for
        # the actual plot data to be in SI units, but display in other units
        if weight_n is None:
            xlabel_loc_metric = (1.09, -0.02)
            yloc = -0.12  # Shared y-location of secondary and tertiary axes
            # x, metric
            ax.set_xlabel("[Pa]")
            ax.xaxis.set_label_coords(*xlabel_loc_metric)
            # x, imperial
            ax2_x = ax.secondary_xaxis(yloc)
            ax2_x.set_xlabel("[lb$_f$/ft$^2$]")
            ax2_x.xaxis.set_label_coords(*xlabel_loc_metric)
            ax2_x.set_xticks([])
            # x, label
            ax3_x = ax.secondary_xaxis(yloc,
                                       functions=(co.pa2lbfft2, co.lbfft22pa))
            ax3_x.set_xlabel("Wing Loading")
            if type_is_power:
                ax.set_ylabel("Power-to-Weight")
            else:
                ax.set_ylabel("Thrust-to-Weight")
            return fig, ax
        # x, metric
        ax.set_xlabel("[m$^2$]")
        ax.xaxis.set_label_coords(1.09, -0.02)
        # x, imperial
        yloc = -0.12  # Shared y-location of secondary and tertiary axes
        ax2_x = ax.secondary_xaxis(yloc, functions=(co.m22feet2, co.feet22m2))
        ax2_x.set_xlabel("[ft$^2$]")
        ax2_x.xaxis.set_label_coords(1.09, -0.02)
        ax2_x.set_xticks([])
        # x, label
        ax3_x = ax.secondary_xaxis(yloc, functions=(co.m22feet2, co.feet22m2))
        ax3_x.set_xlabel("Wing Area")
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
