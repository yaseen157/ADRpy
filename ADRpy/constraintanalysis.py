"""
This module contains tools for the constraint analysis of fixed
wing aircraft.
"""
import math
import typing
import warnings
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants
from scipy.interpolate import interp1d

from ADRpy import atmospheres as at
from ADRpy import unitconversions as co
from ADRpy import mtools4acdc as actools
from ADRpy import propulsion as pdecks

__author__ = "András Sóbester"


# Other contributors: Yaseen Reza


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
                 designatm: at.Atmosphere = None,
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

                to_headwind_kts
                    Float. The speed of the take-off headwind (in knots),
                    parallel to the runway. Optional, defaults to zero.

                to_slope_perc
                    Float. The percent gradient of the runway in the direction
                    of travel. Optional, defaults to zero.

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

                roottaperratio
                    Float. Standard definition of wing tip chord to root chord
                    ratio, zero for sharp, pointed wing-tip delta wings.
                    Optional, defaults to the theoretical optimal value as a
                    function of the quarter-chord sweep angle.

                wingarea_m2
                    Float. Total reference area of the wing (in metres squared).

                wingheightratio
                    Float. The ratio of altitude h to wingspan b, used for the
                    calculation of ground effect. Optional, defaults to 100
                    (produces a ground effect factor of near unity).

                bpr
                    Float. Specifies the propulsion system type. For jet engines
                    (powered by axial gas turbines) this should be the bypass
                    ratio (hence *'bpr'*).

                    *Deprecated: Set to -1 for piston engines, -2 for turboprops
                    and -3 if no power/thrust corrections are needed
                    (e.g., for electric motors).

                spooluptime_s
                    Float. Time in seconds for the engine to reach take-off
                    thrust. Optional, defaults to 5.

                totalstaticthrust_n
                    Float. Maximum thrust achievable at zero airspeed.

                tr
                    Float. Throttle ratio for gas turbine engines. *tr = 1*
                    means that the Turbine Entry Temperature will reach its
                    maximum allowable value in sea level standard day
                    conditions, so higher ambient temperatures will result in
                    power loss. Higher *tr* values mean thrust decay starting at
                    higher altitudes.

                weight_n
                    Float. Specifies the maximum take-off weight of the
                    aircraft.

                weightfractions
                    Dictionary. Specifies at what fraction of the maximum
                    take-off weight do various constraints have to be met. It
                    should contain the following keys: *take-off*, *climb*,
                    *cruise*, *turn*, *servceil*. Optional, each defaults to
                    1.0 if not specified.

                runwayalpha_deg
                    Float. Angle of attack the main wing encounters during
                    take-off roll. Optional, defaults to 0.

                runwayalpha_max_deg
                    Float. Maximum permitted angle of attack before lift-off.

            performance: Definition of key, high level design performance
                estimates.

                CD0TO
                    Float. Zero-lift drag coefficient in the take-off
                    configuration.

                CDTO
                    Float. Take-off drag coefficient. Optional, defaults to
                    0.09.

                CDminclean
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

                CLmaxclean
                    Float. Maximum lift coefficient in flight, in clean
                    configuration.

                CLminclean
                    Float. Minimum lift coefficient in flight, in clean
                    configuration. Typically negative.

                CLslope
                    Float. Lift-curve slope gradient, or Cl/alpha of a design
                    aerofoil (or wing that may be considered 2D) in
                    incompressible flow. Optional, defaults to the flat plate
                    theory maximum of 2*Pi.

                etaprop
                    Dictionary. Propeller efficiency in various phases of the
                    mission. It should contain the following keys: *take-off*,
                    *climb*, *cruise*, *turn*, *servceil*. Optional, unspecified
                    entries in the dictionary default to the following values:

                    :code: `etap = {'take-off': 0.45, 'climb': 0.75, 'cruise': 0.85, 'turn': 0.85, 'servceil': 0.65}`

            designatm: `Atmosphere <https://adrpy.readthedocs.io/en/latest/#atmospheres.Atmosphere>`_
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
        designatm = at.Atmosphere() if designatm is None else designatm
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
        brief.setdefault("to_headwind_kts", 0.0)
        brief.setdefault("to_slope_perc", 0.0)
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
            "roottaperratio",
            # Optimal root taper ratio (if one is not provided)
            # https://www.fzt.haw-hamburg.de/pers/Scholz/OPerA/OPerA_PRE_DLRK_12-09-10_MethodOnly.pdf
            0.45 * np.exp(-0.0375 * design["sweep_25_deg"])
        )
        design.setdefault("wingarea_m2")
        design.setdefault("wingheightratio", 100.0)
        # Propulsion system
        design.setdefault("bpr", -1)
        design.setdefault("spooluptime_s", 5.0)
        design.setdefault("totalstaticthrust_n")
        design.setdefault("tr", 1.07)
        # Weight and loading
        design.setdefault("weight_n")
        design.setdefault("weightfractions", dict())
        design["weightfractions"].setdefault("climb", 1.0)
        design["weightfractions"].setdefault("cruise", 1.0)
        design["weightfractions"].setdefault("servceil", 1.0)
        design["weightfractions"].setdefault("take-off", 1.0)
        design["weightfractions"].setdefault("turn", 1.0)
        # Runway
        design.setdefault("runwayalpha_deg", 0.0)
        design.setdefault("runwayalpha_max_deg")

        # ----- CONCEPT PERFORMANCE HANDLING -----
        # Drag/resistance coefficients
        performance.setdefault("CD0TO")
        performance.setdefault("CDTO")
        performance.setdefault("CDminclean")
        performance.setdefault("mu_R")
        # Lift coefficients
        performance.setdefault("CLTO", 0.95)
        performance.setdefault("CLmaxTO", 1.5)
        performance.setdefault("CLmaxclean")
        performance.setdefault("CLminclean")
        performance.setdefault("CLslope", 2 * np.pi)
        # Propulsive efficiencies
        performance.setdefault("etaprop", dict())
        performance["etaprop"].setdefault("climb", 0.75)
        performance["etaprop"].setdefault("cruise", 0.85)
        performance["etaprop"].setdefault("servceil", 0.65)
        performance["etaprop"].setdefault("take-off", 0.45)
        performance["etaprop"].setdefault("turn", 0.85)

        # ----- PROPULSION HANDLING AND DEPRECATION -----
        propulsion = "Piston" if propulsion is None else propulsion
        if isinstance(propulsion, tuple):
            errormsg = (
                f"Providing a tuple of (EngineDeck, PropellerDeck) is no "
                f"longer supported. This message will be removed in a future "
                f"update"
            )
            raise FutureWarning(errormsg)
        elif propulsion == "jet":
            warnmsg = (
                f"Use of {propulsion=} is deprecated. Please instantiate an "
                f"object of the EngineDeck class, or alternatively use "
                f"'TurbofanHiBPR' for high bypass ratio (subsonic) engines, "
                f"or 'TurbofanLoBPR' and 'Turbojet' for low bypass "
                f"(high-subsonic/supersonic) engines"
            )
            warnings.warn(warnmsg, FutureWarning, stacklevel=2)
            bpr = design["bpr"]
            if bpr is None:
                errormsg = (
                    f"Deprecated propulsion type 'jet' did not also have a "
                    f"design (dict) bypass ratio 'bpr' specified by the user!"
                )
                raise ValueError(errormsg)
            elif bpr == 0:
                propulsion = "Turbojet"
            elif 0 < bpr <= 1:
                propulsion = "TurbofanLoBPR"
            elif 1 < bpr:
                propulsion = "TurbofanHiBPR"
            else:
                raise ValueError(f"Invalid bypass ratio {bpr=}")
        elif propulsion == "piston":
            warnmsg = (
                f"Use of {propulsion=} is deprecated. Please instantiate an "
                f"object of the EngineDeck class, or alternatively use 'Piston'"
            )
            warnings.warn(warnmsg, FutureWarning, stacklevel=2)
            propulsion = "Piston"
        elif propulsion == "turboprop":
            warnmsg = (
                f"Use of {propulsion=} is deprecated. Please instantiate an "
                f"object of the EngineDeck class, or alternatively use "
                f"'Turboprop'"
            )
            warnings.warn(warnmsg, FutureWarning, stacklevel=2)
            propulsion = "Turboprop"
        elif propulsion == "electric":
            errormsg = f"Use of {propulsion=} will be deprecated soon."
            raise DeprecationWarning(errormsg)
        elif isinstance(
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
        self.designatm = designatm
        self.propulsion = propulsion

        return

    # Populate the AircraftConcept object with attributes
    @property
    def climbalt_m(self):
        return getattr(self.brief, "climbalt_m")

    @property
    def climbspeed_kias(self):
        return getattr(self.brief, "climbspeed_kias")

    @property
    def climbrate_fpm(self):
        return getattr(self.brief, "climbrate_fpm")

    @property
    def cruisealt_m(self):
        return getattr(self.brief, "cruisealt_m")

    @property
    def cruisespeed_ktas(self):
        return getattr(self.brief, "cruisespeed_ktas")

    @property
    def cruisethrustfact(self):
        return getattr(self.brief, "cruisethrustfact")

    @property
    def servceil_m(self):
        return getattr(self.brief, "servceil_m")

    @property
    def secclimbspd_kias(self):
        return getattr(self.brief, "secclimbspd_kias")

    @property
    def vstallclean_kcas(self):
        return getattr(self.brief, "vstallclean_kcas")

    @property
    def groundrun_m(self):
        return getattr(self.brief, "groundrun_m")

    @property
    def rwyelevation_m(self):
        return getattr(self.brief, "rwyelevation_m")

    @property
    def to_headwind_kts(self):
        return getattr(self.brief, "to_headwind_kts")

    @property
    def to_slope_perc(self):
        return getattr(self.brief, "to_slope_perc")

    @property
    def to_slope_rad(self):
        return math.atan(self.to_slope_perc / 100)

    @property
    def to_slope_deg(self):
        return math.degrees(self.to_slope_rad)

    @property
    def turnalt_m(self):
        return getattr(self.brief, "turnalt_m")

    @property
    def turnspeed_ktas(self):
        return getattr(self.brief, "turnspeed_ktas")

    @property
    def stloadfactor(self):
        return getattr(self.brief, "stloadfactor")

    @property
    def aspectratio(self):
        return getattr(self.design, "aspectratio")

    @property
    def sweep_le_deg(self):
        return getattr(self.design, "sweep_le_deg")

    @property
    def sweep_le_rad(self):
        return math.radians(self.sweep_le_deg)

    @property
    def sweep_25_deg(self):
        return getattr(self.design, "sweep_25_deg")

    @property
    def sweep_25_rad(self):
        return math.radians(self.sweep_25_deg)

    @property
    def sweep_mt_deg(self):
        return getattr(self.design, "sweep_mt_deg")

    @property
    def sweep_mt_rad(self):
        return math.radians(self.sweep_mt_deg)

    @property
    def roottaperratio(self):
        return getattr(self.design, "roottaperratio")

    @property
    def wingarea_m2(self):
        return getattr(self.design, "wingarea_m2")

    @property
    def wingheightratio(self):
        return getattr(self.design, "wingheightratio")

    @property
    def bpr(self):
        return getattr(self.design, "bpr")

    @property
    def spooluptime_s(self):
        return getattr(self.design, "spooluptime_s")

    @property
    def totalstaticthrust_n(self):
        return getattr(self.design, "totalstaticthrust_n")

    @property
    def throttle_r(self):
        return getattr(self.design, "throttle_r")

    @property
    def weight_n(self):
        return getattr(self.design, "weight_n")

    @property
    def climb_weight_fraction(self):
        return getattr(self.design, "weightfractions")["climb"]

    @property
    def cruise_weight_fraction(self):
        return getattr(self.design, "weightfractions")["cruise"]

    @property
    def sec_weight_fraction(self):
        return getattr(self.design, "weightfractions")["servceil"]

    @property
    def takeoff_weight_fraction(self):
        return getattr(self.design, "weightfractions")["take-off"]

    @property
    def turn_weight_fraction(self):
        return getattr(self.design, "weightfractions")["turn"]

    @property
    def runwayalpha_deg(self):
        return getattr(self.design, "runwayalpha_deg")

    @property
    def runwayalpha_max_deg(self):
        return getattr(self.design, "runwayalpha_max_deg")

    @property
    def CD0TO(self):
        return getattr(self.performance, "CD0TO")

    @property
    def CDTO(self):
        return getattr(self.performance, "CDTO")

    @property
    def CDminclean(self):
        return getattr(self.performance, "CDminclean")

    @property
    def mu_R(self):
        return getattr(self.performance, "mu_R")

    @property
    def CLTO(self):
        return getattr(self.performance, "CLTO")

    @property
    def CLmaxclean(self):
        return getattr(self.performance, "CLmaxclean")

    @property
    def CLminclean(self):
        return getattr(self.performance, "CLminclean")

    @property
    def CLmaxTO(self):
        return getattr(self.performance, "CLmaxTO")

    @property
    def CLslope(self):
        return getattr(self.performance, "CLslope")

    @property
    def etaprop_climb(self):
        return getattr(self.performance, "etaprop")["climb"]

    @property
    def etaprop_cruise(self):
        return getattr(self.performance, "etaprop")["cruise"]

    @property
    def etaprop_sec(self):
        return getattr(self.performance, "etaprop")["servceil"]

    @property
    def etaprop_to(self):
        return getattr(self.performance, "etaprop")["take-off"]

    @property
    def etaprop_turn(self):
        return getattr(self.performance, "etaprop")["turn"]

    def __getattr__(self, item):

        # Method handling
        method_redirectory = {
            "twrequired_clm": self.constrain_climb,
            "twrequired_crs": self.constrain_cruise,
            "twrequired_sec": self.constrain_servceil,
            "thrusttoweight_takeoff": self.constrain_takeoff,
            "twrequired_to": self.constrain_takeoff,
            "thrusttoweight_sustainedturn": self.constrain_turn,
            "twrequired_trn": self.constrain_turn,
        }
        # Attribute handling
        attribute_redirectory = {
            "a_0i": getattr(self.performance, "CLslope")
        }

        if item in method_redirectory:
            method = method_redirectory[item]
            warnmsg = (
                f"'{item}' is a deprecated method of {self}. Please use "
                f"'{type(self).__name__}.{method.__name__}' instead."
            )
            warnings.warn(warnmsg, FutureWarning, stacklevel=2)
            return method

        elif item in attribute_redirectory:
            attribute = attribute_redirectory[item]
            warnmsg = (
                f"'{item}' is a deprecated attribute of {self}. Please consult "
                f"the user documentation on alternative, helpful attributes."
            )
            warnings.warn(warnmsg, FutureWarning, stacklevel=2)
            return attribute

        # Attributes don't exist...
        elif item in ["twrequired", "powerrequired"]:
            errormsg = (
                f"'{item}' is a deprecated method of {self}. Please use the"
                f"unique T/W, P/W functions on a constraint-by-constraint basis"
            )
        elif item == "designspace":
            errormsg = f"Sorry, the attribute {item} does not exist for {self}."
            errormsg += " It was deprecated (for duplicating other attributes)"
        elif item == "designstate":
            errormsg = f"Sorry, the attribute {item} does not exist for {self}."
            errormsg += " It was deprecated (for duplicating other attributes)"
        else:
            errormsg = f"Sorry, the attribute {item} does not exist for {self}."

        raise AttributeError(errormsg)

    # Three different estimates the Oswald efficiency factor:

    def oswaldspaneff1(self):
        """Raymer's Oswald span efficiency estimate, sweep < 30, moderate AR"""
        return 1.78 * (1 - 0.045 * (self.aspectratio ** 0.68)) - 0.64

    def oswaldspaneff2(self):
        """Oswald span efficiency estimate due to Brandt et al."""
        sqrtterm = 4 + self.aspectratio ** 2 * (
                1 + (math.tan(self.sweep_mt_rad)) ** 2)
        return 2 / (2 - self.aspectratio + math.sqrt(sqrtterm))

    def oswaldspaneff3(self):
        """Raymer's Oswald span efficiency estimate, swept wings"""
        return 4.61 * (1 - 0.045 * (self.aspectratio ** 0.68)) * (
                (math.cos(self.sweep_le_rad)) ** 0.15) - 3.1

    def oswaldspaneff4(self, mach_inf=None):
        """Method for estimating the oswald factor from basic aircraft geometrical parameters;
        Original method by Mihaela Nita and Dieter Scholz, Hamburg University of Applied Sciences
        https://www.dglr.de/publikationen/2012/281424.pdf

        The method returns an estimate for the Oswald efficiency factor of a planar wing. The mach
        correction factor was fitted around subsonic transport aircraft, and therefore this method
        is recommended only for use in subsonic analysis with free-stream Mach < 0.69.

        **Parameters:**

        mach_inf
            float, Mach number at which the Oswald efficiency factor is to be estimated, required
            to evaluate compressibility effects. Optional, defaults to 0.3 (incompressible flow).

        **Outputs:**

        e
            float, predicted Oswald efficiency factor for subsonic transport aircraft.

        """

        if mach_inf is None:
            argmsg = 'Mach number unspecified, defaulting to incompressible flow condition'
            warnings.warn(argmsg, RuntimeWarning)
            mach_inf = 0.3

        # THEORETICAL OSWALD FACTOR: For calculating the inviscid drag due to lift only

        taperratio = self.roottaperratio

        # Calculate Hoerner's delta/AR factor for unswept wings (with NASA's swept wing study, fitted for c=25% sweep)
        dtaperratio = -0.357 + 0.45 * math.exp(-0.0375 * self.sweep_25_deg)
        tapercorr = taperratio - dtaperratio
        k_hoernerfactor = (0.0524 * tapercorr ** 4) - (
                0.15 * tapercorr ** 3) + (0.1659 * tapercorr ** 2) - (
                                  0.0706 * tapercorr) + 0.0119
        e_theo = 1 / (1 + k_hoernerfactor * self.aspectratio)

        # CORRECTION FACTOR F: Kroo's correction factor due to reduced lift from fuselage presence
        dfoverb_all = 0.114
        ke_fuse = 1 - 2 * (dfoverb_all ** 2)

        # CORRECTION FACTOR D0: Correction factor due to viscous drag from generated lift
        ke_d0 = 0.85

        # CORRECTION FACTOR M: Correction factor due to compressibility effects on induced drag
        mach_compressible = 0.3
        # M. Nita and D. Scholz, constants from statistical analysis of subsonic aircraft
        if mach_inf > mach_compressible:
            ke_mach = -0.001521 * (
                    ((mach_inf / mach_compressible) - 1) ** 10.82) + 1
        else:
            ke_mach = 1

        e = e_theo * ke_fuse * ke_d0 * ke_mach

        if e < 1e-3:
            e = 1e-3
            calcmsg = 'Specified Mach ' + str(
                mach_inf) + ' is out of bounds for oswaldspaneff4, e_0 ~= 0'
            warnings.warn(calcmsg, RuntimeWarning)

        return e

    def induceddragfact(self, whichoswald=None, mach_inf=None):
        """Lift induced drag factor k estimate (Cd = Cd0 + K.Cl^2) based on the relationship
            (k = 1 / pi * AR * e_0).

        **Parameters:**

        whichoswald
            integer, used to specify the method(s) to estimate e_0 from. Specifying a single digit
            integer selects a single associated method, however a concatenated string of integers
            can be used to specify that e_0 should be calculated from the average of several.
            Optional, defaults to methods 2 and 4.

        mach_inf
            float, the free-stream flight mach number. Optional, defaults to 0.3 (incompressible
            flow prediction).

        **Outputs:**

        induceddragfactor
            float, an estimate for the coefficient of Cl^2 in the drag polar (Cd = Cd0 + K.Cl^2)
            based on various estimates of the oswald efficiency factor.

        **Note**
        This method does not contain provisions for 'wing-in-ground-effect' factors.

        """

        # Identify all the digit characters passed in the whichoswald argument, and assemble as a list of single digits
        oswaldeff_list = []
        if type(whichoswald) == int:
            selection_list = [int(i) for i in str(whichoswald)]
            # k = 1 / pi.AR.e
            if 1 in selection_list:
                oswaldeff_list.append(self.oswaldspaneff1())
            if 2 in selection_list:
                oswaldeff_list.append(self.oswaldspaneff2())
            if 3 in selection_list:
                oswaldeff_list.append(self.oswaldspaneff3())
            if 4 in selection_list:
                # If whichoswald = 4 was *specifically* selected, then throw a warning if Mach was not given
                if mach_inf is None:
                    argmsg = 'Mach number unspecified, defaulting to incompressible flow condition'
                    warnings.warn(argmsg, RuntimeWarning)
                    mach_inf = 0.3
                oswaldeff_list.append(self.oswaldspaneff4(mach_inf=mach_inf))

        # If valid argument(s) were given, take the average of their Oswald results
        if len(oswaldeff_list) > 0:
            oswaldeff = sum(oswaldeff_list) / len(oswaldeff_list)
        # Else default to estimate 2 and 4, Brandt and Nita incompressible
        else:
            oswaldeff = 0.5 * (self.oswaldspaneff2() + self.oswaldspaneff4(
                mach_inf=0.3))

        return 1.0 / (math.pi * self.aspectratio * oswaldeff)

    def findchordsweep_rad(self, xc_findsweep):
        """Calculates the sweep angle at a given chord fraction, for a constant taper wing

        **Parameters:**

        xc_findsweep
            float, the fraction of chord along which the function is being asked to determine the
            sweep angle of. Inputs are bounded as 0 <= xc_findsweep <= 1 (0% to 100% chord),
            where x/c = 0 is defined as the leading edge.

        **Outputs:**

        sweep_rad
            float, this is the sweep angle of the given chord fraction, for a constant taper wing.
        """

        if xc_findsweep is None:
            argmsg = 'Function can not find the sweep angle without knowing the x/c to investigate'
            raise ValueError(argmsg)

        elif not (0 <= xc_findsweep <= 1):
            argmsg = 'Function was called with an out of bounds chord, tried (0 <= x/c <= 1)'
            raise ValueError(argmsg)

        sweeple_rad = self.sweep_le_rad
        sweep25_rad = self.sweep_25_rad

        # Use rate of change of sweep angle with respect to chord progression
        sweep_roc = (sweeple_rad - sweep25_rad) / -0.25
        sweep_rad = sweeple_rad + sweep_roc * xc_findsweep

        return sweep_rad

    def liftslope_prad(self, mach_inf=None):
        """Method for estimating the lift-curve slope from aircraft geometry; Methods from
        http://naca.central.cranfield.ac.uk/reports/arc/rm/2935.pdf (Eqn. 80), by D. Kuchemann;
        DATCOM 1978;

        Several methods for calculating supersonic and subsonic lift-slopes are aggregated to
        produce a model for the lift curve with changing free-stream Mach number.

        **Parameters:**

        mach_inf
            float, the free-stream flight mach number. Optional, defaults to 0.3 (incompressible
            flow prediction).

        **Outputs:**

        liftslope_prad
            float, the predicted lift slope as an average of several methods of computing it, for
            a 'thin' aerofoil (t/c < 5%) - assuming the aircraft is designed with supersonic flight
            in mind. Units of rad^-1.

        **Note**

        Care must be used when interpreting this function in the transonic flight regime. This
        function departs from theoretical models for 0.6 <= Mach_free-stream <= 1.4, and instead
        uses a weighted average of estimated curve-fits and theory to predict transonic behaviour.

        """

        if mach_inf is None:
            argmsg = 'Mach number unspecified, defaulting to incompressible flow condition'
            warnings.warn(argmsg, RuntimeWarning)
            mach_inf = 0.3

        aspectr = self.aspectratio
        a_0i = self.a_0i
        piar = math.pi * aspectr
        oswald = self.oswaldspaneff2()
        sweep_25_rad = self.sweep_25_rad
        sweep_le_rad = self.sweep_le_rad

        # Define transition points of models by Mach
        puresubsonic_mach = 0.6
        puresupsonic_mach = 1.4
        lowertranson_mach = 0.8
        uppertranson_mach = 1.3

        def a_subsonic(machsub):
            """From subsonic mach, determine an approximate lift slope"""
            slopeslist_sub = []

            beta_00 = math.sqrt(1 - machsub ** 2)
            beta_le = math.sqrt(1 - (machsub * math.cos(sweep_le_rad)) ** 2)

            # Subsonic 3-D Wing Lift Slope, with Air Compressibility and Sweep Effects
            sqrt_term = 1 + (piar / a_0i / math.cos(
                sweep_25_rad)) ** 2 * beta_00 ** 2
            a_m0 = piar / (1 + math.sqrt(sqrt_term))
            slopeslist_sub.append(a_m0)

            # High-Aspect-Ratio Straight Wings
            a_m3 = a_0i / (beta_00 + (a_0i / piar / oswald))
            slopeslist_sub.append(a_m3)

            # Low-Aspect-Ratio Straight Wings
            a_m4 = a_0i / (math.sqrt(beta_00 ** 2 + (a_0i / piar) ** 2) + (
                    a_0i / piar))
            slopeslist_sub.append(a_m4)

            # Low-Aspect-Ratio Swept Wings
            a_0_le = a_0i * math.cos(sweep_le_rad)
            a_m5 = a_0_le / (math.sqrt(beta_le ** 2 + (a_0_le / piar) ** 2) + (
                    a_0_le / piar))
            slopeslist_sub.append(a_m5)

            # DATCOM model for sub-sonic lift slope
            sweep_50_rad = self.findchordsweep_rad(xc_findsweep=0.5)
            kappa = a_0i / (
                    2 * math.pi)  # Implementation of 2D lift slope needs checking here
            a_m6 = (2 * piar) / (
                    2 + math.sqrt((aspectr * beta_00 / kappa) ** 2 * (
                    1 + (math.tan(sweep_50_rad) / beta_00) ** 2) + 4))
            slopeslist_sub.append(a_m6)

            # D. Kuchemann's method for subsonic, straight or swept wings
            sweep_50_rad = self.findchordsweep_rad(xc_findsweep=0.5)
            a_0c = a_0i / beta_00
            a_0_50 = a_0c * math.cos(
                sweep_50_rad)  # Mid-chord sweep, lift slope
            sweep_eff = sweep_50_rad / (1 + (a_0_50 / piar) ** 2) ** 0.25
            a_0eff = a_0c * math.cos(sweep_eff)  # Effective sweep
            powerterm = 1 / (4 * (1 + (abs(sweep_eff) / (0.5 * math.pi))))
            n_s = 1 - (1 / (2 * (1 + (
                    a_0eff / piar) ** 2) ** powerterm))  # Shape parameter, swept wing
            a_m7 = ((2 * a_0eff * n_s) / (
                    1 - (math.pi * n_s) * (1 / math.tan(math.pi * n_s)) + (
                    (4 * a_0eff * n_s ** 2) / piar)))
            slopeslist_sub.append(a_m7)

            # D. Kuchemann's method for delta wings with pointed tips and straight trailing edges, up to AR ~ 2.5
            a_0c = a_0i / beta_00
            a_m8 = (a_0c * aspectr) / (math.sqrt(
                4 + aspectr ** 2 + (a_0c / math.pi) ** 2) + a_0c / math.pi)
            slopeslist_sub.append(a_m8)

            return sum(slopeslist_sub) / len(slopeslist_sub)

        def a_supersonic(machsuper):
            """From supersonic mach, determine an approximate lift slope"""
            slopeslist_sup = []

            beta_00 = math.sqrt(machsuper ** 2 - 1)

            # Supersonic Delta Wings
            if sweep_le_rad != 0:  # Catch a divide by zero if the LE sweep is zero (can't be a delta wing)
                if machsuper < 1:
                    sweep_shock_rad = 0
                else:
                    sweep_shock_rad = math.acos(
                        1 / machsuper)  # NOT MACH ANGLE, this is sweep! Mach 1 = 0 deg sweep

                m = math.tan(sweep_shock_rad) / math.tan(sweep_le_rad)
                if 0 <= m <= 1:  # Subsonic leading edge case
                    lambda_polynomial = m * (
                            0.38 + (2.26 * m) - (0.86 * m ** 2))
                    a_m2 = (2 * math.pi ** 2 * (1 / math.tan(sweep_le_rad))) / (
                            math.pi + lambda_polynomial)
                else:  # Supersonic leading edge case, linear inviscid theory
                    a_m2 = 4 / beta_00
                slopeslist_sup.append(a_m2)

            # High-Aspect-Ratio Straight Wings
            a_m3 = 4 / beta_00
            slopeslist_sup.append(a_m3)

            # Low-Aspect-Ratio Straight Wings
            a_m4 = 4 / beta_00 * (1 - (1 / 2 / aspectr / beta_00))
            slopeslist_sup.append(a_m4)

            return sum(slopeslist_sup) / len(slopeslist_sup)

        if mach_inf < puresubsonic_mach:  # Subsonic regime, Mach_inf < mach_sub
            liftslope_prad = a_subsonic(mach_inf)

        elif mach_inf > puresupsonic_mach:  # Supersonic regime, Mach_inf > mach_sup
            liftslope_prad = a_supersonic(mach_inf)

        else:  # Transonic regime, mach_sub < Mach_inf < mach_sup

            # Thickness-to-chord ratio
            tcratio = 0.05

            # Find where the lift-slope peaks
            def slopepeak_mach(aspectratio):
                # http://naca.central.cranfield.ac.uk/reports/1955/naca-report-1253.pdf
                # Assume quadratic fit of graph data from naca report 1253
                def genericquadfunc(x, a, b, c):
                    return a * x ** 2 + b * x + c

                # These are pregenerated static values, to save computational resources
                popt = np.array([3.2784414, -9.73119668, 6.0546588])

                # Create x-data for A(t/c)^(1/3), and then y-data for Speed parameter (M ** 2 - 1) / ((t/c) ** (2/3))
                xfitdata = np.linspace(0, 1.5, 200)
                yfitdata = genericquadfunc(xfitdata, *popt)

                # Convert to x-data for AR, and y-data for Mach
                arfitdata = xfitdata / (tcratio ** (1 / 3))
                machfitdata = (yfitdata * (tcratio ** (2 / 3)) + 1) ** 0.5

                # For a provided aspect ratio, determine if the quadratic or the linear relation should be used
                arinfit_index = np.where(arfitdata >= aspectratio)[0]
                if len(arinfit_index) > 0:
                    machquery = machfitdata[min(arinfit_index)]
                else:
                    machquery = min(machfitdata)
                return machquery

            # This is the Mach number where the liftslope_prad should peak
            mach_apk = slopepeak_mach(aspectratio=aspectr)
            cla_son = (
                              math.pi / 2) * aspectr  # Strictly speaking, an approximation only true for A(t/c)^(1/3) < 1

            delta = 3e-2
            x_mach = []
            y_cla = []
            # Subsonic transition points
            x_mach.extend([puresubsonic_mach, puresubsonic_mach + delta])
            y_cla.extend(
                [a_subsonic(machsub=x_mach[0]), a_subsonic(machsub=x_mach[1])])

            # Lift-slope peak transition points
            x_mach.extend([mach_apk - 2 * delta, mach_apk + 2 * delta])
            y_cla.extend([0.95 * cla_son, 0.95 * cla_son])

            # Supersonic transition points
            x_mach.extend([puresupsonic_mach - delta, puresupsonic_mach])
            y_cla.extend([a_supersonic(machsuper=x_mach[-2]),
                          a_supersonic(machsuper=x_mach[-1])])

            # Recast lists as arrays
            x_mach = np.array(x_mach)
            y_cla = np.array(y_cla)

            interpf = interp1d(x_mach, y_cla, kind='cubic')
            a_transonic = interpf(mach_inf)

            # The slope is weighted either as pure subsonic, subsonic, pure transonic, supersonic, or pure supersonic
            if mach_inf < 1:
                weight_sub = np.interp(mach_inf,
                                       [puresubsonic_mach, lowertranson_mach],
                                       [1, 0])
                weight_sup = 0
            else:
                weight_sub = 0
                weight_sup = np.interp(mach_inf,
                                       [uppertranson_mach, puresupsonic_mach],
                                       [0, 1])

            liftslope_prad = 0
            if puresubsonic_mach < mach_inf < puresupsonic_mach:
                liftslope_prad += (1 - weight_sub - weight_sup) * a_transonic
            if mach_inf < lowertranson_mach:
                liftslope_prad += weight_sub * a_subsonic(machsub=mach_inf)
            elif mach_inf > uppertranson_mach:
                liftslope_prad += weight_sup * a_supersonic(machsuper=mach_inf)

        # To be implemented later: weighted averages for the subsonic and supersonic regimes
        # Lift slope values: Straight Wing (Highest a values) > Delta Wing > Swept Wing (Lowest a values)
        # Lift slope values: High aspect ratio (Highest a values) > Low aspect ratio (Lowest a values)

        return liftslope_prad

    def induceddragfact_lesm(self, wingloading_pa=None, cl_real=None,
                             mach_inf=None):
        """Lift induced drag factor k estimate (Cd = Cd0 + k.Cl^2), from LE suction theory, for aircraft
        capable of supersonic flight.

        **Parameters:**

        wingloading_pa
            float or array, list of wing-loading values in Pa. Optional, provided that an
            aircraft weight and wing area are specified in the design definitions dictionary.

        cl_real
            float or array, the coefficient of lift demanded to perform a maneuver. Optional,
            defaults to cl at cruise.

        mach_inf
            float, Mach number at which the Oswald efficiency factor is to be estimated, required
            to evaluate compressibility effects. Optional, defaults to 0.3 (incompressible flow).

        **Outputs:**

        k
            float, predicted lift-induced drag factor K, as used in (Cd = Cd0 + k.Cl^2)

        **Note**

        This method does not contain provisions for 'wing-in-ground-effect' factors.

        """

        if mach_inf is None:
            argmsg = 'Mach number unspecified, defaulting to incompressible flow condition.'
            warnings.warn(argmsg, RuntimeWarning)
            mach_inf = 0.3

        if cl_real is None:
            argmsg = 'Coefficient of lift attained unspecified, defaulting to cruise lift coefficient.'
            warnings.warn(argmsg, RuntimeWarning)

        if wingloading_pa is None:
            if self.weight_n is False:
                designmsg = 'Maximmum take-off weight not specified in the design definitions dictionary.'
                raise ValueError(designmsg)
            if self.wingarea_m2 is False:
                designmsg = 'Wing area not specified in the design definitions dictionary.'
                raise ValueError(designmsg)
            wingloading_pa = self.weight_n / self.wingarea_m2
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        if (self.cruisespeed_ktas is False) or (self.cruisealt_m is False):
            cruisemsg = 'Cruise Cl could not be determined (missing cruise speed/altitude' \
                        ' in the designbrief dictionary), defaulting to 0.6.'
            warnings.warn(cruisemsg, RuntimeWarning)
            cl_cruise = 0.6
        else:
            cruisespeed_mps = co.kts2mps(self.cruisespeed_ktas)
            qcruise_pa = self.designatm.dynamicpressure_pa(cruisespeed_mps,
                                                           self.cruisealt_m)
            cl_cruise = wingloading_pa * self.cruise_weight_fraction / qcruise_pa

        aspectr = self.aspectratio
        sweep_le_rad = self.sweep_le_rad
        machstar_le = 1.0 / math.cos(
            sweep_le_rad)  # Free-stream mach number required for sonic LE condition

        # Estimate subsonic k with the oswald factor
        k_oswald = self.induceddragfact(whichoswald=24,
                                        mach_inf=min(mach_inf, 0.6))

        # Estimate full regime k from Leading-Edge-Suction method (Aircraft Design, Daniel P. Raymer)

        # Zero-suction case
        k_0 = 1 / self.liftslope_prad(mach_inf=mach_inf)

        # Non-zero-suction case (This function for k_100 produces a messy curve, this needs smoothing somehow)
        if mach_inf < 1:  # Aircraft free-stream mach number is subsonic

            k_100 = 1.0 / (math.pi * aspectr)  # Full-suction case, oswald e = 1

        elif mach_inf < machstar_le:  # Free-stream is (super)sonic, but wing leading edge sees subsonic flow

            # Boundary conditions
            x1, x2 = 1.0, machstar_le
            y1, y2 = 1.0 / (math.pi * aspectr), 1 / self.liftslope_prad(
                mach_inf=machstar_le)
            m1 = 0

            # Solve simultaneous equations
            mat_a = np.array([[x1 ** 2, x1, 1],
                              [x2 ** 2, x2, 1],
                              [2 * x1, 1, 0]])
            mat_b = np.array([y1, y2, m1])
            mat_x = np.linalg.solve(mat_a, mat_b)

            # The polynomial describing the suction case between sonic freestream and sonic leading edge mach
            k_100 = 0
            order = 2
            for index in range(order + 1):
                k_100 += (mat_x[index] * mach_inf ** (order - index))

        else:  # Aircraft is fully supersonic

            k_100 = k_0  # Suction can not take place, therefore k_100 = k_0

        # Find the leading edge suction factor S(from model of Raymer data, assuming max suction ~93 %)

        # Suction model
        def y_suction(cl_delta, cl_crs, a, c, r):
            k = (-0.5 * cl_crs ** 2) - (0.25 * cl_crs) - 0.22
            b = 1 + r * k
            x = cl_delta
            y = a * (x - b) * np.exp(-c * (x - 0.1)) * -np.tan(0.1 * (x - k))
            return y

        if (cl_real is None) or (cl_cruise is None):
            cl_diff = 0
        else:
            cl_diff = cl_real - cl_cruise

        # Suction model for design Cl=0.3 and Cl=0.8
        y_03 = y_suction(cl_delta=cl_diff, cl_crs=0.3, a=22.5, c=1.95, r=0)
        y_08 = y_suction(cl_delta=cl_diff, cl_crs=0.8, a=5.77, c=1, r=-1.29)

        # Find suction at actual cl as a weight of the two sample curves
        weight = np.interp(cl_cruise, [0.3, 0.8], [1, 0])
        suctionfactor = weight * y_03 + (1 - weight) * y_08

        k_suction = suctionfactor * k_100 + (1 - suctionfactor) * k_0

        # Take k to be the weighted average between a subsonic oswald, and supersonic suction prediction
        weight = np.interp(mach_inf, [0, machstar_le], [1, 0]) ** 0.2

        k_predicted = (k_oswald * weight + k_suction * (1 - weight))

        return k_predicted

    def bestclimbspeedprop(self, wingloading_pa, altitude_m):
        """The best rate of climb speed for a propeller aircraft"""

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        dragfactor = np.sqrt(self.induceddragfact() / (3 * self.cdminclean))
        densfactor = 2 / self.designatm.airdens_kgpm3(altitude_m)

        # Gudmundsson, eq. (18-27)
        bestspeed_mps = np.sqrt(densfactor * wingloading_pa * dragfactor)

        if len(bestspeed_mps) == 1:
            return bestspeed_mps[0]

        return bestspeed_mps

    def constrain_climb(self, wingloading_pa):
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        climbalt_m = getattr(self.brief, "climbalt_m")
        climbspeed_kias = getattr(self.brief, "climbspeed_kias")
        climbrate_fpm = getattr(self.brief, "climbrate_fpm")
        CDminclean = getattr(self.performance, "CDminclean")
        CLmaxclean = getattr(self.performance, "CLmaxclean")

        # Determine the thrust and power lapse corrections
        climbspeed_mpsias = co.kts2mps(climbspeed_kias)
        climbspeed_mpstas = self.designatm.eas2tas(
            eas=climbspeed_mpsias,
            altitude_m=climbalt_m
        )
        mach = climbspeed_mpstas / self.designatm.vsound_mps(climbalt_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude=climbalt_m, norm=True,
            eta_prop=getattr(self.performance, "etaprop")["climb"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=climbalt_m, norm=True
            ) * getattr(self.performance, "etaprop")["climb"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = getattr(self.design, "weightfractions")["climb"]

        # Compute climb constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=climbspeed_mpstas, altitude_m=climbalt_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = CLmaxclean * q_pa
        if (ws_pa > CLmaxclean * q_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... induced drag factor
        cl = ws_pa / q_pa
        k = self.induceddragfact_lesm(
            wingloading_pa=wingloading_pa, cl_real=cl, mach_inf=mach)

        # ... rate of climb penalty
        climbrate_mps = co.fpm2mps(climbrate_fpm)
        climbrate_mpstroc = self.designatm.eas2tas(
            eas=climbrate_mps, altitude_m=climbalt_m)
        cos_sq_theta = (1 - (climbrate_mpstroc / climbspeed_mpstas) ** 2)

        # ... "acceleration factor"
        Ka = 1.0

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CDminclean / ws_pa
                     + k / q_pa * ws_pa * cos_sq_theta
                     + Ka * climbrate_mpstroc / climbspeed_mpstas
             ) / tcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * climbspeed_mpstas / pcorr

        return tw, pw

    def constrain_cruise(self, wingloading_pa):
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        cruisealt_m = getattr(self.brief, "cruisealt_m")
        cruisespeed_ktas = getattr(self.brief, "cruisespeed_ktas")
        cruisethrustfact = getattr(self.brief, "cruisethrustfact")
        CDminclean = getattr(self.performance, "CDminclean")
        CLmaxclean = getattr(self.performance, "CLmaxclean")

        # Determine the thrust and power lapse corrections
        cruisespeed_mpstas = co.kts2mps(cruisespeed_ktas)
        mach = cruisespeed_mpstas / self.designatm.vsound_mps(cruisealt_m)
        tcorr = cruisethrustfact * self.propulsion.thrust(
            mach=mach, altitude=cruisealt_m, norm=True,
            eta_prop=getattr(self.performance, "etaprop")["cruise"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=cruisealt_m, norm=True
            ) * getattr(self.performance, "etaprop")["cruise"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = getattr(self.design, "weightfractions")["cruise"]

        # Compute cruise constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=cruisespeed_mpstas, altitude_m=cruisealt_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = CLmaxclean * q_pa
        if (ws_pa > CLmaxclean * q_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... induced drag factor
        cl = ws_pa / q_pa
        k = self.induceddragfact_lesm(
            wingloading_pa=wingloading_pa, cl_real=cl, mach_inf=mach)

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CDminclean / ws_pa
                     + k / q_pa * ws_pa
             ) / tcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * cruisespeed_mpstas / pcorr

        return tw, pw

    def constrain_servceil(self, wingloading_pa):
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        servceil_m = getattr(self.brief, "servceil_m")
        secclimbspd_kias = getattr(self.brief, "secclimbspd_kias")
        CDminclean = getattr(self.performance, "CDminclean")
        CLmaxclean = getattr(self.performance, "CLmaxclean")

        # Determine the thrust and power lapse corrections
        secclimbspd_mpsias = co.kts2mps(secclimbspd_kias)
        secclimbspd_mpstas = self.designatm.eas2tas(
            eas=secclimbspd_mpsias,
            altitude_m=servceil_m
        )
        mach = secclimbspd_mpstas / self.designatm.vsound_mps(servceil_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude=servceil_m, norm=True,
            eta_prop=getattr(self.performance, "etaprop")["servceil"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=servceil_m, norm=True
            ) * getattr(self.performance, "etaprop")["servceil"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = getattr(self.design, "weightfractions")["servceil"]

        # Compute climb constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=secclimbspd_mpstas, altitude_m=servceil_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = CLmaxclean * q_pa
        if (ws_pa > CLmaxclean * q_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... induced drag factor
        cl = ws_pa / q_pa
        k = self.induceddragfact_lesm(
            wingloading_pa=wingloading_pa, cl_real=cl, mach_inf=mach)

        # Service ceiling typically defined in terms of climb rate (at best
        # climb speed) dropping to 100feet/min ~ 0.508m/s
        climbrate_mps = co.fpm2mps(100)
        # What true climb rate does 100 feet/minute correspond to?
        climbrate_mpstroc = self.designatm.eas2tas(
            eas=climbrate_mps, altitude_m=servceil_m)

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CDminclean / ws_pa
                     + k / q_pa * ws_pa
                     + climbrate_mpstroc / secclimbspd_mpstas
             ) / tcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * secclimbspd_mpstas / pcorr

        return tw, pw

    def constrain_takeoff(self, wingloading_pa):
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
            eta_prop=getattr(self.performance, "etaprop")["take-off"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=machbar, altitude=rwyelevation_m, norm=True
            ) * getattr(self.performance, "etaprop")["take-off"]
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
        # Recast as necessary
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        stloadfactor = getattr(self.brief, "stloadfactor")
        turnalt_m = getattr(self.brief, "turnalt_m")
        turnspeed_ktas = getattr(self.brief, "turnspeed_ktas")
        CDminclean = getattr(self.performance, "CDminclean")
        CLmaxclean = getattr(self.performance, "CLmaxclean")

        # Determine the thrust and power lapse corrections
        turnspeed_mpstas = co.kts2mps(turnspeed_ktas)
        mach = turnspeed_mpstas / self.designatm.vsound_mps(turnalt_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude=turnalt_m, norm=True,
            eta_prop=getattr(self.performance, "etaprop")["turn"]
        )
        if self.propulsion.type in ["piston", "turboprop"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude=turnalt_m, norm=True
            ) * getattr(self.performance, "etaprop")["turn"]
        else:
            pcorr = np.nan

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = getattr(self.design, "weightfractions")["turn"]

        # Compute climb constraint
        # ... (upper bounded) wing loading
        q_pa = self.designatm.dynamicpressure_pa(
            airspeed_mps=turnspeed_mpstas, altitude_m=turnalt_m)
        ws_pa = wingloading_pa * wcorr
        wslim_pa = CLmaxclean * q_pa
        if (ws_pa > CLmaxclean * q_pa).any():
            warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            warnings.warn(warnmsg, RuntimeWarning)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... induced drag factor
        cl = ws_pa / q_pa
        k = self.induceddragfact_lesm(
            wingloading_pa=wingloading_pa, cl_real=cl, mach_inf=mach)

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CDminclean / ws_pa
                     + k / q_pa * ws_pa * stloadfactor ** 2
             ) / tcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * turnspeed_mpstas / pcorr

        return tw, pw

    @property
    def cleanstall_wsmax(self):
        # (W/S)_max = q_vstall * CLmaxclean
        # Recast as necessary
        vstallclean_kcas = getattr(self.brief, "vstallclean_kcas")
        CLmaxclean = getattr(self.performance, "CLmaxclean")
        if vstallclean_kcas is None or CLmaxclean is None:
            return np.nan

        # We do the q calculation at SL conditions, TAS ~= EAS ~= CAS
        # (on the basis that the stall Mach number is likely very small)
        stallspeed_mpstas = co.kts2mps(vstallclean_kcas)

        # Dynamic pressure at sea level
        q_pa = self.designatm.dynamicpressure_pa(stallspeed_mpstas, 0.0)
        ws_pa = q_pa * CLmaxclean
        return ws_pa

    @property
    def cleanstall_smin(self):
        # (S)_min = W / (W/S)_max
        # Recast as necessary
        weight_n = getattr(self.design, "weight_n")
        s_m2 = weight_n / self.cleanstall_wsmax
        return s_m2

    def plot_constraints(self, wingloading_pa):
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
                tws[label], pws[label] = function(wingloading_pa)
            except TypeError as _:
                tws[label] = np.nan * wingloading_pa
                pws[label] = np.nan * wingloading_pa

        # Choose between S and W/S, depending on if the MTOW is known
        weight_n = getattr(self.design, "weight_n")
        if weight_n is None:
            xs = wingloading_pa
            xstall = self.cleanstall_wsmax
            xlabel = "Wing loading [Pa]"
        else:
            xs = weight_n / wingloading_pa
            xstall = weight_n / self.cleanstall_wsmax
            xlabel = "Wing area [m$^2$]"

        # Choose between T/W and P/W, depending on propulsion system
        if self.propulsion.type in ["piston", "turboprop"]:
            if weight_n is None:
                ys = pws
                ylabel = "Power-to-Weight"
            else:
                ys = {k: v * weight_n for (k, v) in pws.items()}
                ylabel = "Power [W]"
        else:
            if weight_n is None:
                ys = tws
                ylabel = "Thrust-to-Weight"
            else:
                ys = {k: v * weight_n for (k, v) in tws.items()}
                ylabel = "Thrust [N]"

        fig, ax = plt.subplots(dpi=140, figsize=(6, 4))
        fig.subplots_adjust(right=0.7)

        # Plot the main constraints
        cmap = plt.get_cmap("jet")
        ncolours = len(constraint_fs) + 1
        colours = cmap(np.arange(ncolours) / ncolours)
        for i, (label, values) in enumerate(ys.items()):
            clr = colours[i]
            ax.plot(xs, values, label=label, c=clr, zorder=10, lw=2)
            ax.fill_between(xs, values, fc=clr, alpha=0.2, zorder=5)
        else:
            # Plot the clean stall
            ylim = np.nanmedian(list(ys.values()), axis=1).max() / 0.45
            yx1x2 = [0, ylim], xstall, xs.min() if weight_n else xs.max()
            clr = colours[len(constraint_fs)]
            l2d = ax.axvline(xstall, label="clean stall", c=clr, zorder=10)
            ax.fill_betweenx(*yx1x2, fc=l2d.get_color(), alpha=0.2, zorder=5)

            # Plot the stall limit due to all constraints
            nanindex = np.isnan(np.array(list(ys.values()))).sum(axis=1).max()
            yx1x2 = [0, ylim], xs[-nanindex], xs.min() if weight_n else xs.max()
            if nanindex > 0:
                kkka = (0, 0, 0, 0.2)
                ax.fill_betweenx(
                    *yx1x2, fc=kkka, ec="k", zorder=15, hatch="x",
                    label=r"above $C_{L,max}$"
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(0, ylim)
        ax.legend(bbox_to_anchor=(1.05, 0.9), loc="upper left")
        ax.grid()

        return fig, ax

    def propulsionsensitivity_monothetic(self, wingloading_pa, y_var='tw',
                                         y_lim=None, x_var='ws_pa',
                                         customlabels=None,
                                         show=True, maskbool=False,
                                         textsize=None, figsize_in=None):
        """Constraint analysis in the wing loading (or wing area) - T/W ratio (or power) space.
        The method generates a plot of the combined constraint diagram, with optional sensitivity
        diagrams for individual constraints. These are based on a One-Factor-at-a-Time analysis
        of the local sensitivities of the constraints (required T/W or power values) with respect
        to the variables that define the aircraft concept. The sensitivity charts show the
        relative proportions of these local sensitivities, referred to as 'relative sensitivities'.
        Sensitivities are computed for those inputs that are specified as a range (a [min, max] list)
        instead of a single scalar value and the sensitivity is estimated across this range, with the
        midpoint taken as the nominal value (see more details in `this notebook <https://github.com/sobester/ADRpy/blob/master/docs/ADRpy/notebooks/Constraint%20analysis%20of%20a%20single%20engine%20piston%20prop.ipynb>`_).

        Sensitivities can be computed with respect to components of the design brief, as well as
        aerodynamic parameter estimates or geometrical parameters.

        The example below can serve as a template for setting up a sensitivity study; further
        examples can be found in the notebook.

        This is a higher level wrapper of :code:`twrequired` - please consult its documentation
        entry for details on the individual constraints and their required inputs.

        **Parameters:**

        wingloading_pa
            array, list of wing loading values in Pa.

        y_var
            string, specifies the quantity to be plotted along the y-axis of the combined
            constraint diagram. Set to 'tw' for dimensionless thrust-to-weight required, or 'p_hp'
            for the power required (in horsepower); sea level standard day values in both cases.
            Optional, defaults to 'tw'.

        y_lim
            float, used to define the plot y-limit. Optional, defaults to 105% of the maximum
            value across all constraint curves.

        x_var
            string, specifies the quantity to be plotted along the x-axis of the combined
            constraint diagram. Set to 'ws_pa' for wing loading in Pa, or 's_m2' for wing area
            in metres squared. Optional, defaults to 'ws_pa'.

        customlabels
            dictionary, used to remap design definition parameter keys to labels better suited
            for plot labelling. Optional, defaults to None. See example below for usage.

        show
            boolean/string, used to indicate the type of plot required. Available arguments:
            :code:`True`, :code:`False`, 'combined', 'climb', 'cruise', 'servceil',
            'take-off', and 'turn'. Optional, defaults to True. 'combined' will generate
            the classic combined constraint diagram on its own.

        maskbool
            boolean, used to indicate whether or not constraints that do not affect the
            combined minimum propulsion sizing requirement should be obscured. Optional,
            defaults to False.

        textsize
            integer, sets a representative reference fontsize for the text on the plots.
            Optional, defaults to 10 for multi-subplot figures, and to 14 for singles.

        figsize_in
            list, used to specify custom dimensions of the output plot in inches. Image width
            must be specified as a float in the first entry of a two-item list, with height as
            the second item. Optional, defaults to 14.1 inches wide by 10 inches tall.

        **See also** ``twrequired``

        **Notes**

        1. This is a plotting routine that wraps the various constraint models implemented in ADRpy.
        If specific constraint data is required, use :code:`twrequired`.

        2. Investigating sensitivities of design parameters embedded within the aircraft concept
        definition dictionaries, such as weight fractions or propeller efficiencies for various
        constraints, is not currently supported. Similarly, it uses the atmosphere provided in
        the class argument 'designatm'; the computation of sensitivities with respect to
        atmosphere choice is not supported.

        3. The sensitivities are computed numerically.

        **Example** ::

            import numpy as np

            from ADRpy import atmospheres as at
            from ADRpy import constraintanalysis as ca

            designbrief = {'rwyelevation_m': 0, 'groundrun_m': 313,
                            'stloadfactor': [1.5, 1.65], 'turnalt_m': [1000, 1075], 'turnspeed_ktas': [100, 110],
                            'climbalt_m': 0, 'climbspeed_kias': 101, 'climbrate_fpm': 1398,
                            'cruisealt_m': [2900, 3200], 'cruisespeed_ktas': [170, 175], 'cruisethrustfact': 1.0,
                            'servceil_m': [6500, 6650], 'secclimbspd_kias': 92,
                            'vstallclean_kcas': 69}
            designdefinition = {'aspectratio': [10, 11], 'sweep_le_deg': 2, 'sweep_25_deg': 0, 'bpr': -1,
                                'wingarea_m2': 13.46, 'weight_n': 15000,
                                'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 0.853, 'servceil': 1.0}}
            designperformance = {'CDTO': 0.0414, 'CLTO': 0.59, 'CLmaxTO': 1.69, 'CLmaxclean': 1.45, 'mu_R': 0.02,
                                 'CDminclean': [0.0254, 0.026], 'etaprop': {'take-off': 0.65, 'climb': 0.8,
                                                                            'cruise': 0.85, 'turn': 0.85,
                                                                            'servceil': 0.8}}

            wingloadinglist_pa = np.arange(700, 2500, 5)
            customlabelling = {'aspectratio': 'AR',
                               'sweep_le_deg': '$\\Lambda_{LE}$',
                               'sweep_mt_deg': '$\\Lambda_{MT}$'}

            atm = at.Atmosphere()
            concept = ca.AircraftConcept(designbrief, designdefinition, designperformance, atm)

            concept.propulsionsensitivity_monothetic(wingloading_pa=wingloadinglist_pa, y_var='p_hp', x_var='s_m2',
                                                 customlabels=customlabelling)

        """

        y_types_list = ['tw', 'p_hp']
        if y_var not in y_types_list:
            argmsg = 'Unsupported y-axis variable specified "{0}", using default "tw".'.format(
                str(y_var))
            warnings.warn(argmsg, RuntimeWarning)
            y_var = 'tw'

        if y_lim:
            if (type(y_lim) == float) or (type(y_lim) == int):
                pass
            else:
                argmsg = 'Unsupported plot y-limit specified "{0}", using default.'.format(
                    str(y_lim))
                warnings.warn(argmsg, RuntimeWarning)
                y_lim = None

        x_types_list = ['ws_pa', 's_m2']
        if x_var not in x_types_list:
            argmsg = 'Unsupported x-axis variable specified "{0}", using default "ws_pa".'.format(
                str(x_var))
            warnings.warn(argmsg, RuntimeWarning)
            x_var = 'ws_pa'

        if customlabels is None:
            # There is no need to throw a warning if the following method arguments are left unspecified.
            customlabels = {}

        if textsize is None:
            if show is True:
                textsize = 10
            else:
                textsize = 14

        default_figsize_in = [14.1, 10]
        if figsize_in is None:
            figsize_in = default_figsize_in
        elif type(figsize_in) == list:
            if len(figsize_in) != 2:
                argmsg = 'Unsupported figure size, should be length 2, found {0} instead - using default parameters.' \
                    .format(len(figsize_in))
                warnings.warn(argmsg, RuntimeWarning)
                figsize_in = default_figsize_in

        if self.weight_n is False:
            defmsg = 'Maximum take-off weight was not specified in the aircraft design definitions dictionary.'
            raise ValueError(defmsg)

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        # Colour/alpha dictionary
        style = {
            'focusmask': {'colour': 'white', 'alpha': 0.70},
            'inv_soln': {'colour': 'crimson', 'alpha': 0.10}
        }
        # Pick a colour (red)
        # Step clockwise on the colour wheel and go darker (darkviolet)
        # Use the complementary colour and go lighter (yellowgreen)
        # Step clockwise on the colour wheel and go darker (olive)
        # Use the complementary colour and go lighter (mediumslateblue)
        # etc... until you end up at the start colour. Bright colours on even index, dark colours on odd
        clr_list = ['limegreen', 'olivedrab', 'darkorchid', 'indigo', 'yellow',
                    'darkgoldenrod',
                    'royalblue', 'darkslategrey', 'orange', 'sienna',
                    'darkturquoise', 'forestgreen',
                    'red', 'darkviolet', 'yellowgreen', 'olive',
                    'mediumslateblue', 'navy',
                    'gold', 'chocolate', 'dodgerblue', 'teal', 'lightcoral',
                    'darkred']
        clr_dict = {}

        # Potential design space and nominal design state, design dictionaries stored in lists
        designspace_list = self.designspace
        designstate_list = self.designstate
        designatmosphere = self.designatm
        mass_kg = self.weight_n / constants.g

        # If a design can take a range of values, its value is bounded by the max and min value items of the list
        sensitivityplots_list = ['climb', 'cruise', 'servceil', 'take-off',
                                 'turn']
        propulsionreqprime = dict(zip(sensitivityplots_list, [{} for _ in range(
            len(sensitivityplots_list))]))

        # It's probably not necessary to have these all as functions since they are static - needs future optimisation
        def y_function(aircraft_object, y_type):
            if y_type == 'p_hp':  # If horsepower is to be plotted on the y-axis
                propulsionrequirement = aircraft_object.powerrequired(
                    wingloading_pa=wingloading_pa, tow_kg=mass_kg)
            else:  # else default to T/W plotting on the y-axis
                propulsionrequirement = aircraft_object.twrequired(
                    wingloading_pa=wingloading_pa)
            return propulsionrequirement

        def x_function(x_type):
            if x_type == 's_m2':  # If wing area is to be plotted on the x-axis
                plt_x_axis = self.weight_n / wingloading_pa
            else:  # else default to W/S plotting on the x-axis
                plt_x_axis = wingloading_pa
            return plt_x_axis

        def y_labelling(y_type):
            if y_type == 'p_hp':  # Horsepower is to be plotted on the y-axis
                ylabel = 'Power required [hp]'
            else:  # Else default to T/W plotting on the y-axis
                ylabel = 'Thrust to weight ratio [-]'
            return ylabel

        def x_labelling(x_type):
            if x_type == 's_m2':  # If wing-area is to be plotted on the x-axis
                xlabel = 'Wing area [m$^2$]'
            else:  # Else default to W/S plotting on the x-axis
                xlabel = 'Wing loading [Pa]'
            return xlabel

        def wherecleanstall(x_type):
            if x_type == 's_m2':  # If wing-area is to be plotted on the x-axis
                x_stall = self.smincleanstall_m2(mass_kg)
            else:  # Else default to W/S plotting on the x-axis
                x_stall = self.wsmaxcleanstall_pa()
            return x_stall

        # Perform OFAT monothetic analysis
        for dictionary_i in range(len(designspace_list)):

            for _, (dp_k, dp_v) in enumerate(
                    designspace_list[dictionary_i].items()):

                # If a list was found, create two temporary dictionaries with the maximum and minimum bounded values
                if type(dp_v) == list:

                    # Create copies of the nominal design state, amending the key of interest with the list extremes
                    temp_designstatemax = deepcopy(designstate_list)
                    temp_designstatemax[dictionary_i][dp_k] = max(dp_v)

                    temp_designstatemin = deepcopy(designstate_list)
                    temp_designstatemin[dictionary_i][dp_k] = min(dp_v)

                    # Evaluate the dictionaries as aircraft concepts
                    briefmax, briefmin = temp_designstatemax[0], \
                        temp_designstatemin[0]
                    designdefmax, designdefmin = temp_designstatemax[1], \
                        temp_designstatemin[1]
                    performancemax, performancemin = temp_designstatemax[2], \
                        temp_designstatemin[2]

                    # Evaluate T/W or P
                    acmax = AircraftConcept(briefmax, designdefmax,
                                            performancemax, designatmosphere,
                                            self.propulsion)
                    acmin = AircraftConcept(briefmin, designdefmin,
                                            performancemin, designatmosphere,
                                            self.propulsion)
                    propulsionreqmax = y_function(aircraft_object=acmax,
                                                  y_type=y_var)
                    propulsionreqmin = y_function(aircraft_object=acmin,
                                                  y_type=y_var)

                    # If after evaluating OFAT there was a change in T/W for a constraint, record magnitude of the range
                    for constraint in sensitivityplots_list:
                        propulsionreq_range = abs(
                            propulsionreqmax[constraint] - propulsionreqmin[
                                constraint])
                        # If the range is non-zero at any point, then the OFAT parameter had an impact on the constraint
                        if propulsionreq_range.all() != np.zeros(
                                len(propulsionreq_range)).all():
                            propulsionreqprime[constraint].update(
                                {dp_k: propulsionreq_range})
                            # If parameter impacting the constraint was not previously recorded, assign a unique colour
                            if dp_k not in clr_dict:
                                clr_dict.update({dp_k: clr_list[len(clr_dict)]})

        # Produce data for the combined constraint
        brief, designdef, performance = designstate_list[0], designstate_list[
            1], designstate_list[2]
        acmed = AircraftConcept(brief, designdef, performance, designatmosphere,
                                self.propulsion)
        propulsionreqmed = y_function(aircraft_object=acmed, y_type=y_var)
        # Refactor the x-axis
        x_axis = x_function(x_type=x_var)

        # If a stall constraint exists, create plot data
        if self.vstallclean_kcas and self.clmaxclean:  # If the stall condition is available to plot
            xcrit_stall = wherecleanstall(x_type=x_var)
        else:
            xcrit_stall = None

        # Determine the upper y-limit of the plots
        if y_lim is None:  # If the user did not specify a y_limit using the y_lim argument
            ylim_hi = []
            for sensitivityplot in sensitivityplots_list:
                ylim_hi.append(max(propulsionreqmed[sensitivityplot]))
            ylim_hi = max(ylim_hi) * 1.05
        else:
            ylim_hi = y_lim

        # Find the indices on the x-axis, where the propulsion constraint is feasible (sustained turn does not stall)
        propfeasindex = np.where(np.isfinite(propulsionreqmed['combined']))[0]
        # (Set of x-axis indices) - (set of feasible x-axis indices) = (set of infeasible x-axis indices)
        infeas_x_axis = list(set(x_axis) - set(x_axis[propfeasindex[0: -1]]))
        # x-values that bound the feasible propulsion region
        x2_infeascl = max(infeas_x_axis)
        x1_infeascl = min(infeas_x_axis)

        # GRAPH PLOTTING

        predefinedlabels = {'climb': "Climb", 'cruise': "Cruise",
                            'servceil': "Service ceiling",
                            'take-off': "Take-off ground roll",
                            'turn': "Sustained turn"}

        fontsize_title = 1.20 * textsize
        fontsize_label = 1.05 * textsize
        fontsize_legnd = 1.00 * textsize
        fontsize_tick = 0.90 * textsize

        def sensitivityplots(whichconstraint, ax_sens=None):
            if ax_sens is None:
                ax_sens = plt.gca()

            # Find the sum of all derivatives for a constraint, at every given wing-loading
            primesum = np.zeros(len(wingloading_pa))
            for _, (param_k, param_v) in enumerate(
                    propulsionreqprime[whichconstraint].items()):
                primesum += param_v

            # Find the proportions of unity each parameter contributes to a constraint, and arrange as a list of arrays
            stackplot = []
            parameters_list = []
            keyclrs_list = []
            for _, (param_k, param_v) in enumerate(
                    propulsionreqprime[whichconstraint].items()):
                stackplot.append(param_v / primesum)
                # Use custom labels if they exist
                if param_k in customlabels:
                    parameters_list.append(customlabels[param_k])
                else:
                    parameters_list.append(param_k)
                # Assign a key in the stackplot, its unique colour
                keyclrs_list.append(clr_dict[param_k])

            if len(stackplot) == 0:  # If no parameters could be added, populate stackplot with an empty filler
                stackplot.append([0.] * len(wingloading_pa))
                parameters_list.append("N/A")
                # Also draw a nice red 'x' to clearly identify the graph
                ax_sens.plot([min(x_axis), max(x_axis)], [0, 1], ls='-',
                             color='r')
                ax_sens.plot([min(x_axis), max(x_axis)], [1, 0], ls='-',
                             color='r')
                # The colour list should also be populated with a dummy colour
                keyclrs_list.append('red')
            else:
                pass

            # For the constraint being processed, generate stacked plot from the list of arrays
            ax_sens.stackplot(x_function(x_type=x_var), stackplot,
                              labels=parameters_list, colors=keyclrs_list)
            ax_sens.set_title(predefinedlabels[whichconstraint],
                              size=fontsize_title)
            ax_sens.set_xlim(min(x_axis), max(x_axis))
            ax_sens.set_ylim(0, 1)
            ax_sens.set_xlabel(xlabel=x_labelling(x_type=x_var),
                               fontsize=fontsize_label)
            ax_sens.set_ylabel(
                ylabel=('Rel. sensitivity of ' + y_var.split('_')[0].upper()),
                fontsize=fontsize_label)
            ax_sens.tick_params(axis='x', labelsize=fontsize_tick)
            ax_sens.tick_params(axis='y', labelsize=fontsize_tick)
            # The legend list must be reversed to make sure the legend displays in the same order the plot is stacked
            handles, labels = ax_sens.get_legend_handles_labels()
            ax_sens.legend(reversed(handles), reversed(labels),
                           title='Varying factors', loc='center left',
                           bbox_to_anchor=(1, 0.5),
                           prop={'size': fontsize_legnd},
                           title_fontsize=fontsize_legnd)

            if maskbool:
                # For sensitivity plots, focus user attention with transparent masks
                propreqcomb = np.nan_to_num(propulsionreqmed['combined'],
                                            copy=True)
                maskindex = \
                    np.where(propulsionreqmed[whichconstraint] < propreqcomb)[
                        0]  # Mask where constraint < comb

                # Find discontinuities in maskindex, since we want to mask wherever maskindex is counting consecutively
                masksindex_list = np.split(maskindex,
                                           np.where(np.diff(maskindex) != 1)[
                                               0] + 1)

                for consecregion_index in range(len(masksindex_list)):
                    x2_clmask = x_axis[max(masksindex_list[consecregion_index])]
                    x1_clmask = x_axis[min(masksindex_list[consecregion_index])]

                    # If the cl mask is at the max feasible index, then draw from the min index to the max of the x-axis
                    if x2_clmask == x_axis[max(propfeasindex)]:
                        x2_clmask = x_axis[-1]

                    ax_sens.fill([x1_clmask, x2_clmask, x2_clmask, x1_clmask],
                                 [0, 0, 1, 1],
                                 color=style['focusmask']['colour'],
                                 alpha=style['focusmask']['alpha'])

            return None

        def combinedplot(ax_comb=None):
            if ax_comb is None:
                ax_comb = plt.gca()

            ax_comb.plot(x_axis, propulsionreqmed['combined'], lw=3.5,
                         color='k',
                         label="Combined front \nup to turn stall line")
            # Aggregate the propulsion constraints onto the combined diagram
            for item in sensitivityplots_list:
                ax_comb.plot(x_axis, propulsionreqmed[item],
                             label=predefinedlabels[item], lw=2.0, ls='--',
                             color=clr_list[
                                 sensitivityplots_list.index(item) * 2 + 6])

            # If the code could figure out where the clean stall takes place, plot it
            if xcrit_stall:
                if min(x_axis) < xcrit_stall < max(x_axis):
                    ax_comb.plot([xcrit_stall, xcrit_stall], [0, ylim_hi],
                                 label="Clean stall, 1g")

            # If the code could figure out where the turn stall takes place, plot it
            if len(propfeasindex) > 0:
                xturn_stall = x_axis[propfeasindex[-1]]
                if min(x_axis) < xturn_stall < max(x_axis):
                    ax_comb.plot([xturn_stall, xturn_stall], [0, ylim_hi],
                                 label="Turn stall limit")

            # ax_comb.set_title('Aggregated Propulsion Constraints', size=fontsize_title)
            ax_comb.set_xlim(min(x_axis), max(x_axis))
            ax_comb.set_ylim(0, ylim_hi)
            ax_comb.set_xlabel(xlabel=x_labelling(x_type=x_var),
                               fontsize=fontsize_label)
            ax_comb.set_ylabel(ylabel=y_labelling(y_type=y_var),
                               fontsize=fontsize_label)
            ax_comb.tick_params(axis='x', labelsize=fontsize_tick)
            ax_comb.tick_params(axis='y', labelsize=fontsize_tick)
            ax_comb.legend(title='Constraints', loc='center left',
                           bbox_to_anchor=(1, 0.5),
                           prop={'size': fontsize_legnd},
                           title_fontsize=fontsize_legnd)
            ax_comb.grid(True)

            # For the combined plot, obscure region of unattainable performance due to infeasible CL requirements
            if xcrit_stall:  # If the stall constraint is given, the CL mask should account for turn/stall constraints
                x2_clmask = max(xcrit_stall, x2_infeascl)
                x1_clmask = min(xcrit_stall, x1_infeascl)

            else:  # Else the CL mask should only evaluate the turn constraint CL
                x2_clmask = x2_infeascl
                x1_clmask = x1_infeascl

            # If a non-zero-thickness area was found in the "combined" plot region for which cl is invalid, mask it
            if len(infeas_x_axis) >= 1:
                ax_comb.fill([x1_clmask, x2_clmask, x2_clmask, x1_clmask],
                             [0, 0, ylim_hi, ylim_hi],
                             color=style['inv_soln']['colour'],
                             alpha=style['inv_soln']['alpha'])

            # Produce coordinates that describe the lower bound of the feasible region
            solnfeasindex = np.append(np.where(x2_clmask < x_axis)[0],
                                      np.where(x1_clmask > x_axis)[0])
            # Obscure the remaining region in which the minimum combined constraint is not satisfied
            if len(solnfeasindex) < 1:
                solnfeasindex = [0]
            x_inv = np.append(x_axis[solnfeasindex], [x_axis[solnfeasindex][-1],
                                                      x_axis[solnfeasindex][0]])
            y_inv = np.append(propulsionreqmed['combined'][solnfeasindex],
                              [0, 0])
            ax_comb.fill(x_inv, y_inv, color=style['inv_soln']['colour'],
                         alpha=style['inv_soln']['alpha'])
            xfeas, yfeas = x_inv[0:-2], y_inv[
                                        0:-2]  # Map infeasible region coords, to a lower bound for feas T/W
            return xfeas, yfeas

        # Show the plot if specified to do so by method argument, then clear the plot and figure
        fig = False
        plots_list = ['climb', 'cruise', 'servceil', 'take-off', 'turn',
                      'combined']
        suptitle = {
            't': "OFAT Sensitivity of Propulsion System Constraints (" + y_labelling(
                y_type=y_var) + ")",
            'size': textsize * 1.4}

        if show is True:
            # Plotting setup, arrangement of 6 windows
            fig, axs = plt.subplots(3, 2, figsize=figsize_in,
                                    gridspec_kw={'hspace': 0.4, 'wspace': 0.8},
                                    sharex='all')
            # fig.canvas.set_window_title('ADRpy constraintanalysis.py')
            fig.subplots_adjust(left=0.1, bottom=None, right=0.82, top=None,
                                wspace=None, hspace=None)
            # fig.suptitle(suptitle['t'], size=suptitle['size'])

            axs_dict = dict(zip(plots_list,
                                [axs[0, 0], axs[1, 0], axs[2, 0], axs[0, 1],
                                 axs[1, 1], axs[2, 1]]))

            # Plot INDIVIDUAL constraint sensitivity diagrams
            for sensitivityplottype in sensitivityplots_list:
                sensitivityplots(whichconstraint=sensitivityplottype,
                                 ax_sens=axs_dict[sensitivityplottype])
            # Plot COMBINED constraint diagram
            combinedplot(ax_comb=axs_dict['combined'])

        elif show in plots_list:
            # Plotting setup, single window
            fig, ax = plt.subplots(1, 1, figsize=figsize_in,
                                   gridspec_kw={'hspace': 0.4, 'wspace': 0.8},
                                   sharex='all')
            fig.subplots_adjust(left=0.1, bottom=None, right=0.78, top=None,
                                wspace=None, hspace=None)

            if show in sensitivityplots_list:
                # Plot INDIVIDUAL constraint sensitivity diagram
                # fig.suptitle(suptitle['t'], size=suptitle['size'])
                sensitivityplots(whichconstraint=show, ax_sens=ax)
            else:
                # Plot COMBINED constraint diagram
                # fig.suptitle("Combined View of Propulsion System Requirements", size=suptitle['size'])
                combinedplot(ax_comb=ax)

        if show:
            plt.show()
            plt.close(fig=fig)

        return None

    def map2static(self):
        """Maps the average take-off thrust to static thrust. If a bypass ratio
        is not specified, it returns a value of 1.
        """
        if self.bpr > 1:
            return (4 / 3) * (4 + self.bpr) / (5 + self.bpr)

        return 1.0

    def wigfactor(self):
        """Wing-in-ground-effect factor to account for the change in
        induced drag as a result of the wing being in close proximity
        of the ground. Specify the entry `wingheightratio` in the
        `design` dictionary variable you instantiated the `AircraftConcept`
        object with in order to compute this - if unspecified, a call to this
        method will result in a value practically equal to 1 being returned.

        The factor, following McCormick ("Aerodynamics, Aeronautics, and Flight
        Mechanics", Wiley, 1979) and Gudmundsson (2013) is calculated as:

        .. math::

            \\Phi = \\frac{(16\\,h/b)^2}{1+(16\\,h/b)^2}

        where :math:`h/b` is `design['wingheightratio']`: the ratio of the height
        of the wing above the ground (when the aircraft is on the runway) and the
        span of the main wing.

        The induced drag coefficient adjusted for ground effect thus becomes:

        .. math::

            C_\\mathrm{Di} = \\Phi C_\\mathrm{Di}^\\mathrm{oge},

        where the 'oge' superscript denotes the 'out of ground effect' value.

        **Example**
        ::

            import math
            from ADRpy import constraintanalysis as co

            designdef = {'aspectratio':8}
            wingarea_m2 = 10
            wingspan_m = math.sqrt(designdef['aspectratio'] * wingarea_m2)

            for wingheight_m in [0.6, 0.8, 1.0]:

                designdef['wingheightratio'] = wingheight_m / wingspan_m

                aircraft = co.AircraftConcept({}, designdef, {}, {})

                print('h/b: ', designdef['wingheightratio'],
                      'Phi: ', aircraft.wigfactor())

        Output::

            h/b:  0.06708203932499368  Phi:  0.5353159851301115
            h/b:  0.08944271909999159  Phi:  0.6719160104986877
            h/b:  0.11180339887498948  Phi:  0.761904761904762

        """
        return _wig(self.wingheightratio)


def _wig(h_over_b):
    return ((16 * h_over_b) ** 2) / (1 + (16 * h_over_b) ** 2)
