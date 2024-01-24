"""
This module contains tools for the constraint analysis of fixed wing aircraft.
"""
from functools import wraps
import re
import typing
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants, optimize

from ADRpy import atmospheres as at
from ADRpy import unitconversions as co
from ADRpy.mtools4acdc import recastasnpfloatarray
from ADRpy import propulsion as pdecks

__all__ = ["make_modified_drag_model", "AircraftConcept"]
__author__ = "Yaseen Reza"


def raise_bad_method_error(func):
    """
    A wrapper for functions that take "method" as a keyword argument.
    The wrapper mines the function's docstring for what it thinks are valid
    methods. If the function raises an exception, this wrapper tries to
    identify it might have been because of a bad choice of method.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function.

    """
    # Attempt to extract valid methods from the method section of a docstring
    methodname_pattern = r"""['"][A-z-]+,?['"]"""
    methodsection = re.findall(
        r"method:.+(?:\s+" + methodname_pattern + ".+\n)+", func.__doc__)

    # If we couldn't find anything in the docstring that looks viable
    if not methodsection:
        raise ReferenceError("Couldn't find valid method definition(s)")
    else:
        methodsection, = methodsection

    allowed_methods = set(re.findall(r"""["']([A-z-]+),?["']""", methodsection))
    allowed_methods = allowed_methods | {None}

    @wraps(func)
    def with_method_checking(*args, **kwargs):
        """Tell the user if they've made a mistake when choosing a method."""
        try:
            result = func(*args, **kwargs)
        except Exception:
            # Check if the fail was because of a bad method choice
            if "method" in kwargs:
                method = kwargs["method"]

                # If user made invalid choice, tell user about allowed methods
                if method not in allowed_methods:
                    errormsg = (
                        f"{method=} isn't a valid choice for {func.__name__}. "
                        f"Please select any from {allowed_methods=}"
                    )
                    raise ValueError(errormsg)
            # Otherwise, error couldn't be handled. Do default raise behaviour
            raise
        return result

    return with_method_checking


def revert2scalar(func):
    @wraps(func)
    def with_reverting(*args, **kwargs):
        """Try to turn x into a scalar (if it is an array, list, or tuple)."""

        # Evaluate the wrapped func
        output = func(*args, **kwargs)

        if not isinstance(output, tuple):
            output = (output,)

        # Convert all items in the output to scalar if possible
        new_output = []
        for x in output:
            if isinstance(x, np.ndarray):
                if x.ndim == 0:
                    new_output.append(x.item())
                    continue
                if sum(x.shape) == 1:
                    new_output.append(x[0])
                    continue
            elif isinstance(x, (list, tuple)):
                if len(x) == 1:
                    new_output.append(x[0])
                    continue
            new_output.append(x)

        # If there was only one output from the function, return that as scalar
        if len(new_output) == 1:
            return new_output[0]

        return tuple(new_output)

    return with_reverting


def make_modified_drag_model(CDmin, k, CLmax, CLminD) -> typing.Callable:
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

        # Estimate a CDstall of about 180% of the quadratic model
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

        @revert2scalar
        def drag_model(CL):
            """
            A modified, adjusted drag model.

            Args:
                CL: Coefficent of lift.

            Returns:
                The coefficient of drag, CD.

            """
            # Recast as necessary
            CL = recastasnpfloatarray(CL)

            # Switch between models as we need to
            CDmod = A * CL ** 2 + B * CL + C
            CDquad = quadratic_adjusted(
                _CDmin=_CDmin, _k=_k, CL=CL, _CLminD=_CLminD)
            CD = np.where(CL <= CLm, CDquad, CDmod)

            if (CL > _CLmax).any():
                # warnmsg = f"Coefficient of lift exceeded CLmax={_CLmax}"
                # warnings.warn(warnmsg, category=RuntimeWarning, stacklevel=2)
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
            if dictionary is None:
                return

            for key, value in dictionary.items():

                # If the key points to an attribute of self, set the new value
                if key in self.__annotations__ or hasattr(self, key):

                    # A value that is type dict, should update the original dict
                    if isinstance(value, dict):
                        currentdict = getattr(self, key)
                        setattr(self, key, {**currentdict, **value})
                    else:
                        setattr(self, key, value)

                # The key didn't exist for self
                else:
                    errormsg = f"Unknown {key=} for {type(self).__name__}"
                    raise KeyError(errormsg)
            return

        def __getattr__(self, item):
            # If the item requested is supposed to exist for this object,
            # but hasn't yet been given a value - return this default value
            if item in self.__annotations__:
                warnmsg = f"Concept's '{item}' attribute is undefined"
                warnings.warn(warnmsg, RuntimeWarning, stacklevel=2)
                return None

            # Default behaviour
            super().__getattribute__(item)

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
        aspectratio: float
        sweep_le_deg: float
        sweep_mt_deg: float
        sweep_25_deg: float
        taperratio: float
        # Weight and loading
        weight_n: float = None
        weightfractions = dict([
            ("climb", 1.00), ("cruise", 1.00), ("servceil", 1.00),
            ("take-off", 1.00), ("turn", 1.00)
        ])

        def __init__(self, definition: dict):
            """
            Given design definition, resolve missing geometry in the wing.

            Args:
                definition: Design definition.
            """
            super().__init__(dictionary=definition)

            # Skip the warnings from trying to access undefined attributes
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Refactoring
                sLE_deg = self.sweep_le_deg
                s25_deg = self.sweep_25_deg
                sMT_deg = self.sweep_mt_deg
                AR = self.aspectratio
                TR = self.taperratio

            notNone = lambda x: x is not None
            isNone = lambda x: x is None

            # Map has to be used twice because it's a consumable generator
            if sum(map(notNone, [sLE_deg, s25_deg, sMT_deg, AR, TR])) > 3:
                errormsg = (
                    f"Wing geometry is overdefined. Consider removing a "
                    f"reference to a sweep angle or aspect/taper ratio"
                )
                raise ValueError(errormsg)
            elif sum(map(notNone, [sLE_deg, s25_deg, sMT_deg, AR, TR])) < 3:
                warnmsg = \
                    f"Wing geometry is underdefined. Filling in the gaps..."
                warnings.warn(warnmsg, RuntimeWarning, stacklevel=2)

            xcMT = 0.3  # <-- Assume location of max thickness
            xcMT_from_LEsweep = lambda x: np.interp(x, [25, 60], [0.3, 0.5])

            # While there are any undefined parameters
            while any(map(isNone, [sLE_deg, s25_deg, sMT_deg, AR, TR])):

                if notNone(TR) and TR < 0:
                    errormsg = f"Got wing taper ratio < 0 ({TR=})"
                    raise ValueError(errormsg)

                # If taper and aspect ratio are known, we can work out sweep
                if notNone(TR) and notNone(AR):
                    # No angles are known
                    if all(map(isNone, [sLE_deg, s25_deg, sMT_deg])):
                        sMT_deg = 0.0  # <-- allows straight spar
                        sLE_deg = self.sweep_m_deg(0.00, xcMT, sMT_deg)
                        s25_deg = self.sweep_m_deg(0.25, xcMT, sMT_deg)
                    # One angle is known
                    elif all(map(isNone, [sLE_deg, s25_deg])):
                        sLE_deg = self.sweep_m_deg(0.00, xcMT, sMT_deg)
                        s25_deg = self.sweep_m_deg(0.25, xcMT, sMT_deg)
                    elif all(map(isNone, [sLE_deg, sMT_deg])):
                        sLE_deg = self.sweep_m_deg(0.00, 0.25, s25_deg)
                        sMT_deg = self.sweep_m_deg(xcMT, 0.25, s25_deg)
                    elif all(map(isNone, [s25_deg, sMT_deg])):
                        s25_deg = self.sweep_m_deg(0.25, 0.00, sLE_deg)
                        xcMT = xcMT_from_LEsweep(sLE_deg)
                        sMT_deg = self.sweep_m_deg(xcMT, 0.00, sLE_deg)
                    # Two angles are known
                    elif isNone(sLE_deg):
                        sLE_deg = self.sweep_m_deg(0.00, 0.25, s25_deg)
                    elif isNone(s25_deg):
                        s25_deg = self.sweep_m_deg(0.25, 0.00, sLE_deg)
                    elif isNone(sMT_deg):
                        xcMT = xcMT_from_LEsweep(sLE_deg)
                        sMT_deg = self.sweep_m_deg(xcMT, 0.00, sLE_deg)
                    # All angles are known
                    else:
                        break  # We win! planform is fully defined

                # If only aspect ratio is known, find a taper ratio
                elif isNone(TR) and notNone(AR):
                    # No angles are known
                    if not sLE_deg and not s25_deg and not sMT_deg:
                        sMT_deg = 0.0  # <-- allows straight spar
                        TR = 1.0
                    # One angle is known
                    elif all(map(isNone, [sLE_deg, s25_deg])):
                        TR = 1.0
                    elif all(map(isNone, [sLE_deg, sMT_deg])):
                        # https://www.fzt.haw-hamburg.de/pers/Scholz/OPerA/OPerA_PRE_DLRK_12-09-10_MethodOnly.pdf
                        TR = 0.45 * np.exp(-0.0375 * s25_deg)
                    elif all(map(isNone, [s25_deg, sMT_deg])):
                        TR = 1.0
                    # Two angles are known
                    elif isNone(sLE_deg):
                        tn25, tnMT = np.tan(np.radians([s25_deg, sMT_deg]))
                        factor = AR / 4 * (tn25 - tnMT) / (xcMT - tn25)
                        TR = (1 - factor) / (1 + factor)
                    elif isNone(s25_deg):
                        tnLE, tnMT = np.tan(np.radians([sLE_deg, sMT_deg]))
                        xcMT = xcMT_from_LEsweep(sLE_deg)
                        factor = AR / 4 * (tnLE - tnMT) / (xcMT - 0)
                        TR = (1 - factor) / (1 + factor)
                    elif isNone(sMT_deg):
                        tnLE, tn25 = np.tan(np.radians([sLE_deg, s25_deg]))
                        factor = AR / 4 * (tnLE - tn25) / (0.25 - 0)
                        TR = (1 - factor) / (1 + factor)
                    # Failsafe: we shouldn't reach this if code above works
                    else:
                        raise RuntimeError("Couldn't resolve wing geometry")
                    assert notNone(TR), "Bug: TR should've been defined here!"

                # If the wing is rectangular and skewed, nothing to derive
                elif isNone(AR) and notNone(TR) and TR == 1.0:
                    # Set of angles should contain one element only
                    set_of_angles = {sLE_deg, s25_deg, sMT_deg} - {None}
                    if len({sLE_deg, s25_deg, sMT_deg} - {None}) == 1:
                        sLE_deg, = set_of_angles
                        s25_deg, = set_of_angles
                        sMT_deg, = set_of_angles
                    else:
                        raise RuntimeError("Couldn't resolve wing geometry")

                # If only taper ratio is known, find an aspect ratio
                elif isNone(AR) and notNone(TR):
                    # One or fewer angles are known
                    if sum(map(notNone, [sLE_deg, s25_deg, sMT_deg])) <= 1:
                        break
                    # Two angles are known
                    elif len({sLE_deg, s25_deg, sMT_deg} - {None}) == 1:
                        # If angles aren't unique, it's impossible to get AR
                        break  # ... i.e. the wing's chord is constant
                    # Two *unique* angles are known
                    elif isNone(sLE_deg):
                        tn25, tnMT = np.tan(np.radians([sLE_deg, s25_deg]))
                        num = 4 * (xcMT - 0.25) * (1 - TR) / (1 + TR)
                        AR = num / (tn25 - tnMT)
                    elif isNone(s25_deg):
                        tnLE, tnMT = np.tan(np.radians([sLE_deg, s25_deg]))
                        xcMT = xcMT_from_LEsweep(sLE_deg)
                        num = 4 * (xcMT - 0.00) * (1 - TR) / (1 + TR)
                        AR = num / (tnLE - tnMT)
                    elif isNone(sMT_deg):
                        tnLE, tn25 = np.tan(np.radians([sLE_deg, s25_deg]))
                        num = 4 * (tnLE - 0.25) * (1 - TR) / (1 + TR)
                        AR = num / (tn25 - 0.00)
                    # Failsafe: we shouldn't reach this if code above works
                    else:
                        raise RuntimeError("Couldn't resolve wing geometry")
                    assert notNone(AR), "Bug: AR should've been defined here!"

                # No aspect ratio or taper ratio given!
                else:
                    # Two or fewer angles are known
                    if sum(map(notNone, [sLE_deg, s25_deg, sMT_deg])) <= 2:
                        break

                    # All angles are known, find taper or aspect ratio
                    def f_TR(AR, tanM, tanN, m, n):
                        """TR as a function of AR and sweep angles/positions."""
                        ARfact = 4 / AR
                        sweepfact = (tanM - tanN) / (n - m)
                        taper = (ARfact - sweepfact) / (ARfact + sweepfact)
                        return taper

                    tnLE, tn25, tnMT = \
                        np.tan(np.radians([sLE_deg, s25_deg, sMT_deg]))
                    xcMT = xcMT_from_LEsweep(sLE_deg)

                    def f_opt(AR):
                        """Solver, reaches 0 when correct AR is found."""
                        lhs = f_TR(AR, tnLE, tn25, 0.00, 0.25)
                        rhs = f_TR(AR, tnMT, tn25, xcMT, 0.25)
                        return lhs - rhs

                    AR = optimize.newton(f_opt, x0=8)

                # Lock in our AR and TR (so the sweep function can use them)
                self.aspectratio = AR
                self.taperratio = TR

            # Save the state of the wing
            self.sweep_le_deg = sLE_deg
            self.sweep_25_deg = s25_deg
            self.sweep_mt_deg = sMT_deg
            self.sweep_le_rad = np.radians(sLE_deg)
            self.sweep_25_rad = np.radians(s25_deg)
            self.sweep_mt_rad = np.radians(sMT_deg)

            return

        @revert2scalar
        def sweep_m_rad(self, m, n=None, sweep_n_rad=None):
            """
            Arbitrary chord-line sweep for trapezoidal planform USAF DATCOM.

            Args:
                m: Desired chord fraction for which sweep is computed.
                n: Reference chord fraction n's chordwise fractional position.
                sweep_n_rad: Reference chord fraction n's sweep, in radians.

            Returns:
                Sweep of chord-line m, in radians.

            """
            # Recast as necessary
            m = recastasnpfloatarray(m)
            n = recastasnpfloatarray(0.0 if n is None else n)

            # noinspection PyUnresolvedReferences
            if (n == 0).all() and sweep_n_rad is None:
                sweep_n_rad = self.sweep_le_rad
            elif (n == 0.25).all() and sweep_n_rad is None:
                sweep_n_rad = self.sweep_25_rad

            # Refactoring
            aspectR = self.aspectratio
            taperR = self.taperratio

            ARterm = 4 / aspectR * ((m - n) * (1 - taperR) / (1 + taperR))
            sweep_m_rad = np.arctan(np.tan(sweep_n_rad) - ARterm)

            return sweep_m_rad

        # @revert2scalar <- doesn't need due to dependency on other fcn that has
        def sweep_m_deg(self, m, n=None, sweep_n_deg=None):
            """
            Arbitrary chord-line sweep for trapezoidal planform USAF DATCOM.

            Args:
                m: Desired chord fraction for which sweep is computed.
                n: Reference chord fraction n's chordwise fractional position.
                sweep_n_deg: Reference chord fraction n's sweep, in degrees.


            Returns:
                Sweep of chord-line m, in degrees.

            """
            sweep_n_rad = np.radians(sweep_n_deg)
            sweep_m_rad = self.sweep_m_rad(m=m, n=n, sweep_n_rad=sweep_n_rad)

            return np.degrees(sweep_m_rad)

    class DesignPerformance(BaseMethods):

        # Drag/resistance coefficients
        CDmin = 0.03
        mu_R = 0.03
        # Lift coefficients
        CL0 = 0.0
        CLTO: float
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

        def __init__(self, performance: dict):
            super().__init__(dictionary=performance)

            # Skip the warnings from trying to access undefined attributes
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if self.CLTO is None:
                    self.CLTO = max(
                        self.CLmaxTO - np.radians(12) * self.CLalpha, 0.0)

    return DesignBrief, DesignDefinition, DesignPerformance


ACbrief, ACdefinition, ACperformance = get_default_concept_design_objects()

# Make type hints about what should go in each dictionary
dict_acbrief = typing.TypedDict(
    "dict_acbrief", {**ACbrief.__annotations__}, total=False)
dict_acdefinition = typing.TypedDict(
    "dict_acdefinition", {**ACdefinition.__annotations__}, total=False)
dict_acperformance = typing.TypedDict(
    "dict_acperformance", {**ACperformance.__annotations__}, total=False)


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
    brief: ACbrief
    design: ACdefinition
    performance: ACperformance

    def __init__(self, brief: dict_acbrief = None,
                 design: dict_acdefinition = None,
                 performance: dict_acperformance = None,
                 atmosphere: at.Atmosphere = None,
                 propulsion: typing.Union[
                     pdecks.EngineDeck, str,
                     typing.Union[pdecks.genericdecks]] = None):
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
                    Float. Wing aspect ratio.

                sweep_le_deg
                    Float. Main wing leading edge sweep angle (in degrees).

                sweep_mt_deg
                    Float. Main wing sweep angle measured at the maximum
                    thickness point.

                sweep_25_deg
                    Float. Main wing sweep angle measured at the quarter chord
                    point.

                taperratio
                    Float. Standard definition of wing tip chord to root chord
                    ratio, zero for sharp, pointed wing-tip delta wings.

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

                CDmin
                    Float. Zero lift drag coefficient in clean configuration.
                    Optional, defaults to 0.03.

                mu_R
                    Float. Coefficient of rolling resistance on the wheels.
                    Optional, defaults to 0.03.

                CLTO
                    Float. Take-off lift coefficient. Optional.

                CLalpha
                    Float. The three-dimensional lift curve slope of the
                    aircraft, per radian. Optional, defaults to CLalpha of 5.2.

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
                    This argument does nothing if the user provides a propulsive
                    engine deck for a real-world turboprop.

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
                *SuperchargedPiston*. Optional, defaults to None.

        """

        # Recast as necessary
        self.brief = ACbrief(brief)
        self.design = ACdefinition(design)
        self.performance = ACperformance(performance)
        self.designatm = at.Atmosphere() if atmosphere is None else atmosphere

        # Propulsion handling
        if isinstance(
                propulsion,
                (pdecks.EngineDeck, pdecks.TurbofanHiBPR, pdecks.TurbofanLoBPR,
                 pdecks.Turbojet, pdecks.Turboprop, pdecks.Piston,
                 pdecks.SuperchargedPiston, pdecks.ElectricMotor)
        ):
            pass
        elif isinstance(propulsion, str):
            propulsion = pdecks.EngineDeck(f"class:{propulsion}")
        elif propulsion is None:
            propulsion = pdecks.NoDeck()
        else:
            raise ValueError(f"{propulsion=} is an invalid choice")

        # Make parameters accessible to the public
        self.propulsion = propulsion

        return

    @raise_bad_method_error
    @revert2scalar
    def ground_influence_coefficient(self, wingloading_pa, h_m=None, *,
                                     method: str = None) -> np.ndarray:
        """
        Estimate the ground influence coefficient phi, which is the
        multiplicative factor by which lift-induced drag reduces due to being
        in-ground effect. (CDi)_IGE = phi * (CDi)_OGE.

        Args:
            wingloading_pa: Aircraft MTOW wing loading, in Pascal.
            h_m: Height of the wing above the ground, in metres. Optional,
                defaults to slightly above ground-level (2 metres).
            method: Users may select any of:
                'Wieselberger'; 'Asselin'; 'McCormick';

        Returns:
            Estimate for the ground influence coefficient, phi.

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014. Section 9.5.8 on Ground
            Effect.

        """
        # Recast as necessary
        wingloading_pa = recastasnpfloatarray(wingloading_pa)
        h_m = 2.0 if h_m is None else h_m
        h_m = recastasnpfloatarray(h_m)
        method = "McCormick" if method is None else method

        wingloading_pa, h_m = np.broadcast_arrays(wingloading_pa, h_m)

        if (h_m < 0).any():
            errormsg = f"Height of wing above ground h may not be less than 0 m"
            raise ValueError(errormsg)

        # Get wing span
        aspectratio = self.design.aspectratio
        weight_n = self.design.weight_n

        if weight_n is None:
            warnmsg = f"No known concept weight, ignoring wing in-ground-effect"
            warnings.warn(warnmsg, RuntimeWarning)
            return np.ones(h_m.shape)

        S_m2 = weight_n / wingloading_pa
        b_sq = S_m2 * aspectratio

        if method == "Wieselberger":
            h_over_b = h_m / b_sq ** 0.5
            phi = 1 - (1 - 1.32 * h_over_b) / (1.05 + 7.4 * h_over_b)

        elif method == "McCormick":
            h_over_b_allsq = h_m ** 2 / b_sq
            num = 256 * h_over_b_allsq
            phi = num / (1 + num)

        elif method == "Asselin":
            h_over_b = h_m / b_sq ** 0.5
            phi = 1 - 2 / np.pi ** 2 * np.log(1 + np.pi / 8 / h_over_b)
            phi = np.clip(phi, 0, None)

        else:
            raise ValueError(f"Invalid selection {method=}.")

        return phi

    @raise_bad_method_error
    @revert2scalar
    def CLslope(self, *, method: str = None, **kwargs):
        """
        Estimate the lift-curve slope of the aircraft's main wing.

        Keyword Args:
            method: Users may select any from:
                "Helmbold"; "DATCOM";
            mach: Mach number. Optional, defaults to 0.0 (incompressible).

        Returns:
            The 3D lift slope of the main wing, CLalpha (per radian).

        """
        # Recast as necessary
        method = "DATCOM" if method is None else method

        aspectratio = self.design.aspectratio
        mach = recastasnpfloatarray(kwargs.get("mach", 0.0))

        if method == "Helmbold":
            num = 2 * np.pi * aspectratio
            den = 2 + (aspectratio ** 2 + 4) ** 0.5
            CLalpha = num / den

        elif method == "DATCOM":
            # Get 50% sweep
            sweep = self.design.sweep_m_rad(0.5)

            beta = (1 - mach ** 2) ** 0.5  # Prandtl-Glauert correction
            kappa = 0.96  # Assume ratio of 2D lift slope to 2pi is less than 1

            num = 2 * np.pi * aspectratio
            ARfactor = (aspectratio * beta / kappa) ** 2
            sweepfactor = (1 + (np.tan(sweep) / beta) ** 2)
            den = 2 + (ARfactor * sweepfactor + 4) ** 0.5

            CLalpha = num / den

        else:
            raise ValueError(f"Invalid selection {method=}")

        return CLalpha

    @raise_bad_method_error
    @revert2scalar
    def cdi_factor(self, *, method: str = None, **kwargs):
        """
        Estimate the induced drag factor (as in CD = CD_0 + k * CL^2).

        Keyword Args:
            mach: Freestream Mach number.
            method: Users may select any of:
                'Cavallo', for Oswald span efficiency, e0;
                'Brandt', for inviscid span efficiency, e;
                'Nita-Scholz', for Oswald span efficiency, e0=f(M);
                'Obert', for Oswald span efficiency, e0;
                'Kroo', for Oswald span efficiency, e0;
                'Douglas', for Oswald span efficiency, e0;

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
        mach = recastasnpfloatarray(kwargs.get("mach", 0.3))
        method = "Nita-Scholz" if method is None else method

        # Get wing aspect ratio (basically everything uses it)
        aspectratio = self.design.aspectratio
        piAR = np.pi * aspectratio

        if method == "Cavallo":
            # Find leading edge sweep
            sweep = np.radians(self.design.sweep_le_deg)

            # Compute Oswald span efficiency estimate
            ARfactor = (1 - 0.045 * aspectratio ** 0.68)
            e0_straight = 1.78 * ARfactor - 0.64
            e0_swept = 4.61 * ARfactor * np.cos(sweep) ** 0.15 - 3.1
            wt = np.interp(sweep, [0, np.radians(30)], [1, 0])
            e0_mixed = wt * e0_straight + (1 - wt) * e0_swept

            cdi_factor = 1 / (e0_mixed * piAR)

        elif method == "Brandt":
            # THIS IS ACTUALLY THE SPAN EFFICIENCY FACTOR!!!
            # Find maximum thickness sweep
            sweep = np.radians(self.design.sweep_mt_deg)

            # Compute Oswald span efficiency estimate
            sqrtterm = 4 + aspectratio ** 2 * (1 + (np.tan(sweep)) ** 2)
            e0_estimate = 2 / (2 - aspectratio + np.sqrt(sqrtterm))

            cdi_factor = 1 / (e0_estimate * piAR)

        elif method == "Nita-Scholz":
            # Find quarter chord sweep, wing taper ratio
            sweep = self.design.sweep_25_deg
            taperratio = self.design.taperratio

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
            dfuse_b = 0.114
            ke_fuse = 1 - 2 * (dfuse_b ** 2)

            # CORRECTION FACTOR D0: Correction factor due to viscous drag from
            # generated lift
            ke_d0 = 0.85

            # CORRECTION FACTOR M: Correction factor due to compressibility
            # effects on induced drag - constants from statistical analysis
            with warnings.catch_warnings():
                # Ignore expected warnings for invalid power when mach <= 0.3,
                # because NumPy evaluates for all inputs even if it doesn't end
                # up using the elements of that array in the output
                warnings.simplefilter("ignore")
                ke_mach = np.where(
                    mach <= 0.3,
                    1.0, -0.001521 * (((mach / 0.3) - 1) ** 10.82) + 1)

            e0_estimate = np.clip(e_theo * ke_fuse * ke_d0 * ke_mach, 0, None)
            if (e0_estimate == 0).any():
                warnmsg = (
                    f"Estimate for Oswald span efficiency hit 0%. Perhaps the "
                    f"Mach number is too high? (got {np.amax(mach)=:.3f})"
                )
                warnings.warn(warnmsg, RuntimeWarning, stacklevel=2)

            cdi_factor = 1 / (e0_estimate * piAR)

        elif method == "Obert":

            e0_estimate = 1 / (1.05 + 0.007 * np.pi * aspectratio)

            cdi_factor = 1 / (e0_estimate * piAR)

        elif method == "Kroo":
            # Find CDmin
            CDmin = self.performance.CDmin

            # Inviscid induced drag factor (kept in efficiency form)
            u = 0.99
            s = 0.975  # from s = 1 - 2 * (d_fuselage / b) ** 2, ratio of 11.2%
            e_inviscid = u * s

            # Viscous induced drag factor
            K = 0.38
            CDi_viscous_factor = K * CDmin

            # Oswald span efficiency from conflating CD = CDmin + k * CL^2
            # ---with--- CD = CDmin + CL^2/pi/e_inv/AR + K * CDmin * CL^2
            piAR = np.pi * aspectratio
            e0_estimate = 1 / (1 / e_inviscid + piAR * CDi_viscous_factor)

            cdi_factor = 1 / e0_estimate / piAR

        elif method == "Douglas":
            # This method was described in S. Gudmundsson and came from someone
            # called Shevell, who based the formula on unpublished stuff from
            # the Douglas aircraft company

            # Find CDmin and leading edge sweep
            CDmin = self.performance.CDmin
            sweep = self.design.sweep_le_deg

            dfuse_b = 0.114  # Take Nita-Scholz guess for fuselage diam. to span
            u = 0.985  # Correction due to non-elliptical planform u~[0.98,1.00]
            r = 0.38 - sweep / 3e3 + sweep ** 2 / 15e3  # Parasitic correction

            e0_estimate = 1 / (
                    piAR * r * CDmin
                    + 1 / ((1 + 0.03 * dfuse_b - 2 * dfuse_b ** 2) * u)
            )

            cdi_factor = 1 / e0_estimate / piAR

        elif method == "Raymer":
            raise NotImplementedError("Awaiting implementation")

        else:
            raise ValueError(f"Invalid selection {method=}.")

        return cdi_factor

    @revert2scalar
    def get_bestV_BG(self, wingloading_pa, altitude_m=None):
        """
        Estimate the speed, VBG, which results in best gliding performance
        (maximised operating L/D).

        Args:
            wingloading_pa: Aircraft wing loading, in Pascal.
            altitude_m: Geopotential altitude, in metres. Optional, defaults to
                sea-level (0 metres).

        Returns:
            Best speed for glide VBG, in metres per second.

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        Notes:
            Not only is this the best glide speed for jet and propeller-powered
            aicraft, it is also the best endurance speed for jets and the best
            range speed for propeller aircraft.

        """
        # Recast as necessary
        wingloading_pa = recastasnpfloatarray(wingloading_pa)
        altitude_m = 0 if altitude_m is None else altitude_m
        altitude_m = recastasnpfloatarray(altitude_m)
        wingloading_pa, altitude_m = \
            np.broadcast_arrays(wingloading_pa, altitude_m)

        # The bestCL range/endurance functions maximise CL/CD @ constant speed V
        bestCL, _ = self.get_bestCL_range(constantspeed=True)

        rho_kgpm3 = self.designatm.airdens_kgpm3(altitude_m)
        bestspeed_mps = (2 / rho_kgpm3 * wingloading_pa / bestCL) ** 0.5

        return bestspeed_mps

    @revert2scalar
    def get_bestV_CAR(self, wingloading_pa, altitude_m=None):
        """
        Estimate the speed, VCAR, which results in best jet range. This is also
        known as Carson's speed, for jet and propeller aircraft.

        Args:
            wingloading_pa: Aircraft wing loading, in Pascal.
            altitude_m: Geopotential altitude, in metres. Optional, defaults to
                sea-level (0 metres).

        Returns:
            Best speed for jet range VCAR, in metres per second.

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        """
        # Recast as necessary
        wingloading_pa = recastasnpfloatarray(wingloading_pa)
        altitude_m = 0 if altitude_m is None else altitude_m
        altitude_m = recastasnpfloatarray(altitude_m)
        wingloading_pa, altitude_m = \
            np.broadcast_arrays(wingloading_pa, altitude_m)

        # Gudmundsson, eq. (19-22)
        # The bestCL range function maximise CL**0.5/CD @ variable speed V
        bestCL, _ = self.get_bestCL_range(constantspeed=False)

        rho_kgpm3 = self.designatm.airdens_kgpm3(altitude_m)
        bestspeed_mps = (2 / rho_kgpm3 * wingloading_pa / bestCL) ** 0.5

        return bestspeed_mps

    @revert2scalar
    def get_bestV_X(self, wingloading_pa, altitude_m=None):
        """
        Estimate the speed, VX, which results in the best angle of climb.

        Args:
            wingloading_pa: Aircraft wing loading, in Pascal.
            altitude_m: Geopotential altitude, in metres. Optional, defaults to
                sea-level (0 metres).

        Returns:
            Best climb angle speed VX, in metres per second.

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        """
        # Recast as necessary
        wingloading_pa = recastasnpfloatarray(wingloading_pa)
        altitude_m = 0 if altitude_m is None else altitude_m
        altitude_m = recastasnpfloatarray(altitude_m)
        wingloading_pa, altitude_m = \
            np.broadcast_arrays(wingloading_pa, altitude_m)

        CDmin = self.performance.CDmin
        weight_n = self.design.weight_n

        rho_kgpm3 = self.designatm.airdens_kgpm3(altitude_m)
        vsound_mps = self.designatm.vsound_mps(altitude_m)

        if self.propulsion.type in ["turbofan", "turbojet"]:

            bestCL, LDmax = self.get_bestCL_endurance(constantspeed=True)
            densfactor = 2 / rho_kgpm3

            def f_obj(VX, i):
                """Helper func: Objective function. Accepts guess of VX."""

                # Gudmundsson, eq. (18-21)
                mach = VX / vsound_mps.flat[i]
                thrust = self.propulsion.thrust(
                    mach, altitude_m.flat[i], norm=False
                )
                theta_max = np.arcsin(np.clip(
                    thrust / weight_n - 1 / LDmax,
                    None, 1  # Limit theta_max to 90 degrees climb angle
                ))

                # Gudmundsson, eq. (18-22)
                bestspeed_guess_mps = \
                    (densfactor
                     * wingloading_pa / bestCL
                     * np.cos(theta_max)
                     ).flat[i] ** 0.5

                return bestspeed_guess_mps

            def f_opt(VX, i):
                """Helper func: Solve for VX."""
                return f_obj(VX, i) - VX

        elif self.propulsion.type in ["turboprop", "piston", "electricmotor"]:

            eta_prop = self.performance.eta_prop["climb"]
            weight_n = self.design.weight_n

            def f_opt(VX, i):
                """Helper func: Solve for VX, for each coefficient index i."""

                # Gudmundsson, eq. (18-26)
                mach = VX / vsound_mps.flat[i]
                shaftpower = self.propulsion.shaftpower(
                    mach, altitude_m.flat[i], norm=False)
                power = eta_prop * shaftpower
                c1 = power / rho_kgpm3.flat[i] / (
                        weight_n / wingloading_pa.flat[i]) / CDmin
                k = self.cdi_factor(mach=mach, method="Nita-Scholz")
                c2 = (wingloading_pa.flat[i] ** 2 * 4 * k
                      / rho_kgpm3.flat[i] ** 2 / CDmin)

                return VX ** 4 + c1 * VX - c2

        else:
            raise NotImplementedError("Unsupported propulsion system type")

        bestspeed_mps = np.array([
            optimize.newton(f_opt, x0=100.0, args=(i,))
            for i, _ in enumerate(wingloading_pa.flat)
        ]).reshape(wingloading_pa.shape)

        return bestspeed_mps

    @revert2scalar
    def get_bestV_Y(self, wingloading_pa, altitude_m=None):
        """
        Estimate the speed, VY, which results in the best rate of climb.

        Args:
            wingloading_pa: Aircraft wing loading, in Pascal.
            altitude_m: Geopotential altitude, in metres. Optional, defaults to
                sea-level (0 metres).

        Returns:
            Best rate of climb speed VY, in metres per second.

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        Notes:
            Because this is contingent on minimising the power requirement for
            flight (to maximise specific excess power), this is also the best
            endurance speed for propeller-driven aircraft.

        """
        # Recast as necessary
        wingloading_pa = recastasnpfloatarray(wingloading_pa)
        altitude_m = 0 if altitude_m is None else altitude_m
        altitude_m = recastasnpfloatarray(altitude_m)

        wingloading_pa, altitude_m = \
            np.broadcast_arrays(wingloading_pa, altitude_m)
        CDmin = self.performance.CDmin

        rho_kgpm3 = self.designatm.airdens_kgpm3(altitude_m)
        vsound_mps = self.designatm.vsound_mps(altitude_m)

        if self.propulsion.type in ["turbofan", "turbojet"]:

            wsfactor = wingloading_pa / 3 / rho_kgpm3 / CDmin
            bestCL, LDmax = self.get_bestCL_endurance(constantspeed=True)

            def f_obj(VY, i):
                """Helper func: Objective function. Accepts guess of VY."""

                # Gudmundsson, eq. (18-24)
                mach = VY / vsound_mps.flat[i]
                thrust_n = self.propulsion.thrust(
                    mach, altitude_m.flat[i], norm=False)
                weight_n = self.design.weight_n
                tw = thrust_n / weight_n
                ldfactor = 1 + (1 + 3 / LDmax ** 2 / tw ** 2) ** 0.5
                bestspeed_guess_mps = (tw * wsfactor.flat[i] * ldfactor) ** 0.5

                return bestspeed_guess_mps

            # def f_opt(VY, i):
            #     """Helper func: Solve for VY."""
            #     return f_obj(VY, i) - VY

            bestspeed_mps = np.array([
                optimize.newton(
                    lambda v, arg: f_obj(v, arg) - v, x0=100.0, args=(i,))
                for i, _ in enumerate(wingloading_pa.flat)
            ]).reshape(wingloading_pa.shape)

            return bestspeed_mps

        elif self.propulsion.type in ["turboprop", "piston", "electricmotor"]:

            # Gudmundsson, eq. (20-21) says that...
            # We need to maximise specific excess power
            # ... which happens when we minimise power required
            # ... which happens in eq. (20-21) at best endurance condition
            bestCL, _ = self.get_bestCL_endurance(constantspeed=False)
            bestspeed_mps = (2 / rho_kgpm3 * wingloading_pa / bestCL) ** 0.5

            return bestspeed_mps

        raise NotImplementedError("Unsupported propulsion system type")

    @revert2scalar
    def get_bestCL_range(self, constantspeed: bool = None) -> tuple:
        """
        Finds the best CL (and L/D ratio) for the cruising conditions that
        maximise range in constant attitude cruise profiles.

        Args:
            constantspeed: Flags whether constraint considers cruise speed as
                a constant over the particular cruise profile. Optional,
                defaults to False (constant altitude/constant attitude cruise).

        Returns:
            Tuple of (the best coefficient of lift, L/D ratio at this CL).

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        Notes:
            Gudmundsson, section 20.2 "Range Analysis," provides several
            equations (20-10, 20-11, 20-12) that are all proportional to
            CL ** exponent / CD. This method simply maximises that term.
            If you're just trying to maximise L/D, set 'constantspeed' to True.

        """
        # Recast as necessaary
        constantspeed = False if constantspeed is None else constantspeed
        exponent = 0.5 if constantspeed is False else 1.0

        CDmin = self.performance.CDmin
        CLmax = self.performance.CLmax
        CLminD = self.performance.CLminD

        # Maximising CL / CD is the same as: Minimise -1.0 * CL / CD
        k = self.cdi_factor()
        f_CD = make_modified_drag_model(CDmin, k, CLmax, CLminD)
        bestCL, = optimize.minimize(
            lambda x: -x ** exponent / f_CD(x), x0=np.array([0.8]),
            bounds=((CLminD, CLmax),)
        ).x
        LDratio = bestCL / f_CD(bestCL)

        return bestCL, LDratio

    @revert2scalar
    def get_bestCL_endurance(self, constantspeed: bool = None) -> tuple:
        """
        Finds the best CL (and L/D ratio) for the cruising conditions that
        maximise endurance in constant attitude cruise profiles.

        Args:
            constantspeed: Flags whether constraint considers cruise speed as
                a constant over the particular cruise profile. Optional,
                defaults to False (constant altitude/constant attitude cruise).

        Returns:
            Tuple of (the best coefficient of lift, L/D ratio at this CL).

        References:
            Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures," 1st ed., Elselvier, 2014.

        Notes:
            Gudmundsson, section 20.4 "... Endurance Analysis," provides several
            equations (20-19, 20-20, 20-21, 20-22) that are all proportional to
            CL ** exponent / CD. This method simply maximises that term.
            If you're just trying to maximise L/D, set 'constantspeed' to True.

        """
        # Recast as necessaary
        constantspeed = False if constantspeed is None else constantspeed
        if self.propulsion.type in ["piston", "turboprop", "electricmotor"]:
            exponent = 1.5 if constantspeed is False else 1.0
        elif self.propulsion.type in ["turbojet", "turbofan"]:
            exponent = 1.0
        else:
            raise NotImplementedError("Unsupported propulsion system type")

        CDmin = self.performance.CDmin
        CLmax = self.performance.CLmax
        CLminD = self.performance.CLminD

        # Maximising CL / CD is the same as: Minimise -1.0 * CL / CD
        k = self.cdi_factor()
        f_CD = make_modified_drag_model(CDmin, k, CLmax, CLminD)
        bestCL, = optimize.minimize(
            lambda x: -x ** exponent / f_CD(x), x0=np.array([0.8]),
            bounds=((CLminD, CLmax),)
        ).x
        LDratio = bestCL / f_CD(bestCL)

        return bestCL, LDratio

    def constrain_climb(self, wingloading_pa, **kwargs):
        """
        Compute the thrust-to-weight and the power-to-weight (if applicable)
        requirements to climb as prescribed in the design brief for the concept.

        Args:
            wingloading_pa: MTOW wing loading, in Pascal.

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
        wingloading_pa = recastasnpfloatarray(wingloading_pa)
        climbalt_m = kwargs.get("climbalt_m", self.brief.climbalt_m)
        climbspeed_kias = kwargs.get(
            "climbspeed_kias", self.brief.climbspeed_kias)
        climbrate_fpm = kwargs.get("climbrate_fpm", self.brief.climbrate_fpm)
        CDmin = kwargs.get("CDmin", self.performance.CDmin)
        CLmax = kwargs.get("CLmax", self.performance.CLmax)
        CLminD = kwargs.get("CLminD", self.performance.CLminD)
        methods = kwargs.get("methods", dict())

        # Determine the thrust and power lapse corrections
        climbspeed_mpsias = co.kts_mps(climbspeed_kias)
        climbspeed_mpstas = self.designatm.eas2tas(
            eas=climbspeed_mpsias,
            altitude_m=climbalt_m
        )
        mach = climbspeed_mpstas / self.designatm.vsound_mps(climbalt_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude_m=climbalt_m, norm=True,
            eta_prop=self.performance.eta_prop["climb"]
        )
        if self.propulsion.type in ["piston", "turboprop", "electricmotor"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude_m=climbalt_m, norm=True
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
        wslim_pa = CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            # warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            # warnings.warn(warnmsg, RuntimeWarning, stacklevel=2)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... load factor due to climb
        climbrate_mps = co.fpm_mps(climbrate_fpm)
        climbrate_mpstroc = self.designatm.eas2tas(
            eas=climbrate_mps, altitude_m=climbalt_m)
        cos_theta = (1 - (climbrate_mpstroc / climbspeed_mpstas) ** 2) ** 0.5

        # ... coefficient of drag
        k = self.cdi_factor(mach=mach, method=methods.get("cdi_factor"))
        f_CD = make_modified_drag_model(CDmin, k, CLmax, CLminD)
        CL = ws_pa * cos_theta / q_pa
        CD = f_CD(CL)

        # ... "acceleration factor"
        Ka = 1.0  # Small climb angle approximation! dV/dh ~ 0...

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CD / ws_pa
                     + Ka * climbrate_mpstroc / climbspeed_mpstas
             ) / tcorr * wcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * climbspeed_mpstas / pcorr

        return tw, pw

    def constrain_cruise(self, wingloading_pa, **kwargs):
        """
        Compute the thrust-to-weight and the power-to-weight (if applicable)
        requirements to cruise as prescribed in the design brief for the
        concept.

        Args:
            wingloading_pa: MTOW wing loading, in Pascal.

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
        wingloading_pa = recastasnpfloatarray(wingloading_pa)
        cruisealt_m = kwargs.get("cruisealt_m", self.brief.cruisealt_m)
        cruisespeed_ktas = kwargs.get(
            "cruisespeed_ktas", self.brief.cruisespeed_ktas)
        cruisethrustfact = kwargs.get(
            "cruisethrustfact", self.brief.cruisethrustfact)
        CDmin = kwargs.get("CDmin", self.performance.CDmin)
        CLmax = kwargs.get("CLmax", self.performance.CLmax)
        CLminD = kwargs.get("CLminD", self.performance.CLminD)
        methods = kwargs.get("methods", dict())

        # Determine the thrust and power lapse corrections
        cruisespeed_mpstas = co.kts_mps(cruisespeed_ktas)
        mach = cruisespeed_mpstas / self.designatm.vsound_mps(cruisealt_m)
        tcorr = cruisethrustfact * self.propulsion.thrust(
            mach=mach, altitude_m=cruisealt_m, norm=True,
            eta_prop=self.performance.eta_prop["cruise"]
        )
        if self.propulsion.type in ["piston", "turboprop", "electricmotor"]:
            pcorr = cruisethrustfact * self.propulsion.shaftpower(
                mach=mach, altitude_m=cruisealt_m, norm=True
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
        wslim_pa = CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            # warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            # warnings.warn(warnmsg, RuntimeWarning, stacklevel=2)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... coefficient of drag
        k = self.cdi_factor(mach=mach, method=methods.get("cdi_factor"))
        f_CD = make_modified_drag_model(CDmin, k, CLmax, CLminD)
        CL = ws_pa / q_pa
        CD = f_CD(CL)

        # ... T/W (mapped to sea-level static)
        tw = (q_pa * CD / ws_pa) / tcorr * wcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * cruisespeed_mpstas / pcorr

        return tw, pw

    def constrain_servceil(self, wingloading_pa, **kwargs):
        """
        Compute the thrust-to-weight and the power-to-weight (if applicable)
        requirements to fly at the service ceiling as prescribed in the design
        brief for the concept.

        Args:
            wingloading_pa: MTOW wing loading, in Pascal.

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
        wingloading_pa = recastasnpfloatarray(wingloading_pa)
        servceil_m = kwargs.get("servceil_m", self.brief.servceil_m)
        secclimbspd_kias = kwargs.get(
            "secclimbspd_kias", self.brief.secclimbspd_kias)
        CDmin = kwargs.get("CDmin", self.performance.CDmin)
        CLmax = kwargs.get("CLmax", self.performance.CLmax)
        CLminD = kwargs.get("CLminD", self.performance.CLminD)
        methods = kwargs.get("methods", dict())

        # Determine the thrust and power lapse corrections
        secclimbspd_mpsias = co.kts_mps(secclimbspd_kias)
        secclimbspd_mpstas = self.designatm.eas2tas(
            eas=secclimbspd_mpsias,
            altitude_m=servceil_m
        )
        mach = secclimbspd_mpstas / self.designatm.vsound_mps(servceil_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude_m=servceil_m, norm=True,
            eta_prop=self.performance.eta_prop["servceil"]
        )
        if self.propulsion.type in ["piston", "turboprop", "electricmotor"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude_m=servceil_m, norm=True
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
        wslim_pa = CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            # warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            # warnings.warn(warnmsg, RuntimeWarning, stacklevel=2)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... load factor due to climb
        # Service ceiling typically defined in terms of climb rate (at best
        # climb speed) dropping to 100feet/min ~ 0.508m/s
        climbrate_mps = co.fpm_mps(100)
        # What true climb rate does 100 feet/minute correspond to?
        climbrate_mpstroc = self.designatm.eas2tas(
            eas=climbrate_mps, altitude_m=servceil_m)
        cos_theta = (1 - (climbrate_mpstroc / secclimbspd_mpstas) ** 2) ** 0.5

        # ... coefficient of drag
        k = self.cdi_factor(mach=mach, method=methods.get("cdi_factor"))
        f_CD = make_modified_drag_model(CDmin, k, CLmax, CLminD)
        CL = ws_pa * cos_theta / q_pa
        CD = f_CD(CL)

        # ... T/W (mapped to sea-level static)
        tw = (
                     q_pa * CD / ws_pa
                     + climbrate_mpstroc / secclimbspd_mpstas
             ) / tcorr * wcorr

        # ... P/W (mapped to sea-level static)
        pw = (tw * tcorr) * secclimbspd_mpstas / pcorr

        return tw, pw

    def constrain_takeoff(self, wingloading_pa, **kwargs):
        """
        Compute the thrust-to-weight and the power-to-weight (if applicable)
        requirements to perform take-off as prescribed in the design brief for
        the concept.

        Args:
            wingloading_pa: MTOW wing loading, in Pascal.

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
        wingloading_pa = recastasnpfloatarray(wingloading_pa)
        groundrun_m = kwargs.get("groundrun_m", self.brief.groundrun_m)
        rwyelevation_m = kwargs.get("rwyelevation_m", self.brief.rwyelevation_m)
        CDmin = kwargs.get("CDmin", self.performance.CDmin)
        CLminD = kwargs.get("CLminD", self.performance.CLminD)
        mu_R = kwargs.get("mu_R", self.performance.mu_R)
        CLTO = kwargs.get("CLTO", self.performance.CLTO)
        CLmaxTO = kwargs.get("CLmaxTO", self.performance.CLmaxTO)
        methods = kwargs.get("methods", dict())

        # ... coefficient of drag
        k = self.cdi_factor(method=methods.get("cdi_factor"))
        f_CD = make_modified_drag_model(CDmin, k, CLmaxTO, CLminD)
        CDTO_OGE = f_CD(CLTO)

        # Correct CDTO using ground effect with a wing height of 2 m.
        phi = self.ground_influence_coefficient(
            wingloading_pa, h_m=2,
            method=methods.get("ground_influence_coefficient")
        )
        # ... Assume CDminTO ~ CDmin, ground effect only affects induced drag
        CDTO_IGE = CDmin + phi * (CDTO_OGE - CDmin)

        # Determine the weight lapse (relative to MTOW) correction
        wcorr = self.design.weightfractions["take-off"]

        # Approximation for lift-off speed
        ws_pa = wingloading_pa * wcorr
        airdensity_kgpm3 = self.designatm.airdens_kgpm3(rwyelevation_m)
        vs1to_mps = (2 * ws_pa / airdensity_kgpm3 / CLmaxTO) ** 0.5
        vrotate_mps = vs1to_mps
        vliftoff_mps = vrotate_mps

        # Determine the thrust and power lapse corrections
        # 75% comes from taking the centre of mass of a V vs q diagram for when
        # linear acceleration is assumed
        vbar_mpstas = 0.75 * vliftoff_mps
        machbar = vbar_mpstas / self.designatm.vsound_mps(rwyelevation_m)
        # (maybe we should use the altitude=0 sea-level take-off method below?)
        tcorr = self.propulsion.thrust(
            mach=machbar, altitude_m=rwyelevation_m, norm=True,
            eta_prop=self.performance.eta_prop["take-off"]
        )
        if self.propulsion.type in ["piston", "turboprop", "electricmotor"]:
            pcorr = self.propulsion.shaftpower(
                mach=machbar, altitude_m=rwyelevation_m, norm=True
            ) * self.performance.eta_prop["take-off"]
        else:
            pcorr = np.nan

        # Compute take-off constraint
        # ... T/W (mapped to sea-level static)
        acctakeoffbar = vliftoff_mps ** 2 / 2 / groundrun_m
        tw_penalty = np.where(mu_R > CDTO_IGE / CLTO, mu_R, CDTO_IGE / CLTO)
        tw = (acctakeoffbar / constants.g + tw_penalty) / tcorr * wcorr

        # ... P/W (mapped to sea-level static)
        # Not sure if using vbar_mpstas is correct, but we used it for thrust???
        pw = (tw * tcorr) * vbar_mpstas / pcorr

        return tw, pw

    def constrain_turn(self, wingloading_pa, **kwargs):
        """
        Compute the thrust-to-weight and the power-to-weight (if applicable)
        requirements to carry out the constrained turn as prescribed in the
        design brief for the concept.

        Args:
            wingloading_pa: MTOW wing loading, in Pascal.

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
        wingloading_pa = recastasnpfloatarray(wingloading_pa)
        stloadfactor = kwargs.get("stloadfactor", self.brief.stloadfactor)
        turnalt_m = kwargs.get("turnalt_m", self.brief.turnalt_m)
        turnspeed_ktas = kwargs.get("turnspeed_ktas", self.brief.turnspeed_ktas)
        CDmin = kwargs.get("CDmin", self.performance.CDmin)
        CLmax = kwargs.get("CLmax", self.performance.CLmax)
        CLminD = kwargs.get("CLminD", self.performance.CLminD)
        methods = kwargs.get("methods", dict())

        # Determine the thrust and power lapse corrections
        turnspeed_mpstas = co.kts_mps(turnspeed_ktas)
        mach = turnspeed_mpstas / self.designatm.vsound_mps(turnalt_m)
        tcorr = self.propulsion.thrust(
            mach=mach, altitude_m=turnalt_m, norm=True,
            eta_prop=self.performance.eta_prop["turn"]
        )
        if self.propulsion.type in ["piston", "turboprop", "electricmotor"]:
            pcorr = self.propulsion.shaftpower(
                mach=mach, altitude_m=turnalt_m, norm=True
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
        wslim_pa = CLmax * q_pa
        if (ws_pa > wslim_pa).any():
            # warnmsg = f"Wing loading exceeded limit of {wslim_pa:.0f} Pascal!"
            # warnings.warn(warnmsg, RuntimeWarning, stacklevel=2)
            ws_pa[ws_pa > wslim_pa] = np.nan

        # ... coefficient of drag
        k = self.cdi_factor(mach=mach, method=methods.get("cdi_factor"))
        f_CD = make_modified_drag_model(CDmin, k, CLmax, CLminD)
        CL = ws_pa * stloadfactor / q_pa
        CD = f_CD(CL)

        # ... T/W (mapped to sea-level static)
        tw = (q_pa * CD / ws_pa) / tcorr * wcorr

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
        stallspeed_mpstas = co.kts_mps(vstallclean_kcas)

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

    def plot_constraints(self, wingloading_pa, **kwargs):
        """
        Make a pretty figure with all the constraints outlined in the design
        brief. Depending on the contents of the aircraft concept's brief,
        design, and performance arguments, thrust-to-weight, power-to-weight,
        thrust, and power graphs are intelligently selected from.

        Args:
            wingloading_pa: An ordered array of MTOW wing loading values across
                which the concept's constraints should be evaluated.
            **kwargs: keyword arguments to pass onto the constraint methods.

        Returns:
            A tuple of the matplotlib (Figure, Axes) objects used to plot all
            the data.

        """
        # Recast as necessary
        wingloading_pa = recastasnpfloatarray(wingloading_pa)

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
                tws[label], pws[label] = function(wingloading_pa, **kwargs)
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
        type_is_power = \
            self.propulsion.type in ["piston", "turboprop", "electricmotor"]
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
            errormsg = f"{self.plot_constraints.__name__} found no values"
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
                xstall, label="clean stall, 1$g$", c=clr, zorder=10, lw=2)
            ax.fill_betweenx(*yx1x2, fc=l2d.get_color(), alpha=0.1, zorder=5)

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

        def custom_legend_maker(**kwargs):
            """A matplotlib legend but with default position parameters."""
            # Squash the handles of legend handles shared by more than one label
            handles, labels = ax.get_legend_handles_labels()
            legenddict = {l: tuple(np.array(handles)[np.array(labels) == l])
                          for l in labels}
            legenddict = dict(
                zip(("labels", "handles"), zip(*legenddict.items())))

            default_legend_kwargs = dict(
                [("bbox_to_anchor", (1.05, 0.98)), ("loc", "upper left")])
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
                                       functions=(co.Pa_lbfft2, co.lbfft2_Pa))
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
        ax2_x = ax.secondary_xaxis(yloc, functions=(co.m2_ft2, co.ft2_m2))
        ax2_x.set_xlabel("[ft$^2$]")
        ax2_x.xaxis.set_label_coords(1.09, -0.02)
        ax2_x.set_xticks([])
        # x, label
        ax3_x = ax.secondary_xaxis(yloc, functions=(co.m2_ft2, co.ft2_m2))
        ax3_x.set_xlabel("Wing Area")
        if type_is_power:
            # y, metric
            ax.set_yticks([])
            ax.set_ylabel("[W]")
            ax.yaxis.label.set_color((1.0, 0, 0, 0))
            # y, metric & imperial
            secax_y = ax.secondary_yaxis(0, functions=(co.W_kW, co.kW_W))
            secax_y.set_ylabel("[hp] | [kW]" + " " * 28, rotation=0)
            secax_y.yaxis.set_label_coords(0.0, -.1)  # x-transform is useless
            # y, label
            terax_y = ax.secondary_yaxis(-.14, functions=(co.W_hp, co.hp_W))
            terax_y.set_ylabel("Shaft Power")
        else:
            # y, metric
            ax.set_yticks([])
            ax.set_ylabel("[N]")
            ax.yaxis.label.set_color((1.0, 0, 0, 0))
            # y, metric & imperial
            secax_y = ax.secondary_yaxis(0, functions=(co.N_kN, co.kN_N))
            secax_y.set_ylabel("[lbf] | [kN]" + " " * 22, rotation=0)
            secax_y.yaxis.set_label_coords(0.0, -.1)  # x-transform is useless
            # y, label
            terax_y = ax.secondary_yaxis(-.14, functions=(co.N_lbf, co.lbf_N))
            terax_y.set_ylabel("Thrust")

        return fig, ax

    def plot_planform(self):

        print("Sorry, method doesn't exist yet. Here's some planform data:")
        print(f"\tAspect Ratio: {self.design.aspectratio}")
        print(f"\tTaper Ratio: {self.design.taperratio}")
        print(f"\tSweep (LE) [deg]: {self.design.sweep_le_deg}")
        print(f"\tSweep (QC) [deg]: {self.design.sweep_25_deg}")
        print(f"\tSweep (MT) [deg]: {self.design.sweep_mt_deg}")

        return
