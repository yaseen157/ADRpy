"""
Module for modelling the performance of engines, using "engine performance
decks" - derived from real engine data.
"""
import os
import re
import typing
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CloughTocher2DInterpolator

from ADRpy import atmospheres as at
from ADRpy.mtools4acdc import recastasnpfloatarray
from ADRpy import unitconversions as uc

__all__ = ["engine_catalogue", "EngineDeck", "TurbofanHiBPR", "TurbofanLoBPR",
           "Turbojet", "Turboprop"]
__author__ = "Yaseen Reza"

# Other contributors: Samuel Pearson

# Locate engine decks
engine_data_path = (
    os.path.join(os.path.dirname(__file__), "data", "engine_data"))
engine_deck_types = os.listdir(engine_data_path)

# Catalogue (paths to) decks
engine_catalogue = dict()
for deck_type in engine_deck_types:
    deck_path = os.path.join(engine_data_path, deck_type)

    for filename in os.listdir(deck_path):

        # Find engine data files
        match = re.findall(r"([^_]+)\.xlsx", filename)
        if not match:
            continue
        filepath = os.path.join(deck_path, filename)
        enginename, = match

        # Add to catalogue
        engine_catalogue[enginename] = dict([
            ("path", filepath),
            ("type", deck_type),
            ("dataframes", dict())
        ])

isaref = at.Atmosphere()


class EngineDeck:
    """Base class for all engine decks."""
    name: str
    type: str
    _dataframes: dict
    _f_thrust = NotImplemented
    _f_thrust_SLTO = NotImplemented

    def __new__(cls, engine: str):

        validtypes = {  # generic deck types
            "turbofanHiBPR": TurbofanHiBPR,
            "turbofanLoBPR": TurbofanLoBPR,
            "turbojet": Turbojet,
            "turboprop": Turboprop,
            "piston": Piston
        }
        # If the engine specified is actually in the form "type:<x>"
        if engine.startswith("type:"):

            enginetype = engine.lstrip("type:")
            genericclass = validtypes.get(enginetype)

            if genericclass is None:
                errormsg = f"{engine=} not recognised. Try one of {validtypes=}"
                raise ValueError(errormsg)
            return genericclass()

        # Default behaviour, create an instance and then run the __init__ method
        instance = super(EngineDeck, cls).__new__(cls)
        instance.__init__(engine)
        return instance

    def __init__(self, engine: str):
        """
        Args:
            engine: Name of an engine for which data exists, or a string in the
                form "type:<x>" where <x> is any of the following:

                - turbofanHiBPR
                - turbofanLoBPR
                - turbojet
                - turboprop
                - piston

        """
        if engine not in engine_catalogue:
            errormsg = (
                f"'{engine}' not found in list of available engines: "
                f"{list(engine_catalogue.keys())}")
            raise ValueError(errormsg)

        # Check if the engine has data cached already, otherwise load it
        if engine_catalogue[engine]["dataframes"] == {}:
            engine_catalogue[engine]["dataframes"] = pd.read_excel(
                engine_catalogue[engine]["path"], sheet_name=None)

        self.name = engine
        self.type = engine_catalogue[engine]["type"]
        self._dataframes = engine_catalogue[engine]["dataframes"]

        build_deck(self)
        return

    def __repr__(self):
        reprstr = f"{type(self).__name__}('{self.name}')"
        return reprstr

    @property
    def dataframes(self):
        """The raw dataframes extracted from engine data files."""
        return self._dataframes

    def thrust(self, mach, altitude, *, norm: bool = None):
        """
        Return the thrust available at the given flight conditions.

        Args:
            mach: Flight Mach number.
            altitude: Flight level (above mean sea level), in metres.
            norm: Flag, whether to normalise the output to the sea-level static
                thrust. Optional, defaults to False (no normalisation).

        Returns:
            The thrust developed by the propulsion system, in Newtons. If
            norm = True, this is just the ratio of available thrust to a static
            sea-level performance datum.

        """
        # Recast as necessary
        mach = recastasnpfloatarray(mach)
        altitude = recastasnpfloatarray(altitude)
        norm = False if norm is None else norm

        # Compute thrust (and normalise to sea-level static if needed)
        thrust = self._f_thrust(mach, altitude)
        if norm is True:
            thrust = thrust / self._f_thrust(0, 0)

        return thrust

    def thrust_slto(self, mach, *, norm: bool = None):
        """
        Return the (augmented) thrust during takeoff, at sea level.

        Args:
            mach: Freestream Mach number.
            norm: Flag, whether to normalise the output to the sea-level static
                thrust. Optional, defaults to False (no normalisation).

        Returns:
            The thrust developed by the propulsion system, in Newtons. If
            norm = True, this is just the ratio of available thrust to a static
            sea-level performance datum.

        """
        # Recast as necessary
        mach = recastasnpfloatarray(mach)
        norm = False if norm is None else norm

        # Compute thrust (and normalise to sea-level static if needed)
        if self._f_thrust_SLTO is not NotImplemented:
            thrust = self._f_thrust_SLTO(mach, 0.0)
            if norm is True:
                thrust = thrust / self._f_thrust(0, 0)
        else:
            thrust = self.thrust(mach=mach, altitude=0.0, norm=norm)

        return thrust


def TP0ratio(mach: np.ndarray, altitude: np.ndarray, atmosphere=None):
    """
    Compute the stagnation temperature and pressure ratio (to sea-level static).

    Args:
        mach: Freestream Mach number.
        altitude: Geopotential altitude, in metres.
        atmosphere: An atmosphere object. Optional, defaults to ISA.

    Returns:
        A tuple (theta0, delta0).

    """
    # Recast as necessary
    atmosphere = isaref if atmosphere is None else atmosphere

    # Static quantity ratios
    temp_K = uc.c2k(isaref.airtemp_c(altitude))
    theta = temp_K / 288.15  # static temperature ratio
    press_Pa = isaref.airpress_pa(altitude)
    delta = press_Pa / 101325  # static pressure ratio

    # Convert to stagnation ratios
    gamma = 1.4
    theta0 = theta * (1 + (gamma - 1) / 2 * mach ** 2)
    delta0 = delta * (1 + (gamma - 1) / 2 * mach ** 2) ** (gamma / (gamma - 1))

    return theta0, delta0


class TurbofanHiBPR:
    """
    Performance deck for a high bypass ratio, subsonic turbofan.

    References:
        -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
            2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.
    """
    name = "generic:TurbofanHiBPR"
    type = "turbofan"

    @classmethod
    def thrust(cls, mach, altitude, *, norm: bool = True, **kwargs):
        """
        Return the thrust available at the given flight conditions.

        Args:
            mach: Flight Mach number.
            altitude: Flight level (above mean sea level), in metres.
            norm: For generic propulsion system types, this must be set to True.

        Keyword Args:
            atmosphere: Alternative atmosphere object, if one is desired when
                computing stagnation temperature and pressure ratios.

        Returns:
            The thrust developed by the propulsion system, in Newtons. If
            norm = True, this is just the ratio of available thrust to a static
            sea-level performance datum.

        """
        # Recast as necessary
        mach = recastasnpfloatarray(mach)
        altitude = recastasnpfloatarray(altitude)

        if norm is not True:
            errormsg = f"{cls.name} must use norm=True, found that {norm=}"
            raise RuntimeError(errormsg)

        theta0, delta0 = TP0ratio(mach, altitude, kwargs.get("atmosphere"))
        # Assume theta break (throttle ratio) of 1.05
        TR = 1.05

        lapse = delta0 * (1 - 0.49 * mach ** 0.5)
        slice = theta0 > TR
        lapse[slice] -= (delta0 * 3 * (theta0 - TR) / (1.5 + mach))[slice]

        return np.clip(lapse, 0, None)

    @classmethod
    def thrust_slto(cls, mach, **kwargs):
        """
        Return the takeoff thrust at sea level.

        Args:
            mach: Freestream Mach number.

        Keyword Args:
            norm: For generic propulsion system types, this must be set to True.
            atmosphere: Alternative atmosphere object, if one is desired when
                computing stagnation temperature and pressure ratios.

        Returns:
            The thrust developed by the propulsion system, in Newtons. If
            norm = True, this is just the ratio of available thrust to a static
            sea-level performance datum.

        Notes:
            Since this is an interpretation of a modern high bypass ratio
            turbofan engine, there is no thrust augmentation to speak of.

        """
        return cls.thrust(mach, altitude=0.0, **kwargs)


class TurbofanLoBPR:
    """
    Performance deck for a low bypass ratio, mixed flow turbofan.

    References:
    -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
       2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.

    """
    name = "generic:TurbofanLoBPR"
    type = "turbofan"

    @classmethod
    def thrust(cls, mach, altitude, *, norm: bool = True, **kwargs):
        """
        Return the thrust available at the given flight conditions.

        Args:
            mach: Flight Mach number.
            altitude: Flight level (above mean sea level), in metres.
            norm: For generic propulsion system types, this must be set to True.

        Keyword Args:
            atmosphere: Alternative atmosphere object, if one is desired when
                computing stagnation temperature and pressure ratios.

        Returns:
            The thrust developed by the propulsion system, in Newtons. If
            norm = True, this is just the ratio of available thrust to a static
            sea-level performance datum.

        """
        # Recast as necessary
        mach = recastasnpfloatarray(mach)
        altitude = recastasnpfloatarray(altitude)

        if norm is not True:
            errormsg = f"{cls.name} must use norm=True, found that {norm=}"
            raise RuntimeError(errormsg)

        theta0, delta0 = TP0ratio(mach, altitude, kwargs.get("atmosphere"))
        # Assume theta break (throttle ratio) of 1.05
        TR = 1.05

        lapse = 0.6 * delta0
        slice = theta0 > TR
        lapse[slice] -= (0.6 * delta0 * 3.8 * (theta0 - TR) / theta0)[slice]

        # Reframe the equation so sea-level static dry thrust is 1.0
        lapse *= 1 / 0.6

        return np.clip(lapse, 0, None)

    @classmethod
    def thrust_slto(cls, mach, *, norm: bool = True, **kwargs):
        """
        Return the takeoff (wet) thrust at sea level.

        Args:
            mach: Freestream Mach number.
            norm: For generic propulsion system types, this must be set to True.

        Keyword Args:
            atmosphere: Alternative atmosphere object, if one is desired when
                computing stagnation temperature and pressure ratios.

        Returns:
            The thrust developed by the propulsion system, in Newtons. If
            norm = True, this is just the ratio of available thrust to a static
            sea-level performance datum.

        """
        # Recast as necessary
        mach = recastasnpfloatarray(mach)

        if norm is not True:
            errormsg = f"{cls.name} must use norm=True, found that {norm=}"
            raise RuntimeError(errormsg)

        theta0, delta0 = TP0ratio(mach, np.zeros(1), kwargs.get("atmosphere"))
        # Assume theta break (throttle ratio) of 1.05
        TR = 1.05

        lapse = delta0
        slice = theta0 > TR
        lapse[slice] -= (delta0 * 3.5 * (theta0 - TR) / theta0)[slice]

        # Reframe the equation so sea-level static dry thrust is 1.0
        lapse *= 1 / 0.6

        return np.clip(lapse, 0, None)


class Turbojet:
    """
    Performance deck for a turbojet.

    References:
        -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
            2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.

    """
    name = "generic:Turbojet"
    type = "turbojet"

    @classmethod
    def thrust(cls, mach, altitude, *, norm: bool = True, **kwargs):
        """
        Return the thrust available at the given flight conditions.

        Args:
            mach: Flight Mach number.
            altitude: Flight level (above mean sea level), in metres.
            norm: For generic propulsion system types, this must be set to True.

        Keyword Args:
            atmosphere: Alternative atmosphere object, if one is desired when
                computing stagnation temperature and pressure ratios.

        Returns:
            The thrust developed by the propulsion system, in Newtons. If
            norm = True, this is just the ratio of available thrust to a static
            sea-level performance datum.

        """
        # Recast as necessary
        mach = recastasnpfloatarray(mach)
        altitude = recastasnpfloatarray(altitude)

        if norm is not True:
            errormsg = f"{cls.name} must use norm=True, found that {norm=}"
            raise RuntimeError(errormsg)

        theta0, delta0 = TP0ratio(mach, altitude, kwargs.get("atmosphere"))
        # Assume theta break (throttle ratio) of 1.05
        TR = 1.05

        lapse = 0.8 * delta0 * (1 - 0.16 * mach ** 0.5)
        slice = theta0 > TR
        lapse[slice] -= (
                0.8 * delta0 * 24 * (theta0 - TR) / (9 + mach) / theta0)[slice]

        # Reframe the equation so sea-level static dry thrust is 1.0
        lapse *= 1 / 0.8

        return np.clip(lapse, 0, None)

    @classmethod
    def thrust_slto(cls, mach, *, norm: bool = True, **kwargs):
        """
        Return the takeoff (wet) thrust at sea level.

        Args:
            mach: Freestream Mach number.
            norm: For generic propulsion system types, this must be set to True.

        Keyword Args:
            atmosphere: Alternative atmosphere object, if one is desired when
                computing stagnation temperature and pressure ratios.

        Returns:
            The thrust developed by the propulsion system, in Newtons. If
            norm = True, this is just the ratio of available thrust to a static
            sea-level performance datum.

        """
        # Recast as necessary
        mach = recastasnpfloatarray(mach)

        if norm is not True:
            errormsg = f"{cls.name} must use norm=True, found that {norm=}"
            raise RuntimeError(errormsg)

        theta0, delta0 = TP0ratio(mach, np.zeros(1), kwargs.get("atmosphere"))
        # Assume theta break (throttle ratio) of 1.05
        TR = 1.05

        lapse = delta0 * (1 - 0.3 * (theta0 - 1) - 0.1 * mach ** 0.5)
        slice = theta0 > TR
        lapse[slice] -= (delta0 * 1.5 * (theta0 - TR) / theta0)[slice]

        # Reframe the equation so sea-level static dry thrust is 1.0
        lapse *= 1 / 0.8

        return np.clip(lapse, 0, None)


class Turboprop:
    """
    Performance deck for a turboprop engine.

    References:
        -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
            2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.

    """
    name = "generic:Turboprop"
    type = "turboprop"

    @classmethod
    def thrust(cls, mach, altitude, *, norm: bool = True, **kwargs):
        """
        Return the thrust available at the given flight conditions.

        Args:
            mach: Flight Mach number.
            altitude: Flight level (above mean sea level), in metres.
            norm: For generic propulsion system types, this must be set to True.

        Keyword Args:
            atmosphere: Alternative atmosphere object, if one is desired when
                computing stagnation temperature and pressure ratios.

        Returns:
            The thrust developed by the propulsion system, in Newtons. If
            norm = True, this is just the ratio of available thrust to a static
            sea-level performance datum.

        """
        # Recast as necessary
        mach = recastasnpfloatarray(mach)
        altitude = recastasnpfloatarray(altitude)

        if norm is not True:
            errormsg = f"{cls.name} must use norm=True, found that {norm=}"
            raise RuntimeError(errormsg)

        theta0, delta0 = TP0ratio(mach, altitude, kwargs.get("atmosphere"))
        # Assume theta break (throttle ratio) of 1.05
        TR = 1.05

        # I think Mattingly makes a mistake below in writing M-1 instead of M-.1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lapse = delta0 * (1 - 0.96 * (mach - 0.1) ** 0.25)
            slice = theta0 > TR
            lapse[slice] -= \
                (delta0 * 3 * (theta0 - TR) / 8.13 / (mach - 0.1))[slice]
            lapse[mach <= 0.1] = delta0[mach <= 0.1]

        return np.clip(lapse, 0, None)

    @classmethod
    def thrust_slto(cls, mach, **kwargs):
        """
        Return the takeoff thrust at sea level.

        Args:
            mach: Freestream Mach number.

        Keyword Args:
            norm: For generic propulsion system types, this must be set to True.
            atmosphere: Alternative atmosphere object, if one is desired when
                computing stagnation temperature and pressure ratios.

        Returns:
            The thrust developed by the propulsion system, in Newtons. If
            norm = True, this is just the ratio of available thrust to a static
            sea-level performance datum.

        Notes:
            Since this is an interpretation of a modern turboprop engine, there
            is no thrust augmentation to speak of.

        """
        return cls.thrust(mach, altitude=0.0, **kwargs)


class Piston:
    pass


def build_deck(deck: EngineDeck) -> None:
    """
    Given an EngineDeck object for a real engine with data, use said data to
    create interpolation engines for basic performance attributes like thrust
    and power (where necessary).

    Args:
        deck: An EngineDeck object with associated performance dataframes.

    Returns:
        None.

    """
    # Start with generic methods that apply to all deck types

    # Generic "Deck" sheets should "unpack" themselves
    if "Deck" in deck.dataframes:
        df = deck.dataframes["Deck"]
        basecols = ["Mach Number", "Altitude [m]"]
        if all(x in df for x in basecols):
            notbasecols = [x for x in df.columns.to_list() if x not in basecols]

            if "Thrust [N]" in notbasecols:
                deck.dataframes["Thrust"] \
                    = pd.DataFrame(df[basecols + ["Thrust [N]"]])
            if "Power [W]" in notbasecols:
                deck.dataframes["Power"] \
                    = pd.DataFrame(df[basecols + ["Power [W]"]])

    # Build Thrust = f(Mach, Altitude)
    if "Thrust" in deck.dataframes:
        df = deck.dataframes["Thrust"]
        if all(x in df for x in ["Thrust [N]", "Mach Number", "Altitude [m]"]):
            deck._f_thrust = xyz_interpolator(
                df["Mach Number"].to_numpy(),
                df["Altitude [m]"].to_numpy(),
                df["Thrust [N]"].to_numpy()
            )

    # If the engine data contains augmented Thrust at takeoff...
    if "Thrust SL-TO" in deck.dataframes:
        df = deck.dataframes["Thrust SL-TO"]
        if all(x in df for x in ["Thrust [N]", "Mach Number"]):
            deck._f_thrust_SLTO = xyz_interpolator(
                df["Mach Number"].to_numpy(),
                None,
                df["Thrust [N]"].to_numpy()
            )

    # Build Power = f(Mach, Altitude)
    if "Power" in deck.dataframes:
        df = deck.dataframes["Power"]
        if all(x in df for x in ["Power [W]", "Mach Number", "Altitude [m]"]):
            deck._f_shaftpower = xyz_interpolator(
                df["Mach Number"].to_numpy(),
                df["Altitude [m]"].to_numpy(),
                df["Power [W]"].to_numpy()
            )
        elif all(x in df for x in ["Power [W]", "Speed [RPM]", "Altitude [m]"]):

            # Build an interpolator based constant operating speed
            ansatz_speed = np.median(df["Speed [RPM]"])
            interp = xyz_interpolator(
                df["Speed [RPM]"].to_numpy(),
                df["Altitude [m]"].to_numpy(),
                df["Power [W]"].to_numpy()
            )

            def f_speed(mach, altitude):
                """Shaft operating speed as a function of Mach and altitude."""
                return ansatz_speed * np.ones((mach * altitude).shape)

            def f_power(mach, altitude, speed=None):
                """Shaft power as a function of Mach and altitude."""
                # If speed undefined, use ansatz estimate of operating speed
                if speed is None:
                    speed = getattr(deck, "_f_shaftspeed")(mach, altitude)
                power = interp(speed, altitude)
                return power

            deck._f_shaftspeed = f_speed
            deck._f_shaftpower = f_power

    # Deck-specific processing
    if deck.type == "electric":
        raise NotImplementedError("Sorry, can't do this yet! Missing methods..")
    elif deck.type == "piston":
        pass
    elif deck.type == "turbofan":
        pass
    elif deck.type == "turbojet":
        pass
    elif deck.type == "turboprop":
        # The "Thrust" thus far is core thrust. Real thrust will use shaftpower.
        deck._f_thrust_core = getattr(deck, "_f_thrust")

        def f_eta_propeller(mach, eta_max=0.88):
            """Variable pitch propeller efficiency as a function of Mach."""
            # Use J.D.Mattingly propeller model
            eta_prop = np.zeros_like(mach) * np.nan
            slice0 = mach <= 0.85
            slice1 = mach <= 0.70
            slice2 = (0 <= mach) & (mach <= 0.1)
            eta_prop[slice0] = eta_max * (1 - (mach[slice0] - 0.7) / 3)
            eta_prop[slice1] = eta_max
            eta_prop[slice2] = eta_max * 10 * mach[slice2]
            return eta_prop

        def f_thrust(mach, altitude):
            """Thrust as a function of Mach and altitude."""
            mach = np.clip(mach, 1e-5, None)  # Avoid divide by zero later

            # Core thrust
            thrust_core = getattr(deck, "_f_thrust_core")(mach, altitude)

            # Cold thrust
            eta = f_eta_propeller(mach)
            shaftpower = getattr(deck, "_f_shaftpower")(mach, altitude)
            soundspeed = isaref.vsound_mps(altitude)
            airspeed = mach * soundspeed
            thrust_cold = eta * shaftpower / airspeed

            return thrust_core + thrust_cold

        deck._f_thrust = f_thrust

    return None


def xyz_interpolator(x: np.ndarray, y: typing.Union[np.ndarray, None],
                     z: np.ndarray):
    """
    Interpolates 2D unstructured data for z = f(x, y), lower bounded by (0, 0).

    Args:
        x: Array of positive input coordinates (x >= 0).
        y: Array of positive input coordinates (y >= 0).
        z: Array of output values.

    Returns:
        z = z_interpolator(x, y)

    """
    # Recast as necessary
    if y is None:
        x = np.clip(x, 0, None)
        _interp = interp1d(
            x, z, kind="cubic", bounds_error=False, fill_value=np.nan)

        def interp(x, y):
            """Helper function, to ignore y for one dimensional z=f(x)."""
            return _interp(x)
    else:
        xy = np.clip(np.vstack([x, y]), 0, None)  # x, y > 0
        interp = CloughTocher2DInterpolator(xy.T, z)

    return interp
