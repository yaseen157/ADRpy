"""
Module for modelling the performance of engines, using "engine performance
decks" - derived from real engine data.
"""
import os
import re
import typing

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CloughTocher2DInterpolator

from ADRpy import atmospheres as at
from ADRpy.mtools4acdc import recastasnpfloatarray

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

    def __init__(self, enginename: str):

        if enginename not in engine_catalogue:
            errormsg = (
                f"'{enginename}' not found in list of available engines: "
                f"{list(engine_catalogue.keys())}")
            raise ValueError(errormsg)

        # Check if the engine has data cached already, otherwise load it
        if engine_catalogue[enginename]["dataframes"] == {}:
            engine_catalogue[enginename]["dataframes"] = pd.read_excel(
                engine_catalogue[enginename]["path"], sheet_name=None)

        self.name = enginename
        self.type = engine_catalogue[enginename]["type"]
        self._dataframes = engine_catalogue[enginename]["dataframes"]

        build_deck(self)
        return

    def __repr__(self):
        reprstr = f"{type(self).__name__}('{self.name}')"
        return reprstr

    @property
    def dataframes(self):
        """The raw dataframes extracted from engine data files."""
        return self._dataframes

    def thrust(self, mach, altitude, norm: bool = None):
        """
        Return the thrust available at the given flight conditions.

        Args:
            mach: Flight Mach number.
            altitude: Flight level (above mean sea level), in metres.
            norm: Flag, whether to normalise the output to the sea-level static
                thrust. Optional, defaults to False (no normalisation).

        Returns:
            The thrust developed by the propulsion system, in Newtons.

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

    def thrust_slto(self, mach, norm: bool = None):
        """
        Return the (augmented) thrust during takeoff, at sea level.

        Args:
            mach: Freestream Mach number.
            norm: Flag, whether to normalise the output to the sea-level static
                thrust. Optional, defaults to False (no normalisation).

        Returns:
            The thrust developed by the propulsion system, in Newtons.

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


def build_deck(deck: EngineDeck) -> None:
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
