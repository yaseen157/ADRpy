import numpy as np
import matplotlib.pyplot as plt
from ADRpy import unitconversions as co
from ADRpy import constraintanalysis as ca
from ADRpy import atmospheres as at


def Cirrus_SR22():
    """Returns an AircraftConcept object of the Cirrus SR22."""
    nturn = 1 / np.cos(
        np.radians(60))  # Approx. load factor for 60 degrees of bank

    designbrief = {
        "rwyelevation_m": 0.0, "groundrun_m": co.feet2m(1_082),  # Take-off
        "turnalt_m": 0, "turnspeed_ktas": 101, "stloadfactor": nturn,
        # Sustained turn
        "climbalt_m": 0, "climbspeed_kias": 108, 'climbrate_fpm': 1_251,
        # Climb
        "cruisealt_m": co.feet2m(10e3), "cruisespeed_ktas": 182,  # Cruise
        "servceil_m": co.feet2m(20_125), "secclimbspd_kias": 95,
        # Service Ceiling
        "vstallclean_kcas": 73  # Stall
    }

    designdefinition = {
        "aspectratio": 10.12, "taperratio": 0.5,  # Wing "slenderness"
        "weight_n": co.lbf2n(3400),  # Weight estimate
        "weightfractions": {"cruise": 3400 / 3600}  # Weight fractions
    }

    designperformance = {
        "CDmin": 0.02541, "CLmax": 1.41, "CLminD": 0.20, "CL0": 0.0,
        # General performance
        "mu_R": 0.04, "CLTO": 0.590, "CDTO": 0.0414, "CLmaxTO": 1.69
        # Take-off specific performance
    }

    designatm = at.Atmosphere()

    designpropulsion = "Piston"

    concept = ca.AircraftConcept(
        brief=designbrief,
        design=designdefinition,
        performance=designperformance,
        atmosphere=designatm,
        propulsion=designpropulsion
    )

    return concept
