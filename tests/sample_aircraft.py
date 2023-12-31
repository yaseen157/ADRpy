import numpy as np
import matplotlib.pyplot as plt
from ADRpy import unitconversions as co
from ADRpy import constraintanalysis as ca
from ADRpy import atmospheres as at


def Cirrus_SR22():
    """
    Returns an AircraftConcept object of the Cirrus SR22.

    References:
        -   http://servicecenters.cirrusdesign.com/tech_pubs/SR2X/pdf/POH/SR22-004/pdf/Online13772-004.pdf
        -   Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods
            and Procedures", Butterworth-Heinemann, 2013.
    """
    # The POH does not recommend exceeding 60 degrees of bank
    nturn = 1 / np.cos(
        np.radians(60))  # Approx. load factor for 60 degrees of bank

    # There are two stall speeds for MTOW clean-config, depending on CG
    VS_CG_FWD_kcas = 73
    VS_CG_AFT_kcas = 70
    # Clearly we are looking for limiting constraints, so take the minimum
    vstall_kcas = min(VS_CG_FWD_kcas, VS_CG_AFT_kcas)

    designbrief = {
        # Take-off
        "rwyelevation_m": 0.0, "groundrun_m": co.feet2m(1_082),
        # Sustained turn
        "turnalt_m": 0, "turnspeed_ktas": 101, "stloadfactor": nturn,
        # Climb
        "climbalt_m": 0, "climbspeed_kias": 108, 'climbrate_fpm': 1_251,
        # Cruise
        "cruisealt_m": co.feet2m(10e3), "cruisespeed_ktas": 182,
        # Service Ceiling
        "servceil_m": co.feet2m(20e3), "secclimbspd_kias": 90,
        # Stall, 1g
        "vstallclean_kcas": vstall_kcas
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
