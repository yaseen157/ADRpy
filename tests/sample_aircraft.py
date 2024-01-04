import numpy as np
import matplotlib.pyplot as plt
from ADRpy import unitconversions as uc
from ADRpy import constraintanalysis as ca
from ADRpy import atmospheres as at
from ADRpy import propulsion as pdecks


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

    # There are two stall speeds for MTOW clean-config, depending on CG location
    # vstall_CG_FWD_kcas = 73
    # vstall_CG_AFT_kcas = 70
    # Use the above to guess a representative speed at regular CG position
    vstall_kcas = 72.12

    designbrief = {
        # Take-off
        "rwyelevation_m": 0.0, "groundrun_m": uc.ft_m(1_082),
        # Sustained turn
        "turnalt_m": 0, "turnspeed_ktas": 108, "stloadfactor": nturn,
        # Climb
        "climbalt_m": 0, "climbspeed_kias": 108, 'climbrate_fpm': 1_251,
        # Cruise
        "cruisealt_m": uc.ft_m(10e3), "cruisespeed_ktas": 182,
        # Service Ceiling
        "servceil_m": uc.ft_m(20e3), "secclimbspd_kias": 90,
        # Stall, 1g
        "vstallclean_kcas": vstall_kcas
    }

    designdefinition = {
        "aspectratio": 10.12, "taperratio": 0.5,  # Wing "slenderness"
        "weight_n": uc.lbf_N(3600),  # Weight estimate
        "weightfractions": {"cruise": 3400 / 3600}  # Weight fractions
    }

    designperformance = {
        # General performance
        "CDmin": 0.02541, "CLmax": 1.41, "CLminD": 0.20, "CL0": 0.0,
        # Take-off specific performance
        "mu_R": 0.04, "CLTO": 0.590, "CLmaxTO": 1.69
    }
    eta_prop = {
        "cruise": 0.88, "turn": 0.85, "climb": 0.71,
        "servceil": 0.65, "take-off": 0.45
    }
    designperformance.update({"eta_prop": eta_prop})

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


def gudmundsson_Vn():
    designbrief = {"vstallclean_kcas": 46.3}
    b_ft, S_ft2 = 38, 130

    designdef = {
        "aspectratio": b_ft ** 2 / S_ft2,
        "weight_n": uc.lbf_N(1320)
    }

    designperf = {
        "CLalpha": 5.25, "CLmax": 1.45, "CLmin": -1.00,
        #     "CLmaxHL": 2.10, "CLminHL": -0.75  # <-- Excluded in the example!
    }

    designatm = at.Atmosphere()

    designpropulsion = "Piston"

    concept = ca.AircraftConcept(
        brief=designbrief,
        design=designdef,
        performance=designperf,
        atmosphere=designatm,
        propulsion=designpropulsion
    )
    return concept


def keane_smallUAV():
    """Keane, Sobester and Scanlan small UAV"""
    designbrief = {
        'rwyelevation_m': 0, 'groundrun_m': 60,  # <- Take-off requirements
        'stloadfactor': 1.41, 'turnalt_m': 0, 'turnspeed_ktas': 40,
        # <- Turn requirements
        'climbalt_m': 0, 'climbspeed_kias': 46.4, 'climbrate_fpm': 591,
        # <- Climb requirements
        'cruisealt_m': 122, 'cruisespeed_ktas': 58.3, 'cruisethrustfact': 1.0,
        # <- Cruise requirements
        'servceil_m': 152, 'secclimbspd_kias': 40,
        # <- Service ceiling requirements
        'vstallclean_kcas': 26.4  # <- Required clean stall speed
    }
    TOW_kg = 15.0
    wfract = {'turn': 1.0, 'climb': 1.0, 'cruise': 1.0, 'servceil': 1.0}
    designdefinition = {
        'aspectratio': 9.0, 'sweep_le_deg': 2, 'sweep_mt_deg': 0,
        # <- Wing geometry
        'weightfractions': wfract, "weight_n": uc.kg_N(TOW_kg)
        # <- Weight definitions
    }
    etas = {k: 0.6 for k in ["take-off", "climb", "cruise", "turn", "servceil"]}

    designperformance = {
        'CLTO': 0.97, 'CLmaxTO': 1.7, 'mu_R': 0.17,
        # Take-off specific performance
        'CLmax': 1.0, 'CDmin': 0.0418, 'eta_prop': etas  # General performance
    }

    designatm = at.Atmosphere()

    concept = ca.AircraftConcept(designbrief, designdefinition,
                                 designperformance, designatm, "Piston")

    return concept


if __name__ == "__main__":
    sampleac = Cirrus_SR22()
    pass
