"""
This module contains tools for converting between units commonly used in
aircraft design.
"""

__author__ = "Andras Sobester"

import scipy.constants as sc


def C_F(temp_c):
    """Convert temperature value from Celsius to Fahrenheit"""
    return temp_c * 9 / 5 + 32


def ft_m(length_feet):
    """Converts length value from feet to meters"""
    return length_feet * 0.3048


def ft2_m2(area_ft2):
    """Converts area value from feet squared to meters squared"""
    return area_ft2 / 10.7639


def m2_ft2(area_m2):
    """Converts area value from meters squared to feet squared"""
    return area_m2 * 10.7639


def m_km(length_m):
    """Converts length value from meters to kilometres"""
    return length_m / 1000.0


def km_m(length_km):
    """Converts length value from kilometres to metres"""
    return length_km * 1000.0


def C_K(temp_c):
    """Convert temperature value from Celsius to Kelvin"""
    return temp_c + 273.15


def K_C(temp_k):
    """Convert temperature value from Kelvin to Celsius"""
    return temp_k - 273.15


def C_R(temp_c):
    """Convert temperature value from Celsius to Rankine"""
    return (temp_c + 273.15) * 9 / 5


def R_C(temp_r):
    """Convert temperature value from Rankine to Celsius"""
    return (temp_r - 491.67) * 5 / 9


def K_R(temp_k):
    """Convert temperature value from Kelvin to Rankine"""
    return temp_k * 9 / 5


def R_K(temp_r):
    """Convert temperature value from Rankine to Kelvin"""
    return temp_r * 5 / 9


def Pa_mbar(press_pa):
    """Convert pressure value from Pascal to mbar"""
    return press_pa * 0.01


def mbar_Pa(press_mbar):
    """Convert pressure value from mbar to Pascal"""
    return press_mbar / 0.01


def inHg_mbar(press_inhg):
    """Convert pressure value from inHg to mbar"""
    return press_inhg * 33.8639


def mbar_inHg(press_mbar):
    """Convert pressure value from mbar to inHg"""
    return press_mbar / 33.8639


def mbar_lbfft2(press_mbar):
    """Convert pressure value from mbar to lb/ft^2"""
    return press_mbar * 2.08854


def lbfft2_mbar(press_lbfft2):
    """Convert pressure value from lb/ft^2 to mbar"""
    return press_lbfft2 / 2.08854


def mps_kts(speed_mps):
    """Convert speed value from m/s to knots"""
    return speed_mps * 1.9438445


def kts_mps(speed_kts):
    """Convert speed value knots to mps"""
    return speed_kts * 0.5144444


def m_ft(length_m):
    """Convert length value from meters to feet"""
    return length_m / 0.3048


def Pa_kgm2(pressure_pa):
    """Convert pressure value from Pa to kg/m^2"""
    return pressure_pa * 0.1019716212978


def kg_N(mass_kg):
    """Converts mass in kg to weight in N"""
    return mass_kg * sc.g


def N_kg(force_n):
    """Converts force in N to mass in kg"""
    return force_n / sc.g


def kgm2_Pa(pressure_kgm2):
    """Convert pressure value from kg/m^2 to Pa"""
    return pressure_kgm2 * sc.g


def fpm_mps(speed_fpm):
    """Convert speed value from feet/min to m/s"""
    mpm = ft_m(speed_fpm)
    return mpm / 60.0


def lb_kg(mass_lbs):
    """Convert mass value from lbs to kg"""
    return mass_lbs * 0.453592


def kg_lb(mass_kg):
    """Convert mass value from kg to lbs"""
    return mass_kg / 0.453592


def kN_N(force_kn):
    """Convert force from kN to N"""
    return force_kn * 1e3


def N_kN(force_n):
    """Convert force from N to kN"""
    return force_n / 1e3


def lbf_N(force_lbf):
    """Convert force from lbf to N"""
    return force_lbf / 0.224809


def N_lbf(force_n):
    """Convert force from N to lbf"""
    return force_n * 0.224809


def lbf_kN(force_lbf):
    """Convert force from lbf to kN"""
    return lbf_N(force_lbf) / 1e3


def kN_lbf(force_n):
    """Convert force from kN to lbf"""
    return N_lbf(force_n) * 1e3


def WN_Wkg(powertoweight_wn):
    """Convert power to weight (Watt/N) to W/kg"""
    return powertoweight_wn * sc.g


mps_Wkg = WN_Wkg


def WN_kWkg(powertoweight_wn):
    """Convert power to weight (Watt/N) to kW/kg"""
    return WN_Wkg(powertoweight_wn) / 1000.0


mps_kWkg = WN_kWkg


def WN_hpkg(powertoweight_wn):
    """Convert power to weight (Watt/N) to hp/kg"""
    return WN_kWkg(powertoweight_wn) * 1.34102


mps_hpkg = WN_hpkg


def W_kW(power_w):
    """Convert power from watts to kilowatts"""
    return power_w / 1e3


def kW_W(power_kw):
    """Convert power from kilowatts to watts"""
    return power_kw * 1e3


def kW_hp(power_kw):
    """Convert power from kW to HP"""
    return power_kw * 1.34102


def hp_kW(power_hp):
    """Convert power from HP to kW"""
    return power_hp / 1.34102


def W_hp(power_w):
    """Convert power from W to HP"""
    return kW_hp(power_w / 1e3)


def hp_W(power_hp):
    """Convert power from HP to kW"""
    return hp_kW(power_hp) * 1e3


def kgm3_sft3(density_kgm3):
    """Convert density from kg/m^3 to slugs/ft^3"""
    return density_kgm3 * 0.00194032


def sft3_kgm3(density_slft3):
    """Convert density from slugs/ft^3 to kg/m^3"""
    return density_slft3 / 0.00194032


def tas2eas(tas, localairdensity_kgm3):
    """Convert True Air Speed to Equivalent Air Speed"""
    return tas * ((localairdensity_kgm3 / 1.225) ** 0.5)


def eas2tas(eas, localairdensity_kgm3):
    """Convert True Air Speed to Equivalent Air Speed"""
    return eas / ((localairdensity_kgm3 / 1.225) ** 0.5)


def Pa_lbfft2(pressure_pa):
    """Convert pressure from Pascal to lbf(pound-force)/ft^2"""
    return pressure_pa * 0.020885434273039


def lbfft2_Pa(pressure_lbft2):
    """Convert pressure from lbf(pound-force)/ft^2 to Pascal"""
    return pressure_lbft2 / 0.020885434273039


def RPM_rads(rotationalspeed_rpm):
    """Convert rotational speed from RPM to angular speed in radians/second"""
    return rotationalspeed_rpm * (2 * sc.pi) / 60


def rads_RPM(angularspeed_radps):
    """Convert angular speed from radians/second to rotational speed in RPM"""
    return angularspeed_radps * 60 / (2 * sc.pi)
