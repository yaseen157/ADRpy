"""
This module contains tools for defining the environment in which
aircraft performance analyses, trade-off studies, conceptual sizing
and other aircraft engineering calculations can be carried out.

"""
from numbers import Number
import warnings
import csv
import os
import numpy as np
from scipy import interpolate
from ADRpy import unitconversions as co
from ADRpy import mtools4acdc as mtools

__author__ = "Andras Sobester"

# Specific gas constant for dry air
# (in Joules ) per kilogram per Kelvin
R_JPKGPK = 287.05287
# Radius of Earth at 45deg latitude, adopted in ISA for
# the calculation of geopotential height
R_EARTH_M = 6356766
# Dry air ratio of specific heats
GAMMA_DRY_AIR = 1.401


def idealgasdens_kgm3(p_pa, temp_k):
    """
    Computes density from pressure and temperature, using ideal gas assumptions.

    Args:
        p_pa: The pressure, in Pascal.
        temp_k: The temperature, in Kelvin.

    Returns:
        Density, in kilograms per metre cubed.

    """
    """Density from pressure and temperature, on ideal gas assumption"""
    return p_pa / R_JPKGPK / temp_k


def idealgaspress_pa(rho_kgpm3, temp_k):
    """
    Computes pressure from density and temperature, using ideal gas assumptions.

    Args:
        rho_kgpm3: The density, in kilograms per metre cubed.
        temp_k: The temperature, in Kelvin.

    Returns:
        Gas pressure, in Pascal.

    """
    return R_JPKGPK * rho_kgpm3 * temp_k


def geom2geop45m(altitude_m):
    """
    Converts geometric height to geopotential (m) assuming 45deg latitude.

    Args:
        altitude_m: The geometric altitude above mean sea level.

    Returns:
        The geopotential altitude above mean sea level.

    """
    return R_EARTH_M * altitude_m / (R_EARTH_M + altitude_m)


def geop2geom45m(altitude_m):
    """
    Converts geopotential height to geometric (m) assuming 45deg latitude.

    Args:
        altitude_m: The geopotential altitude above mean sea level.

    Returns:
        The geometric altitude above mean sea level.

    """
    return R_EARTH_M * altitude_m / (R_EARTH_M - altitude_m)


class Runway:
    """
    Runway model to be used for take-off/landing performance calculations.

    Examples:

        Creating and querying a Runway object:

        >>> from ADRpy import atmospheres as at
        >>> runway = at.Runway('EGLL', 0)

        >>> print(f"Runway: {runway.le_ident} / {runway.he_ident}\\n"
        ...       f"True headings: {runway.le_heading_degt} / "
        ...       f"{runway.he_heading_degt} degrees\\n"
        ...       f"Elevation (low end): {runway.le_elevation_ft} ft\\n"
        ...       f"Length: {runway.length_ft} ft")
        Runway: 09L / 27R
        True headings: 89.6 / 269.6 degrees
        Elevation (low end): 79.0 ft
        Length: 12799.0 ft

    """

    def __init__(self, icao_code: str = None, rwyno: int = 0,
                 elevation_ft=0, heading=0, surf='ASP',
                 length_ft=10000, width_ft=100):
        """
        Args:
            icao_code: International Civil Aviation Organisation code of the
                airport. Required if the user wishes to equip this object with
                the attributes of a specific, existing runway, e.g., 'EGLL'
                (London Heathrow airport). Runway data is obtained from an
                off-line image of the `ourairports.com` database.
            rwyno: If an icao_code is provided, use to select a runway.
            elevation_ft: For custom runways (icao_code=None), prescribe runway
                elevation. Optional, defaults to 0.
            heading: For custom runways (icao_code=None), prescribe runway
                heading. Optional, defaults to 0 (due North).
            surf: For custom runways (icao_code=None), prescribe the runway
                surface type. Optional, defaults to "ASP" (asphalt).
            length_ft: For custom runways (icao_code=None), prescribe the runway
                length in feet. Optional, defaults to 10,000.
            width_ft: For custom runways (icao_code=None), prescribe the runway
                width in feet. Optional, defaults to 100.

        Raises:
            ValueError: Thrown if 'rwyno' parameter exceeds the number of
                runways at the airport specified by the `icao_code`. The number
                of runways can be found in the `nrways` attribute of the runway
                object:

                >>> runway = Runway('KDEN')
                >>> runway.nrways
                6

        """
        if icao_code:
            # Read relevant runway data from the ourairports.com database
            # le_... / he_... - numbers referring to the low/high end respectively
            rwy_file = os.path.join(os.path.dirname(__file__), "data",
                                    "runways.csv")
            with open(rwy_file, newline='') as rwyfile:
                runwaydata = csv.reader(rwyfile, delimiter=',')
                runwaylist = []
                for row in runwaydata:
                    runwaylist.append(row)
            rindlst = [i for i, rwy in enumerate(runwaylist) if
                       rwy[2] == icao_code]
            self.nrways = len(rindlst)
            if rwyno > self.nrways - 1:
                print('Requested rwy. nr. exceeds the nr. of runways at ',
                      icao_code, '(', self.nrways, ')')
                raise ValueError('Incorrect runway number.')
            rind = rindlst[0]
            self.ident = runwaylist[rind + rwyno][0]
            self.airport_ref = runwaylist[rind + rwyno][1]
            self.airport_ident = runwaylist[rind + rwyno][2]
            self.length_ft = float(runwaylist[rind + rwyno][3])
            self.width_ft = float(runwaylist[rind + rwyno][4])
            self.surface = runwaylist[rind + rwyno][5]
            self.lighted = runwaylist[rind + rwyno][6]
            self.closed = runwaylist[rind + rwyno][7]
            self.le_ident = runwaylist[rind + rwyno][8]
            self.le_latitude_deg = runwaylist[rind + rwyno][9]
            self.le_longitude_deg = runwaylist[rind + rwyno][10]
            self.le_elevation_ft = float(runwaylist[rind + rwyno][11])
            self.le_heading_degt = float(runwaylist[rind + rwyno][12])
            try:
                self.le_displaced_threshold_ft = float(
                    runwaylist[rind + rwyno][13])
            except ValueError:
                self.le_displaced_threshold_ft = float('nan')
            self.he_ident = runwaylist[rind + rwyno][14]
            self.he_latitude_deg = runwaylist[rind + rwyno][15]
            self.he_longitude_deg = runwaylist[rind + rwyno][16]
            self.he_elevation_ft = float(runwaylist[rind + rwyno][17])
            self.he_heading_degt = float(runwaylist[rind + rwyno][18])
            try:
                self.he_displaced_threshold_ft = float(
                    runwaylist[rind + rwyno][19])
            except ValueError:
                self.he_displaced_threshold_ft = float('nan')
        else:
            # Define a custom runway based on the data provided
            self.ident = '0000000'
            self.airport_ref = '0000'
            self.airport_ident = 'CUST'
            self.length_ft = length_ft
            self.width_ft = width_ft
            self.surface = surf
            self.lighted = 1
            self.closed = 0
            self.le_ident = str(int(heading / 10))
            self.le_latitude_deg = 0
            self.le_longitude_deg = 0
            self.le_elevation_ft = elevation_ft
            self.le_heading_degt = heading
            self.le_displaced_threshold_ft = 0
            self.he_ident = str(int(reciprocalhdg(heading) / 10))
            self.he_latitude_deg = 0
            self.he_longitude_deg = 0
            self.he_elevation_ft = elevation_ft
            self.he_heading_degt = reciprocalhdg(heading)
            self.he_displaced_threshold_ft = 0

        # Metric versions of the imperial fields listed above
        self.length_m = co.ft_m(self.length_ft)
        self.width_m = co.ft_m(self.width_ft)
        self.le_elevation_m = co.ft_m(self.le_elevation_ft)
        self.le_displaced_threshold_m = co.ft_m(
            self.le_displaced_threshold_ft)
        self.he_elevation_m = co.ft_m(self.he_elevation_ft)
        self.he_displaced_threshold_m = co.ft_m(
            self.he_displaced_threshold_ft)

    def windcomponents(self, wind_dirs_deg: list[float],
                       wind_speeds: list[float]):
        """
        Resolves list of wind speeds and directions into runway/cross components
        on the current runway.

        Args:
            wind_dirs_deg: Wind directions expressed in degrees true
                (e.g. directions specified in a METAR).
            wind_speeds: Wind_speeds (in the units in which the output is
                desired).

        Returns:
            A tuple (runway_component, crosswind_component) where each item is
            a scalar or numpy array. These correspond with the runway direction
            component of the wind (sign convention: headwinds are positive),
            and the cross component of the wind (sign convention: winds from the
            right are positive).


        Examples:

            Given a METAR, calculate the wind components on Rwy 09 at Yeovilton

            >>> from ADRpy import atmospheres as at
            >>> from metar import Metar

            >>> runway = at.Runway('EGDY', 1)
            >>> egdywx = Metar.Metar('EGDY 211350Z 30017G25KT 9999 FEW028 BKN038 08/01 Q1031')

            >>> direction_deg = egdywx.wind_dir.value()
            >>> windspeed_kts = egdywx.wind_speed.value()

            >>> rwy_knots, cross_knots = runway.windcomponents(direction_deg, windspeed_kts)

            >>> print(f"Runway component: {rwy_knots}\\n"
            ...       f"Cross component: {cross_knots}")
            Runway component: -13.594639194280536
            Cross component: -10.207143830540032

        """

        speeds = mtools.recastasnpfloatarray(wind_speeds)

        # Wind speed is considered as a positive scalar
        speeds = np.abs(speeds)

        directions_deg = mtools.recastasnpfloatarray(wind_dirs_deg)

        relative_heading_rad = np.deg2rad(directions_deg - self.le_heading_degt)

        runway_component = speeds * np.cos(relative_heading_rad)  # Headwind: +
        crosswind_component = speeds * np.sin(relative_heading_rad)  # Right: +

        # Scalar output to a scalar input
        if isinstance(wind_dirs_deg, Number):
            return runway_component[0], crosswind_component[0]

        return runway_component, crosswind_component


class Obsprofile:
    """
    Observed atmosphere profile data.

    Instances of Obsprofile can be passed as the 'profile' argument of
    the Atmosphere class.
    """

    def __init__(self, alt_m, temp_k=None, rho_kgpm3=None, p_pa=None):
        """
        Args:
            alt_m: Altitudes at which observations were made.
            temp_k: Observed temperatures, in Kelvin. Optional.
            rho_kgpm3: Observed densities, in kilograms per metre cubed.
                Optional.
            p_pa: Observed pressures, in Pascal. Optional.

        Notes:
             If either of pressure or density arrays are missing from the
             arguments, the missing array of data can be calculated from
             ideal gas laws.

        """
        self.alt_m = np.array(alt_m)

        self.temp_k = np.array(temp_k)
        self.rho_kgpm3 = np.array(rho_kgpm3)
        self.p_pa = np.array(p_pa)

        # Check and complete the data table and build the interp functions

        # temperature (K) and density are given against an altitude scale
        if np.size(self.alt_m) == np.size(self.temp_k) == np.size(
                self.rho_kgpm3):
            self.p_pa = R_JPKGPK * self.rho_kgpm3 * self.temp_k
        # temperature (K) and pressure are given against an altitude scale
        elif np.size(self.alt_m) == np.size(self.temp_k) == np.size(self.p_pa):
            self.rho_kgpm3 = idealgasdens_kgm3(self.p_pa, self.temp_k)
        # speed of sound
        self.vs_mps = np.sqrt(1.4 * R_JPKGPK * self.temp_k)

        # all data points ready, the interpolators can now be constructed
        self.ftemp_k = interpolate.interp1d(self.alt_m, self.temp_k)
        self.fp_pa = interpolate.interp1d(self.alt_m, self.p_pa)
        self.frho_kgpm3 = interpolate.interp1d(self.alt_m, self.rho_kgpm3)
        self.fvs_mps = interpolate.interp1d(self.alt_m, self.vs_mps)

    @property
    def loalt(self):
        """The minimum valid altitude (in m) of the interpolators."""
        return min(self.alt_m)

    @property
    def hialt(self):
        """The maximum valid altitude (in m) of the interpolators."""
        return max(self.alt_m)


class Atmosphere:
    """
    Standard or off-standard/custom atmospheres.

    Available atmosphere (profile) types include:

    -   The International Standard Atmosphere (**ISA**) model. Based on ESDU
        Data Item 77022, `"Equations for calculation of International
        Standard Atmosphere and associated off-standard atmospheres"`,
        published in 1977, amended in 2008. It covers the first  50km of
        the atmosphere.
    -   Off-standard, temperature offset versions of the above.
    -   Extremely warm/cold, and low/high density atmospheres from US
        MIL HDBK 310
    -   User-defined atmospheres based on interpolated data.

    Examples:

        >>> from ADRpy import atmospheres as at
        >>> from ADRpy import unitconversions as co

        Instantiate an atmosphere object: an off-standard ISA with a -10C offset

        >>> isa_minus10 = at.Atmosphere(offset_deg=-10)

        >>> # Query altitude
        >>> altitude_ft = 38000
        >>> altitude_m = co.ft_m(altitude_ft)

        >>> # Query the ambient density in this model at the specified altitude
        ... print(f"ISA-10C density at {str(altitude_ft)} feet (geopotential):",
        ...       isa_minus10.airdens_kgpm3(altitude_m), "kg/m^3")
        ISA-10C density at 38000 feet (geopotential): 0.3480494789991107 kg/m^3


        >>> # Query the speed of sound in this model at the specified altitude
        ... print(f"ISA-10C speed of sound at {str(altitude_ft)} feet",
        ... "(geopotential):", isa_minus10.vsound_mps(altitude_m), "m/s")
        ISA-10C speed of sound at 38000 feet (geopotential): 288.1792251702055 m/s

    Notes:
        The unit tests (found in tests/t_atmospheres.py in the GitHub
        repository) compare the atmosphere outputs against data from the 1976 US
        Standard Atmosphere, NASA-TM-X-74335. ESDU 77022 describes its
        ISA model as being identical for all practical purposes with the US
        Standard Atmospheres.

    """

    # INTERNATIONAL STANDARD ATMOSPHERE CONSTANTS
    # _Level limits in m =======================================================
    _Level1 = 11000
    _Level2 = 20000
    _Level3 = 32000
    _Level4 = 47000
    _Level5 = 50000
    # _Level limits in terms of density (kg/m3)=================================
    _dLevel1 = 0.36391700650017816
    _dLevel2 = 0.088034556579455497
    _dLevel3 = 0.013224937668421609
    _dLevel4 = 0.0014275295197313882
    _dLevel5 = 0.00097752213943149563
    # _Level limits in terms of density (kg/m3)=================================
    _pLevel1 = 22632.000999016603
    _pLevel2 = 5474.8699475808515
    _pLevel3 = 868.01381950678511
    _pLevel4 = 110.90599734942646
    # ISA constants, base temperatures A in K, lapse rates B in deg/m
    # Based on Table 11.3, defined in terms of 'pressure height Hp', SI Units
    # Layer 1 ------------------------------------------------------------------
    _A1 = 288.15
    _B1 = -6.5e-3
    _C1 = 8.9619638
    _D1 = -0.20216125e-3
    _E1 = 5.2558797
    _I1 = 1.048840
    _J1 = -23.659414e-6
    _L1 = 4.2558797
    # Layer 2 -----------------------------------------------------------------
    _A2 = 216.65
    _B2 = 0
    _F2 = 128244.5
    _G2 = -0.15768852e-3
    _M2 = 2.0621400
    _N2 = -0.15768852e-3
    # Layer 3 -----------------------------------------------------------------
    _A3 = 196.65
    _B3 = 1e-3
    _C3 = 0.70551848
    _D3 = 3.5876861e-6
    _E3 = -34.163218
    _I3 = 0.9726309
    _J3 = 4.94600e-6
    _L3 = -35.163218
    # Layer 4 -----------------------------------------------------------------
    _A4 = 139.05
    _B4 = 2.8e-3
    _C4 = 0.34926867
    _D4 = 7.0330980e-6
    _E4 = -12.201149
    _I4 = 0.84392929
    _J4 = 16.993902e-6
    _L4 = -13.201149
    # Layer 5 -----------------------------------------------------------------
    _A5 = 270.65
    _B5 = 0
    _F5 = 41828.42
    _G5 = -0.12622656e-3
    _M5 = 0.53839563
    _N5 = -0.12622656e-3

    # ==========================================================================

    def __init__(self, offset_deg=0, profile: Obsprofile = None):

        self.offset_deg = offset_deg  # Relative temperature (C or K)
        self.profile = profile

        # If an altitude vector is not specified...
        if profile is not None:
            self.is_isa = False  # Interpolated atmosphere
        else:
            # ...then we are building an ISA
            self.is_isa = True

    def _alttest(self, reqalt_m):

        if isinstance(reqalt_m, Number):
            reqalt_m = [reqalt_m]
        # Recast as numpy floats, as there is no sensible reason for integers
        reqalt_m = np.array(reqalt_m, dtype=float)

        if self.is_isa:
            if (reqalt_m > self._Level5).any():
                print('Altitudes had to be limited to 50km where higher.')
                reqalt_m[reqalt_m > self._Level5] = self._Level5
        else:
            minalt_m = np.amin(self.profile.alt_m)
            maxalt_m = np.amax(self.profile.alt_m)
            if np.asarray(reqalt_m < minalt_m).any():
                print('Requested altitude below interpolation range.')
                reqalt_m[reqalt_m < minalt_m] = minalt_m
            if np.asarray(reqalt_m > maxalt_m).any():
                print('Requested altitude above interpolation range.')
                reqalt_m[reqalt_m > maxalt_m] = maxalt_m

        return reqalt_m

    def _isatemp_k(self, altitude_m):
        """
        ISA temperature as a function of geopotential altitude.

        Args:
            altitude_m: Geopotential altitude above mean sea level.

        Returns:
            The ISA temperature, in Kelvin.

        """
        alt_it = np.nditer([altitude_m, None])
        for alt, t_k in alt_it:
            if alt < self._Level1:
                # Troposphere, temperature decreases linearly
                t_k[...] = self._A1 + self._B1 * alt
            elif alt < self._Level2:
                # Lower stratopshere, temperature is constant
                t_k[...] = self._A2 + self._B2 * alt
            elif alt < self._Level3:
                # Upper stratopshere, temperature is increasing
                t_k[...] = self._A3 + self._B3 * alt
            elif alt < self._Level4:
                # Between 32 and 47 km
                t_k[...] = self._A4 + self._B4 * alt
            else:
                # Between 47km and 51km
                t_k[...] = self._A5 + self._B5 * alt
            # Adjust for the temperature offset
            t_k[...] = t_k[...] + self.offset_deg
        return alt_it.operands[1]

    def _isapress_pa(self, altitude_m):
        """
        ISA pressure as a function of geopotential altitude.

        Args:
            altitude_m: Geopotential altitude above mean sea level.

        Returns:
            The ISA pressure, in Pascal.

        """
        alt_it = np.nditer([altitude_m, None])
        for alt, pressure_pa in alt_it:
            if alt < self._Level1:
                # Troposphere
                pressure_pa[...] = (self._C1 + self._D1 * alt) ** self._E1
            elif alt < self._Level2:
                # Lower stratopshere
                pressure_pa[...] = self._F2 * np.exp(self._G2 * alt)
            elif alt < self._Level3:
                # Upper stratopshere
                pressure_pa[...] = (self._C3 + self._D3 * alt) ** self._E3
            elif alt < self._Level4:
                # Between 32 and 47 km
                pressure_pa[...] = (self._C4 + self._D4 * alt) ** self._E4
            else:
                # Between 47km and 51km
                pressure_pa[...] = self._F5 * np.exp(self._G5 * alt)
        return alt_it.operands[1]

    def _isapressalt_m(self, pressure_pa):
        """
        Returns the geopotential altitude at which the ISA has the given
        pressure.

        Args:
            pressure_pa: Pressure in Pascal.

        Returns:
            The corresponding (geopotential) altitude above mean sea level.

        """
        press_it = np.nditer([pressure_pa, None])
        for press_pa, alt_m in press_it:
            if press_pa > self._pLevel1:
                # Troposphere
                alt_m[...] = (press_pa ** (1 / self._E1) - self._C1) / self._D1
            elif press_pa > self._pLevel2:
                # Lower stratopshere
                alt_m[...] = (1 / self._G2) * np.log1p(
                    press_pa / self._F2 - 1)
            elif press_pa > self._pLevel3:
                # Upper stratopshere
                alt_m[...] = (press_pa ** (1 / self._E3) - self._C3) / self._D3
            elif press_pa > self._pLevel4:
                # Between 32 and 47 km
                alt_m[...] = (press_pa ** (1 / self._E4) - self._C4) / self._D4
            else:
                # Between 47km and 51km
                alt_m[...] = (1 / self._G5) * np.log1p(
                    press_pa / self._F5 - 1)
        return press_it.operands[1]

    def _isadens_kgpm3(self, altitude_m):
        """
        ISA density as a function of geopotential altitude.

        Args:
            altitude_m: Geopotential altitude above mean sea level.

        Returns:
            The ISA density, in kilograms per metre cubed.

        """
        alt_it = np.nditer([altitude_m, None])
        for alt, dens_kgpm3 in alt_it:
            if alt < self._Level1:
                # Troposphere
                dens_kgpm3[...] = (self._I1 + self._J1 * alt) ** self._L1
            elif alt < self._Level2:
                # Lower stratopshere
                dens_kgpm3[...] = self._M2 * np.exp(self._N2 * alt)
            elif alt < self._Level3:
                # Upper stratopshere
                dens_kgpm3[...] = (self._I3 + self._J3 * alt) ** self._L3
            elif alt < self._Level4:
                # Between 32 and 47 km
                dens_kgpm3[...] = (self._I4 + self._J4 * alt) ** self._L4
            else:
                # Between 47km and 51km
                dens_kgpm3[...] = self._M5 * np.exp(self._N5 * alt)
            # Adjust for the temperature offset
            dens_kgpm3[...] = \
                dens_kgpm3[...] / (1 + self.offset_deg / (
                        self._isatemp_k(alt) - self.offset_deg))

        return alt_it.operands[1]

    def _isadensalt_m(self, dens_kgpm3):
        """
        Returns the geopotential altitude at which the ISA has given density
        (kg/m^3).

        Args:
            dens_kgpm3: Air density in kilograms per metre cubed.

        Returns:
            The corresponding (geopotential) altitude above mean sea level.

        """
        dense_it = np.nditer([dens_kgpm3, None])
        for d_kgpm3, alt_m in dense_it:
            if d_kgpm3 > self._dLevel1:
                # Troposphere
                alt_m[...] = (d_kgpm3 ** (1 / self._L1) - self._I1) / self._J1
            elif d_kgpm3 > self._dLevel2:
                # Lower stratopshere
                alt_m[...] = (1 / self._N2) * np.log1p(
                    dens_kgpm3 / self._M2 - 1)
            elif d_kgpm3 > self._dLevel3:
                # Upper stratopshere
                alt_m[...] = (d_kgpm3 ** (1 / self._L3) - self._I3) / self._J3
            elif dens_kgpm3 > self._dLevel4:
                # Between 32 and 47 km
                alt_m[...] = (d_kgpm3 ** (1 / self._L4) - self._I4) / self._J4
            else:
                # Between 47km and 51km
                alt_m[...] = (1 / self._N5) * np.log1p(d_kgpm3 / self._M5 - 1)
            # Adjust for the temperature offset
        return dense_it.operands[1]

    def airtemp_k(self, altitude_m=0):
        """
        Compute temperatures in the selected atmosphere, in Kelvin.

        Args:
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            Ambient (static air) temperature, in Kelvin.

        Examples:

            >>> isa = Atmosphere()

            >>> print(f"ISA temperatures at SL, 5km, 10km (geopotential):",
            ...       f"{isa.airtemp_k([0, 5e3, 10e3])} K")
            ISA temperatures at SL, 5km, 10km (geopotential): [288.15 255.65 223.15] K

        """
        altitude_m = self._alttest(altitude_m)
        if self.is_isa:
            temperatures_k = self._isatemp_k(altitude_m)
        else:
            temperatures_k = self.profile.ftemp_k(altitude_m)
        return _reverttoscalar(temperatures_k)

    def airpress_pa(self, altitude_m=0):
        """
        Compute pressures in the selected atmosphere, in Pascal.

        Args:
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            Ambient (static air) pressure, in Pascal.

        """
        altitude_m = self._alttest(altitude_m)
        if self.is_isa:
            pressures_pa = self._isapress_pa(altitude_m)
        else:
            pressures_pa = self.profile.fp_pa(altitude_m)
        return _reverttoscalar(pressures_pa)

    def vsound_mps(self, altitude_m=0):
        """
        Speed of sound in m/s at an altitude given in m.

        Args:
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            Speed of sound at altitude, in metres per second.

        """
        altitude_m = self._alttest(altitude_m)
        temperatures_k = self.airtemp_k(altitude_m)
        vsounds_mps = _vsound_mps(temperatures_k)
        return _reverttoscalar(vsounds_mps)

    def airdens_kgpm3(self, altitude_m=0):
        """
        Ambient density in the current atmosphere in kilograms per metre cubed.

        Args:
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            Ambient air density, in kilograms per metre cubed.

        """
        altitude_m = self._alttest(altitude_m)
        if self.is_isa:
            densities_kgpm3 = self._isadens_kgpm3(altitude_m)
        else:
            densities_kgpm3 = self.profile.frho_kgpm3(altitude_m)
        return _reverttoscalar(densities_kgpm3)

    def mach(self, airspeed_mps, altitude_m=0):
        """
        Mach number at a given speed (m/s) and altitude (m).

        Args:
            airspeed_mps: The freestream speed of the air, in metres per second.
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            Mach number at altitude.

        """

        airspeed_mps = mtools.recastasnpfloatarray(airspeed_mps)
        # Airspeed may be negative, e.g., when simulating a tailwind, but Mach
        # must be >0
        if airspeed_mps.any() < 0:
            negmsg = (
                "Airspeed < 0. If intentional, ignore this. Positive Mach no."
                " returned."
            )
            warnings.warn(negmsg, RuntimeWarning)
            airspeed_mps = abs(airspeed_mps)

        # Check altitude range
        altitude_m = self._alttest(altitude_m)

        # Compute speed of sound at the given altitude(s)
        vs_mps = self.vsound_mps(altitude_m)

        return airspeed_mps / vs_mps

    def vsound_kts(self, altitude_m=0):
        """
        Compute the local speed of sound in knots.

        Args:
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            The local speed of sound, in knots.

        """
        return co.mps_kts(self.vsound_mps(altitude_m))

    def airpress_mbar(self, altitude_m=0):
        """
        Compute the local air pressure in millibar.

        Args:
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            The ambient air pressure, in millibar.

        """
        return co.Pa_mbar(self.airpress_pa(altitude_m))

    def airtemp_c(self, altitude_m=0):
        """
        Compute the local air temperature in Celsius.
        Args:
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            The ambient air temperature, in degrees Celsius.

        """
        return co.K_C(self.airtemp_k(altitude_m))

    def dynamicpressure_pa(self, airspeed_mps=0, altitude_m=0):
        """
        Compute the dynamic pressure in the current atmosphere at a given true
        airspeed and altitude.

        Args:
            airspeed_mps: The freestream speed of the air, in metres per second.
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            Dynamic pressure values, in Pascal.

        Examples:

            >>> isa = Atmosphere()

            >>> # True airspeed in metres per second
            >>> MPSTAS = 20
            >>> altitudelist_m = [0, 500, 1000, 1500]

            >>> q_Pa = isa.dynamicpressure_pa(MPSTAS, altitudelist_m)
            >>> q_mbar = co.Pa_mbar(q_Pa)
            >>> print(q_mbar)
            [2.44999974 2.33453737 2.22328473 2.11613426]

        """
        return 0.5 * self.airdens_kgpm3(altitude_m) * (airspeed_mps ** 2)

    def eas2tas(self, eas, altitude_m):
        """
        Converts EAS to TAS at a given altitude.

        The method first calculates the density ratio (the ratio of ambient
        density at altitude to the ambient density at sea level). Since
        equivalent airspeed is the airspeed at sea-level which produces the same
        dynamic pressure that true airspeed does at altitude, we simply correct
        EAS by the density ratio.

        Args:
            eas: The equivalent airspeed.
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            True airspeed in the same unit as the input.

        """
        dratio = self.airdens_kgpm3(altitude_m) / self.airdens_kgpm3(0)
        return eas / np.sqrt(dratio)

    def tas2eas(self, tas, altitude_m):
        """
        Convert TAS to EAS at a given altitude.

        Args:
            tas: The true airspeed.
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            Equivalent airspeed in the same unit as the input.

        """
        dratio = self.airdens_kgpm3(altitude_m) / self.airdens_kgpm3(0)
        return tas * np.sqrt(dratio)

    def mpseas2mpscas(self, mpseas, altitude_m):
        """
        Convert EAS (m/s) to CAS (m/s) at a given altitude (m).

        The relationship between the two depends on the Mach number M and the
        ratio of ambient air pressure at the current altitude to that of sea
        level. The relationship is approximated using:

        delta = p_alt / p_sl
        CAS ~ EAS * (1 + (1-delta)/8 * M^2 + 3 * (1-10delta+9delta^2)/640 * M^4)

        Args:
            mpseas: Equivalent airspeed, in metres per second.
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            A tuple (calibrated airspeed, Mach number), where calibrated
            airspeed is given in metres per second.

        """
        # Note: unit specific, as the calculation requires Mach no.
        mpstas = self.eas2tas(mpseas, altitude_m)
        machno = self.mach(mpstas, altitude_m)
        delta = self.airpress_pa(altitude_m) / self.airpress_pa()
        m2_term = (1.0 / 8.0) * (1 - delta) * (machno ** 2)
        m4_term = (3.0 / 640.0) * (1 - 10 * delta + 9 * (delta ** 2)) * (
                machno ** 4)
        return mpseas * (1 + m2_term + m4_term), machno

    def keas2kcas(self, keas, altitude_m):
        """
        Converts equivalent airspeed into calibrated airspeed.

        Args:
            keas: Equivalent airspeed, in knots.
            altitude_m: Geopotential altitudes at which the temperature is to
                be interrogated (float or array of floats).

        Returns:
            A tuple (kcas, Machno), where kcas is the calibrated airspeed in
            knots and Machno is the Mach number.

        Examples:

            >>> isa = Atmosphere()

            >>> keas = np.array([100, 200, 300])
            >>> altitude_m = co.ft_m(40_000)

            >>> kcas, mach = isa.keas2kcas(keas, altitude_m)
            >>> print(kcas)
            [101.25392563 209.93839073 333.01861569]

        """
        # Note: unit specific, as the calculation requires Mach no.
        np.asarray(keas)
        mpseas = co.kts_mps(keas)
        mpscas, machno = self.mpseas2mpscas(mpseas, altitude_m)
        kcas = co.mps_kts(mpscas)
        return kcas, machno


def mil_hdbk_310(high_or_low: str, temp_or_dens: str, alt_km: int):
    """
    Load an atmospheric data set from US Military Handbook 310.

    Args:
        high_or_low: Either of the strings "high" or "low".
        temp_or_dens: Either of the strings "temp" or "dens".
        alt_km: The altitude at which the high/low temp/dens condition occurs.

    Returns:
        An ObsProfile object.

    """

    m310name = high_or_low + '_' + temp_or_dens \
               + '_at_' + str(alt_km) + 'km.m310'

    _fstr = os.path.join(os.path.dirname(__file__), "data",
                         "_MHDBK310", m310name)

    # Also looking for m310 files in the current directory if not found
    # in a dedicated _MHDBK310 folder
    if not os.path.isfile(_fstr):
        _fstr = high_or_low + '_' + temp_or_dens \
                + '_at_' + str(alt_km) + 'km.m310'

    try:
        if temp_or_dens == 'temp':
            alt_km, t_1pct_k, rho_1pct_kgpm3, t_10pct_k, rho_10pct_kgpm3 = \
                np.loadtxt(_fstr, skiprows=0, unpack=True)
        else:
            alt_km, rho_1pct_kgpm3, t_1pct_k, rho_10pct_kgpm3, t_10pct_k = \
                np.loadtxt(_fstr, skiprows=0, unpack=True)

        atm1pct = Obsprofile(
            alt_m=co.km_m(alt_km), temp_k=t_1pct_k, rho_kgpm3=rho_1pct_kgpm3)
        atm10pct = Obsprofile(
            alt_m=co.km_m(alt_km), temp_k=t_10pct_k, rho_kgpm3=rho_10pct_kgpm3)

        return atm1pct, atm10pct

    except FileNotFoundError:
        print("No atmosphere data file found for this combination of inputs.")
        return None, None


def _reverttoscalar(scalarorvec):
    """ Return scalar response to scalar input. """
    if isinstance(scalarorvec, np.ndarray):
        if scalarorvec.ndim == 0:
            return scalarorvec.item()
        if sum(scalarorvec.shape) == 1:
            return scalarorvec[0]
    return scalarorvec


def _vsound_mps(temp_k):
    return _reverttoscalar(np.sqrt(1.4 * R_JPKGPK * temp_k))


def reciprocalhdg(heading_deg):
    """The reciprocal of a heading in degrees"""
    if heading_deg + 180 > 360:
        return heading_deg - 180
    return heading_deg + 180


def tempratio(temp_c, mach):
    """Ratio of total temperature and the standard SL temperature"""
    temp_k = co.C_K(temp_c)
    sealevelstdtmp_k = co.C_K(15.0)
    theta0 = (temp_k / sealevelstdtmp_k) * (
            1 + (mach ** 2) * (GAMMA_DRY_AIR - 1) / 2)
    return theta0


def tatbysat(mach, recfac=1.0):
    """Ratio of total and static air temperature at a given Mach no"""
    return 1 + 0.5 * (GAMMA_DRY_AIR - 1) * recfac * (mach ** 2)


def pressratio(pressure_pa, mach):
    """Ratio of total pressure and the standard SL pressure"""
    sealevelstdtpress_pa = 101325
    exp = GAMMA_DRY_AIR / (GAMMA_DRY_AIR - 1)
    delta0 = (pressure_pa / sealevelstdtpress_pa) * \
             (1 + (mach ** 2) * (GAMMA_DRY_AIR - 1) / 2) ** exp
    return delta0


def turbofanthrustfactor(temp_c, pressure_pa, mach, \
                         throttleratio=1, ptype="highbpr"):
    """Multiply SL static thrust by this to get thrust at specified conditions"""

    # Model based on Mattingly, J. D., "Elements of Gas Turbine Propulsion",
    # McGraw-Hill, 1996 and Mattingly, J. D. et al., "Aircraft Engine Design",
    # AIAA, 2000.

    theta0 = tempratio(temp_c, mach)
    delta0 = pressratio(pressure_pa, mach)

    if ptype == "highbpr":  # high bypass ratio
        if theta0 <= throttleratio:
            return delta0 * (1 - 0.49 * np.sqrt(mach))
        return delta0 * (
                1 - 0.49 * np.sqrt(mach) - 3 * (theta0 - throttleratio) / (
                1.5 + mach))

    if ptype == "lowbpr":  # low bypass, afterburner on
        if theta0 <= throttleratio:
            return delta0
        return delta0 * (1 - 3.5 * (theta0 - throttleratio) / theta0)

    # low bypass, afterburner off
    if theta0 <= throttleratio:
        return 0.6 * delta0
    return 0.6 * delta0 * (1 - 3.8 * (theta0 - throttleratio) / theta0)


def turbopropthrustfactor_matt(temp_c, pressure_pa, mach, throttleratio=1):
    """Multiply SL static thrust by this to get thrust at specified conditions"""

    # Model based on Mattingly, J. D., "Elements of Gas Turbine Propulsion",
    # McGraw-Hill, 1996 and Mattingly, J. D. et al., "Aircraft Engine Design",
    # AIAA, 2000.

    theta0 = tempratio(temp_c, mach)
    delta0 = pressratio(pressure_pa, mach)

    if mach <= 0.1:
        return delta0

    if theta0 <= throttleratio:
        return delta0 * (1 - 0.96 * (mach - 0.1) ** 0.25)
    return delta0 * (1 - 0.96 * (mach - 0.1) ** 0.25 - \
                     3 * (theta0 - throttleratio) / (8.13 * (mach - 0.1)))


def turbopropthrustfactor(dens_ratio, mach):
    """Multiply SL static thrust by this to get thrust at specified dens. ratio and Mach no."""

    # Model based on TPE331-10 thrust curves

    X_alt_ft = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
    X_alt_ft = [40000, 35000, 30000, 25000, 20000, 15000, 10000, 5000, 0]
    X_alt_ft = X_alt_ft[::-1]
    isa_sl_dens = 1.2249998682209657  # SSOT violation in the interest of speed
    X_dens_ratio = [0.24616951531253428, 0.3098749509963361, 0.3741322424507335,
                    0.44811893933199765, 0.5328112272029571, 0.6292375381232487,
                    0.7384790974933775, 0.8616704548972803, 1.0]
    Y_Mach = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    Z_thr_ratio = \
        np.array([[0.24301874, 0.22386998, 0.17652686, 0.14221442, 0.11592392,
                   0.09937799, 0.06240678],
                  [0.3128897, 0.30047179, 0.26491134, 0.21632642, 0.17804871,
                   0.1498797, 0.12426883],
                  [0.38302311, 0.3725948, 0.33679446, 0.27993518, 0.23118738,
                   0.20371041, 0.17623344],
                  [0.45869702, 0.44896454, 0.40851692, 0.33699743, 0.27831082,
                   0.24504059, 0.21177037],
                  [0.54553835, 0.53270352, 0.47928574, 0.39099644, 0.32349623,
                   0.27757851, 0.23166079],
                  [0.6440402, 0.62518827, 0.55076698, 0.44448741, 0.36925856,
                   0.30213092, 0.23500329],
                  [0.75391673, 0.72629241, 0.62626668, 0.49791383, 0.41741569,
                   0.3135676, 0.20971952],
                  [0.87466664, 0.83573268, 0.70577692, 0.53837968, 0.44090342,
                   0.25876522, 0.07662703],
                  [1., 0.92961537, 0.7332334, 0.53288748, 0.42905484, 0.2518872,
                   0.07471956]])

    thrust_ratio_spline = interpolate.RectBivariateSpline(X_dens_ratio, Y_Mach,
                                                          Z_thr_ratio, kx=2,
                                                          ky=2)

    return thrust_ratio_spline(dens_ratio, mach)[0]


def turbojetthrustfactor(temp_c, pressure_pa, mach, \
                         throttleratio=1, afterburner=False):
    """Multiply SL static thrust by this to get thrust at specified conditions"""

    # Model based on Mattingly, J. D., "Elements of Gas Turbine Propulsion",
    # McGraw-Hill, 1996 and Mattingly, J. D. et al., "Aircraft Engine Design",
    # AIAA, 2000.

    theta0 = tempratio(temp_c, mach)
    delta0 = pressratio(pressure_pa, mach)

    if afterburner:
        if theta0 <= throttleratio:
            return delta0 * (1 - 0.3 * (theta0 - 1) - 0.1 * np.sqrt(mach))
        return delta0 * (1 - 0.3 * (theta0 - 1) - 0.1 * np.sqrt(mach) - \
                         1.5 * (theta0 - throttleratio) / throttleratio)

    if theta0 <= throttleratio:
        return delta0 * 0.8 * (1 - 0.16 * np.sqrt(mach))
    return delta0 * 0.8 * (1 - 0.16 * np.sqrt(mach) - \
                           24 * (theta0 - throttleratio) / (
                                   (9 + mach) * theta0))


def pistonpowerfactor(density_kgpm3):
    """Gagg-Ferrar model. Multiply by this to get power at given density."""

    # Density ratio
    sigma = density_kgpm3 / 1.225
    return 1.132 * sigma - 0.132
