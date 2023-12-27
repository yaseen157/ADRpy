<!--
    Aircraft Design Recipes in Python (ADRpy)
    Copyright (C) 2023  András Sóbesteer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

![ADRpy](https://github.com/sobester/ADRpy/raw/master/docs/ADRpy/ADRpy_splash.png)

<h1 align="center">Aircraft Design Recipes in Python</h1>
<p align="center">András Sóbester, Yaseen Reza</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/ADRpy.svg)](https://badge.fury.io/py/ADRpy)
[![Build Status](https://travis-ci.com/sobester/ADRpy.svg?branch=master)](https://travis-ci.com/sobester/ADRpy)

---

ADRpy is an academic, teaching resource containing aircraft conceptual design
and performance analysis tools such as:
- Virtual (design) atmospheres
- Constraint analysis methods
- Propulsion system performance models
- Unit conversion functions
- *and much more...*

For a detailed description of the library, please consult the
[Documentation](https://adrpy.readthedocs.io/en/latest/). To get started,
follow the instructions below.

For video tutorials and explainers (a.k.a. *ADRpy Shorts*) scroll to the bottom
of this page.

Components of this library are written to be both easy to read and modify 
without requiring significant coding knowledge. Confident coders looking for a
more flexible and expandable library architecture may wish to check out ADRpy's
companion library [CARPy](https://github.com/yaseen157/carpy/tree/main) (currently a work in progress!).

---

## Installation

ADRpy is written in Python 3 and tested in Python version 3.9.

*It is not available for Python 2.*

On most systems you should be able to simply open an operating system terminal
and at the command prompt type

    $ pip install -e ADRpy
    
or

    $ python -m pip install -e ADRpy

NOTE: the `-e` flag is optional, but recommended. It makes the ADRpy install
*editable*, allowing users to poke and prod the machinery of the library. 

NOTE: `pip` is a Python package; if it is not available on your system, download
[get-pip.py](https://bootstrap.pypa.io/get-pip.py) and run it in Python by entering

    $ python get-pip.py
    
at the operating system prompt.

An alternative approach to installing ADRpy is to clone the GitHub repository, by typing

    $ git clone https://github.com/sobester/ADRpy.git

at the command prompt and then executing the setup file in the same directory by entering:

    $ python setup.py install

---
    
## A 'hello world' example: atmospheric properties


There are several options for running the examples shown here: you could copy and paste them 
into a `.py` file, save it and run it in Python, or you could enter the lines, in sequence,
at the prompt of a Python terminal. You could also copy and paste them into a Jupyter notebook
(`.ipynb` file) cell and execute the cell.

```python
from ADRpy import atmospheres as at
from ADRpy import unitconversions as co

# Instantiate an atmosphere object: an ISA with a +10C offset
isa = at.Atmosphere(offset_deg=10)

# Query the ambient density in this model at 41,000 feet 
print("ISA+10C density at 41,000 feet (geopotential):", 
      isa.airdens_kgpm3(co.feet2m(41000)), "kg/m^3")
```

You should see the following output:

    ISA+10C density at 41,000 feet (geopotential): 0.27472588853063956 kg/m^3

A design example: wing/powerplant sizing for take-off
-----------------------------------------------------

```python
# Compute the thrust to weight ratio required for take-off, given
# a basic design brief, a basic design definition and a set of 
# atmospheric conditions

from ADRpy import atmospheres as at
from ADRpy import constraintanalysis as ca
from ADRpy import unitconversions as co


# The environment: 'unusually high temperature at 5km' atmosphere
# from MIL-HDBK-310. 

# Extract the relevant atmospheric profiles...
profile_ht5_1percentile, _ = at.mil_hdbk_310('high', 'temp', 5)

# ...then use them to create an atmosphere object 
m310_ht5 = at.Atmosphere(profile=profile_ht5_1percentile)

#====================================================================

# The take-off aspects of the design brief:
designbrief = {'rwyelevation_m':1000, 'groundrun_m':1200}

# Basic features of the concept:
# aspect ratio, throttle ratio 
designdefinition = {'aspectratio':7.3, 'tr':1.05}

# Initial estimates of aerodynamic performance:
designperf = {'CDTO':0.04, 'CLTO':0.9, 'CLmaxTO':1.6,
              'mu_R':0.02} # ...and wheel rolling resistance coeff.

# An aircraft concept object can now be instantiated
concept = ca.AircraftConcept(designbrief, designdefinition,
                             designperf, m310_ht5, "Piston")

#====================================================================

# Compute the required standard day sea level thrust/MTOW ratio reqd.
# for the target take-off performance at a range of wing loadings:
wingloadinglist_pa = [2000, 3000, 4000, 5000]

tw_sl, _ = concept.constrain_takeoff(wingloadinglist_pa)

print("Required T/W under MIL-HDBK-310 conditions:")
print("\nT/W (SL, static thrust):", tw_sl)
```

You should see the following output:

    Required T/W under MIL-HDBK-310 conditions:

    T/W (SL, static thrust): [0.32348606 0.44523742 0.56698878 0.68874015]

---
## More extensive examples - a library of notebooks


To view them on GitHub, go to [ADRpy's notebooks folder](https://github.com/sobester/ADRpy/tree/master/docs/ADRpy/notebooks).

Alternatively, grab the whole repository as a .zip by clicking the big, green 'Code' button at the top of this page. 

---
## ADRpy Shorts - video tutorials and explainers


**1. An Aircraft Engineer's Brief Introduction to Modelling the Atmosphere**

[![1. An Aircraft Engineer's Brief Introduction to Modelling the Atmosphere](http://img.youtube.com/vi/II9vuVCgV-w/0.jpg)](http://www.youtube.com/watch?v=II9vuVCgV-w)


**2. On V-n Diagrams and How to Build them in ADRpy**

[![2. On V-n Diagrams and How to Build them in ADRpy](http://img.youtube.com/vi/s-d5z-BQovY/0.jpg)](http://www.youtube.com/watch?v=s-d5z-BQovY)


**3. Speed in aviation - GS, WS, TAS, IAS, CAS and EAS**

[![3. Speed in aviation - GS, WS, TAS, IAS, CAS and EAS](http://img.youtube.com/vi/WSzDXlTlXiI/0.jpg)](http://www.youtube.com/watch?v=WSzDXlTlXiI)


**4. Wing and propulsion system sizing with ADRpy**

[![4. Wing and propulsion system sizing](http://img.youtube.com/vi/TMM7mE1NjaE/0.jpg)](https://www.youtube.com/watch?v=TMM7mE1NjaE)
