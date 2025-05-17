# BTE
This is a C++/OpenMP code that performs Monte-Carlo simulation of phonon and electron transport in a 3D domain. Currently, a simple rectangular on the order of a few thousand nanometers 
are considered. The Input folder contains all the phonon and electron properties, including phonon dispersion and electron band structure in the full Brillouin zone and the ph-ph/ph-el 
scattering rates. In the case of a temperature gradient across the domain in one direction, the particles diffuse and scatter in the domain according to the dimensions and the scattering
probabilities, resulting in a temperature and energy distribution. Statistic analysis of energy and flux eventually produces thermal conductivity and heat-map of the simulation domain.
