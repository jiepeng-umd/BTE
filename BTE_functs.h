/*******************************************************************************************
** FILE: BTE_functs.h
*******************************************************************************************/

#ifndef _BTE_functs_h_included_
#define _BTE_functs_h_included_

#include <vector> // For std::vector
#include "Input.h"
#include "PhononClass.h"

// Calculate total phonon energy at temperature Temp, over omega dispersion
double E_eq(const std::vector<std::vector<double>>& omega, double Temp);

// Bose-Einstein distribution function
double feq(double frequency, double temperature);

// Calculate cell temperature by inverting energy using bisection method
double calc_cell_temp(double low_temp, double high_temp, double energy, const std::vector<std::vector<double>>& omega);

// Compute mean and standard deviation of vector data
void SD(double& mean, double& sd, const std::vector<double>& vec);

// Distribute N elements among ntask processes, output sendcount and displacement arrays for MPI
void vector_distribute(int ntask, int N, int sendcount[], int disp[]);

#endif
