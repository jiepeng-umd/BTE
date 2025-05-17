/******************************************************************************
** FILE: BTE_functs.cpp
******************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include "BTE_functs.h"
#include "Input.h"

using namespace std;

// Bose-Einstein distribution function
double feq(double frequency, double temperature)
{
    // Protect against division by zero or negative temperature
    if (temperature <= 0.0 || frequency <= 0.0) return 0.0;
    return 1.0 / (exp(hbar * frequency / (kB * temperature)) - 1.0);
}

/* Calculate total phonon energy at temperature Temp, 
   summing over phonon modes in omega, scaled by volumes Vu, V_FC */
double E_eq(const std::vector<std::vector<double>>& omega, double Temp)
{
    double Etot = 0.0;
    for (int i = 0; i < Nq; i++)
    {
        for (int j = 0; j < Nbranch; j++)
        {
            if (omega[i][j] > 0)
            {
                Etot += hbar * omega[i][j] * Vu / V_FC / Nq * feq(omega[i][j], Temp);
            }
        }
    }
    return Etot;
}

// Find cell temperature from cell energy using bisection method
double calc_cell_temp(double low_temp, double high_temp, double energy, const std::vector<std::vector<double>>& omega)
{
    const int max_iterations = 30;
    const double tolerance = 1.0E-4;
    double a = low_temp, b = high_temp;
    double f_at_a = W * energy - E_eq(omega, a);
    double f_at_b = W * energy - E_eq(omega, b);

    if (f_at_a * f_at_b > 0.0)
    {
        cerr << "Error in calc_cell_temp: interval does not bracket root." << endl;
        return 0.0;
    }
    else if (f_at_a == 0.0) return a;
    else if (f_at_b == 0.0) return b;

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        double mid = 0.5 * (a + b);
        double f_at_mid = W * energy - E_eq(omega, mid);

        if (f_at_mid == 0.0) return mid;

        if (f_at_mid * f_at_a < 0.0)
        {
            b = mid;
            f_at_b = f_at_mid;
        }
        else
        {
            a = mid;
            f_at_a = f_at_mid;
        }

        if ((b - a) / 4.0 < tolerance)
            return mid;
    }

    cerr << "Error: maximum iterations exceeded in calc_cell_temp." << endl;
    return 0.0;
}

// Compute mean and standard deviation of vector values
void SD(double& mean, double& sd, const std::vector<double>& vec)
{
    int len = vec.size();
    if (len == 0)
    {
        mean = 0.0;
        sd = 0.0;
        return;
    }

    mean = 0.0;
    for (const auto& val : vec)
        mean += val;
    mean /= len;

    sd = 0.0;
    for (const auto& val : vec)
        sd += (val - mean) * (val - mean);

    sd = sqrt(sd / len);
}

// Distribute a vector of size N among ntask processes,
// filling sendcount[] and disp[] arrays for MPI scatter/gather operations
void vector_distribute(int ntask, int N, int sendcount[], int disp[])
{
    int num_per_task = N / ntask;
    int remainder = N % ntask;
    for (int i = 0; i < ntask; i++)
    {
        sendcount[i] = num_per_task + (i < remainder ? 1 : 0);
        disp[i] = (i == 0) ? 0 : disp[i-1] + sendcount[i-1];
    }
}
