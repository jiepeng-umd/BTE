/*******************************************************************************************
** FILE: PhononClass.h
*******************************************************************************************/
#ifndef _PHONONCLASS_H_INCLUDED_
#define _PHONONCLASS_H_INCLUDED_

#include <vector>
#include "MersenneTwister.h"  // Random Number Generator
#include "BTE_functs.h"
#include "Input.h"

// Phonon structure: stores phonon properties
struct Phonon
{
    double temp;
    double omega;
    int branch;
    int kvector;
    double velocity[3];
    int cell;
    double position[3];
};

// Cell structure: stores cell properties and cumulative PDFs for sampling
struct Cell
{
    double temp;
    double Energy;
    int Np;  // Number of phonons in the cell

    std::vector<double> F_sc = std::vector<double>(Nq, 0.0); // cumulative distribution function for phonon frequency

    // 2D vector for branch sampling: size Nq x Nbranch
    std::vector<std::vector<double>> Polarize = std::vector<std::vector<double>>(Nq, std::vector<double>(Nbranch, 0.0));

    // Scattering probability: size Nq x Nbranch
    std::vector<std::vector<double>> Psc = std::vector<std::vector<double>>(Nq, std::vector<double>(Nbranch, 0.0));
};

// Functions declarations:

// Initialize phonon with frequency, branch, velocity, and position sampled from distributions
void set_phonon(Phonon &pPhonon, MTRand &mtrand, const Cell &cells,
                const std::vector<std::vector<double>> &omega,
                const std::vector<std::vector<std::vector<double>>> &velocity,
                int s, int m);

// Update phonon position with drift and enforce periodic boundary conditions
void phonon_drift(Phonon &pPhonon);

// Calculate cumulative PDFs (F_sc and Polarize) for phonon sampling in a cell
void calc_CPDFs(const std::vector<std::vector<double>> &omega,
                const std::vector<std::vector<double>> &Psc,
                double temp, Cell &cells, double &ntot);

// Print phonon properties to file
void print_phonon(const Phonon &pPhonon, FILE *printfile);

#endif // _PHONONCLASS_H_INCLUDED_
