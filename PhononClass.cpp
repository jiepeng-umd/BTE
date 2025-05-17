/******************************************************************************
** FILE: PhononClass.cpp
******************************************************************************/
#include <iostream>
#include <vector>
#include <algorithm>
#include "PhononClass.h"
#include "Input.h"

// Use standard namespace only locally
using std::vector;

// Thread safety note: MTRand should be thread-safe or use thread-local instances when parallelizing

void set_phonon(Phonon &pPhonon, MTRand &mtrand, const Cell &cells,
                const vector<vector<double>> &omega,
                const vector<vector<vector<double>>> &velocity,
                int s, int m)
{
    double Rk = mtrand.rand();  // random [0,1]
    size_t i = 0, j = 0;
    const size_t Nq = cells.F_sc.size();
    const size_t Nbranch = (Nq > 0) ? cells.Polarize[0].size() : 0;

    // Find wavevector index
    if (Rk <= 0.0)
        i = 0;
    else if (Rk >= 1.0)
        i = Nq - 1;
    else
        i = std::distance(cells.F_sc.begin(), std::lower_bound(cells.F_sc.begin(), cells.F_sc.end(), Rk));

    // Find branch index
    double Pk = mtrand.rand();
    if (Pk <= 0.0)
        j = 0;
    else if (Pk >= 1.0)
        j = Nbranch - 1;
    else
        j = std::distance(cells.Polarize[i].begin(), std::lower_bound(cells.Polarize[i].begin(), cells.Polarize[i].end(), Pk));

    // Assign phonon properties
    pPhonon.omega = omega[i][j];
    pPhonon.kvector = i;
    pPhonon.branch = j;
    for (int k = 0; k < 3; ++k)
        pPhonon.velocity[k] = velocity[i][j][k];

    pPhonon.cell = s;
    pPhonon.position[0] = mtrand.rand() * Lx;
    pPhonon.position[1] = mtrand.rand() * Ly;
    pPhonon.position[2] = s * Lz + mtrand.rand() * Lz;
    pPhonon.temp = cells.temp;
}

void phonon_drift(Phonon &pPhonon)
{
    for (int i = 0; i < 3; ++i)
        pPhonon.position[i] += pPhonon.velocity[i] * dt;

    // Periodic boundary conditions in x,y
    if (pPhonon.position[0] < 0.0) pPhonon.position[0] += Lx;
    if (pPhonon.position[0] > Lx)  pPhonon.position[0] -= Lx;
    if (pPhonon.position[1] < 0.0) pPhonon.position[1] += Ly;
    if (pPhonon.position[1] > Ly)  pPhonon.position[1] -= Ly;

    // Set cell index in z, no periodic BCs
    pPhonon.cell = static_cast<int>(pPhonon.position[2] / Lz);
    if (pPhonon.position[2] < 0.0) pPhonon.cell = 0;
    if (pPhonon.position[2] > Lz * Nz) pPhonon.cell = Nz - 1;
}

// Calculate cumulative PDFs for phonon sampling in a cell
void calc_CPDFs(const vector<vector<double>> &omega,
                const vector<vector<double>> &Psc,
                double temp, Cell &cells, double &ntot)
{
    const size_t Nq = omega.size();
    const size_t Nbranch = (Nq > 0) ? omega[0].size() : 0;

    ntot = 0.0;
    double denom = 0.0;
    double tmp1 = 0.0;

    // Resize Cell arrays to correct sizes if needed
    if (cells.F_sc.size() != Nq) cells.F_sc.resize(Nq, 0.0);
    if (cells.Polarize.size() != Nq)
        cells.Polarize.assign(Nq, vector<double>(Nbranch, 0.0));
    else
        for (auto &vec : cells.Polarize)
            vec.assign(Nbranch, 0.0);

    vector<vector<double>> ndensity(Nq, vector<double>(Nbranch, 0.0));
    vector<double> sumN(Nq, 0.0);

    for (size_t i = 0; i < Nq; ++i)
    {
        double tmp = 0.0;
        for (size_t j = 0; j < Nbranch; ++j)
        {
            if (omega[i][j] > 0.0)
            {
                ndensity[i][j] = Vu / V_FC / Nq * feq(omega[i][j], temp);
                ntot += ndensity[i][j];
            }
            else
            {
                ndensity[i][j] = 0.0;
                // Optionally Psc[i][j] = 0.0; // but maybe keep it unchanged
            }
            tmp += ndensity[i][j] * Psc[i][j];
        }
        sumN[i] = tmp;
        denom += tmp;
    }

    if (denom == 0.0) denom = 1e-16; // avoid divide by zero

    for (size_t i = 0; i < Nq; ++i)
    {
        double tmp2 = 0.0;
        for (size_t j = 0; j < Nbranch; ++j)
        {
            tmp1 += ndensity[i][j] * Psc[i][j];
            tmp2 += ndensity[i][j] * Psc[i][j] / sumN[i];
            cells.Polarize[i][j] = tmp2;
        }
        cells.F_sc[i] = tmp1 / denom;
    }
}

void print_phonon(const Phonon &pPhonon, FILE *printfile)
{
    std::fprintf(printfile, "\n");
    std::fprintf(printfile, "The phonon is located in cell: %d\n", pPhonon.cell);
    std::fprintf(printfile, "The phonon frequency: %0.3e THz\n", pPhonon.omega * 1E-12);
    std::fprintf(printfile, "The phonon temperature: %0.1f\n", pPhonon.temp);
    std::fprintf(printfile, "The phonon kvector and branch index: q=%d, b=%d\n", pPhonon.kvector, pPhonon.branch);
    std::fprintf(printfile, "The phonon velocity: vx=%0.3f, vy=%0.3f, vz=%0.3f\n",
                 pPhonon.velocity[0], pPhonon.velocity[1], pPhonon.velocity[2]);
    std::fprintf(printfile, "The phonon position: x=%0.3f, y=%0.3f, z=%0.3f\n",
                 pPhonon.position[0] * 1E9, pPhonon.position[1] * 1E9, pPhonon.position[2] * 1E9);
    std::fprintf(printfile, "The phonon energy: %0.3e J\n", hbar * pPhonon.omega);
    std::fprintf(printfile, "\n");
}
