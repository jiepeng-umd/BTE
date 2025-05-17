#ifndef INPUT_H
#define INPUT_H

#include <string>

// Avoid "using namespace std;" in headers to prevent polluting global namespace

// Global variables declared as extern to be defined in one .cpp file
extern int i, j, k, s, Nir, Nq, Natom, Nbranch, Nfull, Nz, Nstep, Nout, Navg, Nprint, Np_hot, Np_cold, Tstart, Tend, Tincrement, nT;
extern double Lx, Ly, Lz, Vu, V_FC, W, dt, Vtot, T_target, d_T, T_hot, T_cold;

// Physical constants (defined as constexpr since they are constants)
constexpr double hbar = 6.62607015E-34;   // Planck constant (J·s)
constexpr double kB = 1.3806503E-23;      // Boltzmann constant (J/K)
constexpr double unit_radps2Hz = 0.5 * 1.0E12 / 3.14159265358979323846;

// Function declarations
void input(const std::string& in_path);
double extractDouble(const std::string& text);

#endif // INPUT_H
