#include <iostream>
#include <cmath>  //For exp()
#include <cstdlib>  //For abs()
#include <fstream>  //For outputting to files
#include <vector>  //For STD vector class
#include <algorithm> //For vector swap function
#include <chrono> //For measuring time
#include <numeric>
#include <sys/types.h>
#include <sys/stat.h> //For writing to selected directories
#include <typeinfo> //For examining variable type
#include <string>
#include <ctime>

#include "MersenneTwister.h"  //Random number generator
#include "PhononClass.h"
#include "BTE_functs.h"
#include "Input.h"
#include "mpi.h"
// #include "fmt/format.h"

using namespace std;

//***********************************Declare global variables so that all the functions can use them**********************
int i, j, k, s, Nir, Nq, Natom, Nbranch, Nfull, Nz,  Nstep, Nout, Navg, Nprint, Np_hot, Np_cold, Tstart, Tend, Tincrement, nT;
double Lx, Ly, Lz, Vu, V_FC, W, dt, Vtot, T_target, d_T, T_hot, T_cold;
double flux[3];

//***************************************************End global variables**********************************************

int main(int argc, char **argv)
{
	//*****************************************Define simulation parameters*******************************************

	int m, rank, ntask; // m: MC timestep index. rank: ID of the process. ntask: number of processes.
	double dummy, start, end, g_start, Time, real_T_hot, real_T_cold; //dummy: dummy variable. start and end: variables for timing the code execution. real_T_hot and real_T_cold: temperature of the hot and cold cell resulting from initializing the phonons.

	// Initialize MPI.
	int ierr = MPI_Init ( &argc, &argv );
	if ( ierr != 0 )
	{
		cout << "Monte-Carlo BTE solver - Fatal error!\n";
		cout << "  MPI_Init returned ierr = " << ierr << "\n";
		exit ( 1 );
	}
	MPI_Comm_size(MPI_COMM_WORLD, &ntask); //get # of processes
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get ID of processes

	//start time of the run
	g_start = MPI_Wtime();
	//***********************************Output Files****************************************************************
	//***************************************************************************************************************
	// Creating a directory to write all output files to
	std::string out_path = "./output/";
	mkdir(out_path.c_str(), 0777);
	// int N=70;
	// char Buffer[N];

	//MC-BTEoutput file
	FILE *out_file = fopen(string(out_path + "BTE.out").c_str(), "w");
	setbuf(out_file, NULL); //No buffer flushing, data is written to file immediately

	//Thermal conductivity data file
	FILE *flux_file = fopen(string(out_path + "flux.dat").c_str(), "w");
	// setbuf(flux_file, NULL);

	//Average Thermal conductivity data file
	FILE *kappa_file = fopen(string(out_path + "kappa.dat").c_str(), "w");
	// setbuf(kappa_file, NULL);

	//Temperature of each cell data file
	FILE *T_file = fopen(string(out_path + "temperature.dat").c_str(), "w");
	// setbuf(T_file, NULL);

	//Average temperature data file
	FILE *T_avg_file = fopen(string(out_path + "temperature_avg.dat").c_str(), "w");
	// setbuf(T_avg_file, NULL);

	//Phonon number in each cell data file
	FILE *Nfile = fopen(string(out_path + "PhononNumber.dat").c_str(), "w");
	// setbuf(Nfile, NULL);

	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	//***********************************Input Files**************************************************************
	//************************************************************************************************************
	std::string in_path = "./Inputs/"; //Directory that contains all the input files
	input(in_path); //call the input function to read in parameters

	Vu = Lx * Ly * Lz; //volume of the simulation cell
	Vtot = Vu * Nz; //total volume of the simulation domain
	T_hot = T_target + d_T;
	T_cold = T_target - d_T; //The hot and cold temperature is set according to the target temperature and the preset temperature difference.
	Nbranch = 3 * Natom; //# of phonon branches.

	/*Formating the output file*/
	int totalsize = 100;
	std::string printout = "Simulation parameters";
	int left = int((totalsize - printout.size()) / 2), right = totalsize - left - printout.size();

	if (rank == 0)
	{
		std::fprintf(out_file, "%s\n", (std::string(left, '-') + printout + std::string(right, '-')).c_str()); //Formatting blocks
		std::fprintf(out_file, "Timestep: dt = %0.1f ps.\n", dt * 1.0E12); //timestep size
		std::fprintf(out_file, "Number of atoms per unit cell: Natom = %d.\n", Natom); //Number of atoms per unit cell
		std::fprintf(out_file, "Total simulation time: t_tot = %0.1f ns.\n", Nstep * dt * 1.0E9); //Total simulation time
		std::fprintf(out_file, "The temperature of interest is %0.1f K.\n", T_target); //Total simulation time
		std::fprintf(out_file, "The temperature of the two ends cells are: T_hot = %0.1fK, T_cold = %0.1fK.\n", T_hot, T_cold); //Temperature of two boundary cells
		std::fprintf(out_file, "Number of cells along z-direction: Nz = %d.\n", Nz); //# of cells along z-direction
		std::fprintf(out_file, "The simulation cell dimensions are : Lx = %0.1f nm, Ly = %0.1f nm, Lz = %0.1f nm.\n", Lx * 1.0E9, Ly * 1.0E9, Lz * 1.0E9); //Dimensions of the simulation cell
		std::fprintf(out_file, "Volume of the unit cell is : V_u = %0.3e Angstrom^3.\n", V_FC * 1.0E30); //unitcell volume
		std::fprintf(out_file, "Volume of the Simulation domain is : V_s = %0.3e Angstrom^3.\n", Vu * Nz * 1.0E30); //simulation cell volume
		std::fprintf(out_file, "%s\n", std::string(totalsize, '-').c_str());

		printout = "Data file inputs";
		left = int((totalsize - printout.size()) / 2); right = totalsize - left - printout.size();
		std::fprintf(out_file, "\n%s\n", (std::string(left, '-') + printout + std::string(right, '-')).c_str()); //Formatting blocks
	}
	//***************Read in the phonon wavevectors in the irreducible wedge******************************************
	/*Read in the wavevectors in the irreducible wedge of Brillouin zone (BZ) from ShengBTE output, of which the relative coordinates with respect to the reciprocal lattice vectors are shown in the last 3 columns. The matrix "k" is of size Nir*6. The 1st and 2nd columns correspond the indices of those kpoints numbered in the irreducible wedge and in the whole Brillouin zone, respectively. The 3rd column lists the corresponding degeneracies.*/
	Nir = 0, Nq = 0; //number of qpoints in the irreducible wedge (Nir) and full BZ (Nq)
	std::string filename_idx = "BTE.qpoints", line;
	std::ifstream idx_stream(std::string(in_path + filename_idx).c_str());
	if (idx_stream.is_open())
	{
		// If file has been correctly opened...
		while (getline(idx_stream, line)) Nir++; //number of qpoints in the irreducible wedge

		idx_stream.clear(); // clear any fail and EOF bits
		idx_stream.seekg(0); // rewind the buffer to the beginning of the file
	}
	else
	{
		if (rank == 0) std::fprintf(out_file, "Unable to open the phonon wavevector file in the irreducible wedge...\n");
		return 0;
	}
	std::vector<std::vector <double>> idx(Nir, std::vector<double>(6));

	for (i = 0; i < Nir; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			idx_stream >> idx[i][j]; // fill the row with col elements
		}
		Nq += (int) idx[i][2];
	}
	if (rank == 0) std::fprintf(out_file, "Successfully read in the phonon wavevector file in the irreducible wedge.\n");
	idx_stream.close();

	// FILE *idxfile = fopen(string(out_path + "idx.dat").c_str(), "w");
	// for (i = 0; i < Nir; i++)
	// {
	// 	fprintf(idxfile, "%d  %d  %d  %0.5e  %0.5e  %0.5e\n", int(idx[i][0]), int(idx[i][1]), int(idx[i][2]), idx[i][3], idx[i][4], idx[i][5]);
	// }
	// fclose(idxfile);


	//***************Read in the phonon wavevectors in the full BZ from ShengBTE output******************************************
	/*Read in the wavevectors in the full BZ from ShengBTE output, of which the relative coordinates with respect to the reciprocal lattice vectors are shown in the last 3 columns. The matrix "wavevector" is of size Nq*5. The 1st and 2nd columns correspond the indices of those qpoints numbered in the full BZ and in the irreducible wedge, respectively.*/
	std::string filename_wavevector = "BTE.qpoints_full";
	std::ifstream wavevector_stream(std::string(in_path + filename_wavevector).c_str());

	std::vector<std::vector <double>> wavevector(Nq, std::vector<double>(5)); //phonon wavevectors in the full BZ

	// Read file
	if (wavevector_stream.is_open())
	{
		for (i = 0; i < Nq; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				wavevector_stream >> wavevector[i][j]; // fill the row with col elements
			}
		}
		if (rank == 0) std::fprintf(out_file, "Successfully read in the phonon wavevector file in the full BZ.\n");
	}
	else
	{
		if (rank == 0) std::fprintf(out_file, "Unable to open the phonon wavevector file in the full BZ...\n");
		return 0;
	}
	wavevector_stream.close();

	// FILE *idxfile = fopen(string(out_path + "idx.dat").c_str(), "w");
	// for (i = 0; i < Nq; i++)
	// {
	// 	fprintf(idxfile, "%d  %d  %0.5e  %0.5e  %0.5e\n", int(wavevector[i][0]), int(wavevector[i][1]), wavevector[i][2], wavevector[i][3], wavevector[i][4]);
	// }
	// fclose(idxfile);

	//***********************Read in the phonon frequencies from ShengBTE output**********************************
	std::string filename_omega = "BTE.omega";
	std::ifstream dispersion_stream(std::string(in_path + filename_omega).c_str());
	std::vector<std::vector <double>> omega_irr(Nir, std::vector<double>(Nbranch)); //phonon dispersion in the irreducible wedge.
	if (dispersion_stream.is_open())
	{
		for (i = 0; i < Nir; i++)
		{
			for (j = 0; j < Nbranch; j++)
			{
				dispersion_stream >> omega_irr[i][j];
			}
		}
		if (rank == 0) std::fprintf(out_file, "Successfully read in the phonon dispersion file.\n");
	}
	else
	{
		if (rank == 0) std::fprintf(out_file, "Unable to open the phonon dispersion file...\n");
		return 0;
	}
	dispersion_stream.close();

	// FILE *idxfile = fopen(string(out_path + "idx.dat").c_str(), "w");
	// for (i = 0; i < Nir; i++)
	// {
	// 	fprintf(idxfile, "%0.5e  %0.5e  %0.5e  %0.5e  %0.5e  %0.5e\n", omega_irr[i][0], omega_irr[i][1], omega_irr[i][2], omega_irr[i][3], omega_irr[i][4],omega_irr[i][5]);
	// }
	// fclose(idxfile);

	std::vector<std::vector <double>> omega(Nq, std::vector<double>(Nbranch)); //phonon dispersion in the full BZ.
	for (i = 0; i < Nq; i++)
	{
		for (j = 0; j < Nbranch; j++)
		{
			omega[i][j] = omega_irr[int(wavevector[i][1]) - 1][j] * unit_radps2Hz; //convert omega from ShengBTE (rad/ps) to SI (Hz)
		}
	}
	// FILE *idxfile = fopen(string(out_path + "idx.dat").c_str(), "w");
	// for (i = 0; i < Nq; i++)
	// {
	// 	fprintf(idxfile, "%0.5e  %0.5e  %0.5e  %0.5e  %0.5e  %0.5e\n", omega[i][0], omega[i][1], omega[i][2], omega[i][3], omega[i][4],omega[i][5]);
	// }
	// fclose(idxfile);

	//**************************Read in the phonon velocities from ShengBTE output******************************************
	//The matrix "velocity" is of size Nq*Nbranch*3
	std::string filename_velocity = "BTE.v_full";
	std::ifstream velocity_stream(std::string(in_path + filename_velocity).c_str());
	std::vector<std::vector<std::vector<double> > > velocity( Nq, std::vector<std::vector<double> >(Nbranch, std::vector<double>(3))); //phonon group velocity in the full BZ

	if (velocity_stream.is_open())
	{	// If file has correctly opened...
		// Dynamically store data into array
		for (int j = 0; j < Nbranch; j++)
		{
			for (int i = 0; i < Nq; i++)
			{
				for (int k = 0; k < 3; k++)
				{
					velocity_stream >> velocity[i][j][k];
					velocity[i][j][k] *= 1.0E3; //convert velovity from ShengBTE output(Km/s) to SI(m/s)
				}
			}
		}
		if (rank == 0) std::fprintf(out_file, "Successfully read in the phonon velocity file.\n");
	}
	else
	{
		if (rank == 0) std::fprintf(out_file, "Unable to open velocity file...\n");
		return 0;
	}
	velocity_stream.close();

	// FILE *idxfile = fopen(string(out_path + "idx.dat").c_str(), "w");
	// for (int j = 0; j < Nbranch; j++)
	// {
	// 	for (int i = 0; i < Nq; i++)
	// 	{
	// 		fprintf(idxfile, "%0.5e  %0.5e  %0.5e\n", velocity[i][j][0]/1E3, velocity[i][j][1]/1E3, velocity[i][j][2]/1E3);
	// 	}
	// }

	// fclose(idxfile);


	nT = int((Tend - Tstart) / Tincrement) + 1; //Number of temperatures
	std::vector<int> temperature(nT);

//Relaxation times are stored in matrix "RT" of size "nT*Nq*Nbranch", i.e, at each temperature, RTs of phonons are stored in a "Nq*Nbranch" matrix for each phonon mode.
	vector<vector<vector<double> > > RT_irr( nT, vector<vector<double> >(Nir, vector<double>(Nbranch))); //RTs of phonons in the irreducible wedge
	std::string filename_RT; //RT file name
	std::ifstream RT_stream; //For reading data in the RT files

	for (k = 0; k < nT; k++)
	{
		temperature[k] = Tstart + k * Tincrement;
		filename_RT = "T" + std::to_string(temperature[k]) + "K/";

		RT_stream.open(std::string(in_path + filename_RT + "BTE.w").c_str());
		if (RT_stream.is_open())
		{
			for (j = 0; j < Nbranch; j++)
			{
				for (i = 0; i < Nir; i++)
				{
					RT_stream >> dummy >> RT_irr[k][i][j]; //Because of the format of BTE.w file by ShengBTE, the first column is the frequency of each phonon mode which we do not need to store here.
				}
			}
		}
		else
		{
			if (rank == 0) std::fprintf(out_file, "Unable to open the ph-ph relaxation time file at %dK.../n", int(temperature[k]));
			return 0;
		}
		RT_stream.close();
	}
	if (rank == 0) std::fprintf(out_file, "Successfully read in all the ph-ph relaxation time files.\n");
	// FILE *idxfile = fopen(string(out_path + "idx.dat").c_str(), "w");
	// for (j=0;j<Nbranch;j++)
	// {
	// 	for (i = 0; i < Nir; i++) fprintf(idxfile, "%0.5e\n", RT_irr[0][i][j]);
	// }
	// fclose(idxfile);


	vector<vector<vector<double> > > RT( nT, vector<vector<double> >(Nq, vector<double>(Nbranch))); //RTs of phonons in the full BZ
	for (k = 0; k < nT; k++)
	{
		for (i = 0; i < Nq; i++)
		{
			for (j = 0; j < Nbranch; j++)
			{
				RT[k][i][j] = 1 / RT_irr[k][int(wavevector[i][1]) - 1][j] * 1.0E-12; //convert scattering rates (ps^(-1)) to RT (s)
			}
		}
	}
	if (rank == 0) std::fprintf(out_file, "%s\n", std::string(totalsize, '-').c_str()); //End of "Data file inputs" section


	// FILE *RT_data = fopen("RT.dat", "w");
	// for (k = 0; k < nT; k++)
	// {
	// 	i = 1; j = 4;
	// 	if (rank == 0) std::fprintf(RT_data, "%d %0.1e\n", temperature[k], RT[k][i][j]);
	// }
	// fclose(RT_data);
	// if (rank == 0)
	// {
	// 	FILE *ph_flux = fopen("phonon_flux.dat", "w");
	// 	setbuf(ph_flux, NULL);
	// 	double tot_flux[3] = {0}, tot_velocity[3] = {0};
	// 	for (i = 0; i < Nq; i++)
	// 	{
	// 		for (j = 0; j < Nbranch; j++)
	// 		{
	// 			fprintf(ph_flux, "%10.3e  ", 1 / Vtot  * W * hbar * omega[i][j] * velocity[i][j][2] * Vtot / V_FC / Nq * feq(omega[i][j], 300));

	// 			for (k = 0; k < 3; k++)
	// 			{
	// 				tot_velocity[k] += velocity[i][j][k];
	// 				if (omega[i][j] != 0) tot_flux[k] += 1 / Vtot  * W * hbar * omega[i][j] * velocity[i][j][k] * Vtot / V_FC / Nq * feq(omega[i][j], 300);
	// 			}
	// 		}
	// 		fprintf(ph_flux, "\n");
	// 	}
	// 	fprintf(ph_flux, "%10.3e  %10.3e  %10.3e\n", tot_velocity[0], tot_velocity[1], tot_velocity[2]);
	// 	fprintf(ph_flux, "%10.3e  %10.3e  %10.3e\n", tot_flux[0], tot_flux[1], tot_flux[2]);
	// 	fclose(ph_flux);
	// }
	// std::fclose(ph_flux);
	// FILE *ph_freq = fopen("omega.dat", "w");
	// for (i = 0; i < Nq; i++)
	// {
	// 	for (j = 0; j < Nbranch; j++)
	// 	{
	// 		fprintf(ph_freq, "%10.3e  ", omega[i][j] * 1e-12);
	// 	}
	// 	fprintf(ph_freq, "\n");
	// }
	// std::fclose(ph_freq);
	//***********************************End Input Files**************************************************************
	//****************************************************************************************************************
	//Time spent on reading inputs
	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();
	Time = end - start;
	if (rank == 0) std::fprintf(out_file, "The read in cost time: %0.2f s\n", Time);

	//***********************************Variables declaration and Initialization*********************
	double Np_ToT = 0;
	int Np_tot = 0; //# of phonons in the entire simulation domain. To avoid data overflow in C++ due to large number of phonons in a real simulation, a double number is used first, which is converted to an integer using the scaling factor
	std::vector<double> cell_np(Nz);
	std::vector<int> cell_Np(Nz);
	// Inital number of phonons can be vary large and may exceed maximum number C++ can store as an integer, therefore we use double "cell_np(Nz)" to store this number first and scale it down using the Scaling factor W, which is finally converted to an integer number "cell_Np(Nz)".
	//Initialize the scattering matrix
	std::vector< std::vector<double> > Psc(Nq, std::vector<double>(Nbranch, 1.0E0)); //Scattering probability for each phonon mode, initialized to 1.0.
	std::vector< std::vector<double> > E(Nq, std::vector<double>(Nbranch)); //Phonon mode-dependent energy per unitcell
	std::vector< std::vector<double> > ndensity(Nq, std::vector<double>(Nbranch)); //# of phonons per cell (physical)

	//Initialize the cell state
	std::vector<Cell> cells(Nz);
	/*The cell state is defined by a number of variables.
	cells.temp - pseudo-temperature of the cell
	cells.Energy - energy of the cell by summing over all the phonons in the cell
	cells.F_sc - F_sc function for phonon frequency sampling in the cell, which depends on the temperature of the cell and phonon dispersion
	cells.Polarize - Polarize function for phonon branch sampling in the cell, which depends on the temperature of the cell, phonon dispersion, and scattering prbability.
	cells.Psc - scattering probability of each phonon mode, which depends on the temperature of the cell.
	cells.Np - number of phonons in the cell
	*/
	printout = "System initialization";
	left = int((totalsize - printout.size()) / 2); right = totalsize - left - printout.size();
	if (rank == 0)
	{
		std::fprintf(out_file, "\n%s\n", (std::string(left, '-') + printout + std::string(right, '-')).c_str()); //Formatting blocks
		std::fprintf(out_file, "Initializing the system...\n");
	}

	start = MPI_Wtime();
	for (int s = 0; s < Nz; s++ )
	{
		cells[s].temp = (s == 0) ? T_hot : T_cold; //initialize temperature of each cell, hot cell is initialized to T_hot while all the other cells are initialized to T_cold.
		cell_np[s] = 0;
		double ntot, tot = 0; //some dummy variables

		for (i = 0; i < Nq; i++)
		{
			double tmp = 0;
			for (j = 0; j < Nbranch; j++)
			{
				if (omega[i][j] > 0)
				{
					ndensity[i][j] = Vu / V_FC / Nq * feq(omega[i][j], cells[s].temp); //mode-dependent # of phonons in the simulation cell
					cell_np[s] +=  Vu / V_FC / Nq * feq(omega[i][j], cells[s].temp); //This is real # of phonons in a simulation cell

					E[i][j] = hbar * omega[i][j] *  ndensity[i][j];
					//Mode-dependent phonon energy per simulation cell
					tot += E[i][j]; //energy of the cell
				}
				else
				{
					E[i][j] = 0;
					ndensity[i][j] = 0;
				}
				tmp += ndensity[i][j];
			}
		}

		Np_ToT += cell_np[s]; //Total # of phonons in the simulation domain (physical)
		calc_CPDFs(omega, Psc, cells[s].temp, cells[s], ntot); //calculate CPDFs using the current cell state variables
	}
	if (rank == 0) std::fprintf(out_file, "Total number of phonons in the entire simulation domain: %0.2e.\n", Np_ToT);

	// FILE *kvec = fopen("Fsc.dat", "w");
	// FILE *branch = fopen("Pol.dat", "w");
	// for (i = 0; i < Nq; i++)
	// {
	// 	fprintf(kvec, "%0.5f \n", cells[8].F_sc[i]);
	// 	for (j = 0; j < Nbranch; j++)
	// 	{
	// 		fprintf(branch, "%0.4f  ", cells[1].Polarize[i][j]);
	// 	}
	// 	fprintf(branch, "\n");
	// }
	// std::fclose(kvec);
	// std::fclose(branch);

	//Recalculate phonon numbers in each cell and the entire simulation domain
	for (int s = 0; s < Nz; s++ )
	{
		cells[s].Np = int(cell_np[s] / W); //scale # of phonons in each cell by the scaling factor
		Np_tot += cells[s].Np;
	}
	Np_hot = cells[0].Np;
	Np_cold = cells[Nz - 1].Np; //The number of phonons in hot and cold cells at two ends will be set to these two numbers at the end of each timestep, in order to keep their temperature fixed
	if (rank == 0)
	{
		std::fprintf(out_file, "Using the Scaling factor W = %d , the total number of phonons is now %d.\n", int(W), Np_tot);
		std::fprintf(out_file, "Number of phonons in the hot cell: %d.\n", Np_hot);
		std::fprintf(out_file, "Number of phonons in the cold cell: %d.\n", Np_cold);
	}

	//**************************************Initialize phonons******************************************************
	//Information of phonons are stored as a vector "Phonons"
	std::vector<Phonon> Phonons(Np_tot);
	/*The phonon state is defined by a number of variables.
	Phonons.temp - Since phonons are populated from CPDFs defined by a cell state, this temperature here stores the corresponding temperature of that cell state.
	Phonons.omega - phonon frequency
	Phonons.kvector - the index of the phonon wavevector in the matrix "wavevector"
	Phonons.branch - the phonon branch index
	Phonons.velocity - a 3-element array storing phonon group velocity along x-, y-, and z-directions
	Phonons.cell - the index of cell in which the phonon is located
	Phonons.position - a 3-element array storing location of the phonon, namely x-, y-, and z-coordinates
	*/
	// MTRand mtrand(rank + time(NULL)); //Seed random number generator
	MTRand mtrand(20); //Seed random number generator
	// std::cout<<mtrand()<<"\n";

	std::vector<double> kappa_x, kappa_y, kappa_z;
	std::vector< std::vector<double> > Temp(Nstep, std::vector<double>(Nz));

	//*****************Create datatype for comunicating phonons across processors *******************************************
	//Define the MPI phonon data type
	MPI_Datatype phonon_type;
	int lengths[11];
	for (int i = 0; i < 11; i++) lengths[i] = 1;
	//Prepare inputs to the MPI datatype constructor
	MPI_Aint base_address;
	MPI_Aint displacements[11];
	MPI_Get_address(&Phonons[0], &base_address);
	MPI_Get_address(&Phonons[0].temp, &displacements[0]);
	MPI_Get_address(&Phonons[0].omega, &displacements[1]);
	MPI_Get_address(&Phonons[0].branch, &displacements[2]);
	MPI_Get_address(&Phonons[0].kvector, &displacements[3]);
	MPI_Get_address(&Phonons[0].velocity[0], &displacements[4]);
	MPI_Get_address(&Phonons[0].velocity[1], &displacements[5]);
	MPI_Get_address(&Phonons[0].velocity[2], &displacements[6]);
	MPI_Get_address(&Phonons[0].cell, &displacements[7]);
	MPI_Get_address(&Phonons[0].position[0], &displacements[8]);
	MPI_Get_address(&Phonons[0].position[1], &displacements[9]);
	MPI_Get_address(&Phonons[0].position[2], &displacements[10]);
	for (int i = 0; i < 11; i++) displacements[i] = MPI_Aint_diff(displacements[i], base_address);
	MPI_Datatype types[11] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
	MPI_Type_create_struct(11, lengths, displacements, types, &phonon_type);
	MPI_Type_commit(&phonon_type);
	//Now the "phonon_type" can be used in MPI to distribute phonon vector among processors.

	int tsize;
	MPI_Type_size ( phonon_type, &tsize );
	// **************************************************************************************

	int sendcount[ntask], disp[ntask];

	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();

	// FILE *ph_file = fopen(string(out_path + "ph_init.dat").c_str(), "w");
	// for (size_t i = 0; i < Phonons.size(); i++) {if (rank == 0) print_phonon(Phonons[i], ph_file);}
	std::vector<Phonon> rbuf;
	int idx_start = 0;
	double local_energy;
	for (s = 0; s < Nz; s++)
	{
		// cells[s].Energy = 0; //set energy in each cell to zero
		local_energy = 0;

		vector_distribute(ntask, cells[s].Np, sendcount, disp);
		rbuf.resize(sendcount[rank]);
		// MPI_Scatterv(&Phonons[idx_start], sendcount, disp, phonon_type, &rbuf[0], sendcount[rank], phonon_type, 0, MPI_COMM_WORLD);
		for (i = 0; i < sendcount[rank]; i++)
		{
			set_phonon(rbuf[i], mtrand, cells[s], omega, velocity, s, 1); //Sample the phonon frequency, wavevector, branch, position, temperature, and group velocity from the cell state
			// print_phonon(rbuf[i], ph_file);
			local_energy += hbar * rbuf[i].omega; //The phonon contribute to total energy of the cell
		}
		MPI_Gatherv(&rbuf[0], sendcount[rank], phonon_type, &Phonons[idx_start], sendcount, disp, phonon_type, 0, MPI_COMM_WORLD);
		MPI_Reduce(&local_energy, &cells[s].Energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		// if (rank == 0)
		// {
		// 	dummy = calc_cell_temp(cells[s].temp - 60 , cells[s].temp + 60 , cells[s].Energy, omega);

		// 	if (cells[s].temp - dummy >= 10) //If the pseudo-temperature differs from the prescribed temperature too much (>10 K), it means either the phonon is under sampled or the qpoint mesh is not dense enough to represent the full BZ.
		// 	{
		// 		std::fprintf(out_file, "Warning: In the %dth cell, the pseudo-temperature of the phonons populated is too far away from the target temperature, consider increasing number of the phonons or using a denser qpoint mesh.\n", s + 1);
		// 	}
		// }
		idx_start += cells[s].Np;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();
	Time = end - start;


// FILE *init_file = fopen(string(out_path + "ph_init.dat").c_str(), "w");
// for (size_t i = 0; i < Phonons.size(); i++) {if (rank == 0) print_phonon(Phonons[i], init_file);}
// fclose(init_file);
// FILE *init_file = fopen(string(out_path + "ph_init.dat").c_str(), "w");
// 	for (size_t i = 0; i < Phonons.size(); i++) {if (rank == 0) print_phonon(Phonons[i], init_file);}
// 	fclose(init_file);
// fclose(ph_file);
// int what = 0;
// for ( size_t i = 0; i < Phonons.size(); i++)
// {
// 	if (Phonons[i].omega != 0) what++;
// 	// std::cout<<Phonons[i].omega<<"\n";
// }
// std::cout << what << "\n";
// FILE *freq_file = fopen(string(out_path + "sample_omega.dat").c_str(), "w");
// for (s = 0; s < Nz; s++)
// {
// 	cells[s].Energy = 0; //set energy in each cell to zero
// 	for (i = 0; i < cells[s].Np; i++)
// 	{
// 		Phonons.push_back(Phonon()); //Create a phonon
// 		Phonons.back().temp = cells[s].temp; //temperature of the cell state is assigned as the temperature of the phonon
// 		set_phonon(Phonons.back(), mtrand, cells[s], omega, velocity, s); //Sample the phonon frequency, wavevector, branch, position, temperature, and group velocity from the cell state
// 		// fprintf(freq_file, "%10.3e\n", Phonons[Phonons.size() - 1].omega);
// 		cells[s].Energy += hbar * Phonons.back().omega; //The phonon contribute to total energy of the cell
// 	}
// 	dummy = calc_cell_temp(cells[s].temp - 60 , cells[s].temp + 60 , cells[s].Energy, omega);
// 	if (cells[s].temp - dummy >= 10) //If the pseudo-temperature differs from the prescribed temperature too much (>10 K), it means either the phonon is under sampled or the qpoint mesh is not dense enough to represent the full BZ.
// 	{
// 		if (rank == 0) std::fprintf(out_file, "Warning: In the %dth cell, the pseudo-temperature of the phonons populated is too far away from the target temperature, consider increasing number of the phonons or using a denser qpoint mesh.\n", s + 1);
// 	}
// }
	if (rank == 0)
	{
		real_T_hot = calc_cell_temp(cells[0].temp - 60 , cells[0].temp + 60 , cells[0].Energy, omega);
		// std::cout<<real_T_hot<<"\n";
		real_T_cold = calc_cell_temp(cells[Nz - 1].temp - 60 , cells[Nz - 1].temp + 60 , cells[Nz - 1].Energy, omega); //We store the initial pseudo-temperature of the hot and cold cell seperately, since they don't change as the simulation advances in time whereas the temperatures of all the other interior cells are changing.
		std::fprintf(out_file, "Finish populating phonons in all the cells.\n");
		std::fprintf(out_file, "The phonon initialization costs time: %0.2f s\n", Time);
		std::fprintf(out_file, "%s\n", std::string(totalsize, '-').c_str());
	}
			if (rank == 0 )
			{
				FILE *test_file = fopen(string(out_path + "phonon.dat").c_str(), "w");
				for (size_t i = 0; i < Phonons.size() ; i++)
				{fprintf(test_file, "%d", int(i) + 1); print_phonon(Phonons[i], test_file);}
				fclose(test_file);
			}
	//*********************************************Enter timesteps************************************************
	//Format output file
	if (rank == 0)
	{
		//Formatting blocks
		printout = "Advance in time";
		left = int((totalsize - printout.size()) / 2); right = totalsize - left - printout.size();
		std::fprintf(out_file, "\n%s\n", (std::string(left, '-') + printout + std::string(right, '-')).c_str());

		//Format output "temperature.dat" file
		std::fprintf(T_file, "%10s%10s", "Timestep", "Time(ns)");
		for (i = 0; i < Nz; i++) std::fprintf(T_file, "%10s", std::string("cell[" + std::to_string(i + 1) + "]").c_str());
		std::fprintf(T_file, "\n");

		//Format output "temperature_avg.dat" file
		std::fprintf(T_avg_file, "%10s%10s", "Timestep", "Time(ns)");
		for (i = 0; i < Nz; i++) std::fprintf(T_avg_file, "%10s", std::string("cell[" + std::to_string(i + 1) + "]").c_str());
		std::fprintf(T_avg_file, "\n");

		//Format output "flux.dat" file
		std::fprintf(flux_file, "%10s%10s%15s%15s%15s\n", "Timestep", "Time(ns)", "flux_x(W/m^2)", "flux_y(W/m^2)", "flux_z(W/m^2)");

		//Format output "kappa.dat" file
		std::fprintf(kappa_file, "%10s%10s%20s%10s%20s%10s%20s%10s\n", "Timestep", "Time(ns)", "kappa_x(W/m/K)", "STD", "kappa_y(W/m/K)", "STD", "kappa_z(W/m/K)", "STD");

		//Format the "PhononNumber.dat file"
		std::fprintf(Nfile, "%*s%*s", 10, "Timestep", 10, "Time(ns)");
		for (int i = 0; i < Nz; i++) std::fprintf(Nfile, "%10s", std::string("Cell[" + to_string(i + 1) + "]").c_str());
		std::fprintf(Nfile, "%10s\n", "Ntot");
	}

	int Nint_before, local_escape, N_escape, DISP[ntask], NUM[ntask], acc;
	double time_drift;
	for (i = 0; i < ntask; i++) {NUM[i] = 1; DISP[i] = i;}
	for (m = 0; m < Nstep; m++)
	{
		if (((m + 1) % Nprint == 0 || m == Nstep - 1 || m == 0) && rank == 0) std::fprintf(out_file, "timestep %d:\n", m + 1);
		//clear cell energy and number of phonons in each cell before start drifting
		for (int i = 0; i < Nz; i++ )
		{
			cells[i].Energy = 0.0E0;
			cells[i].Np = 0;
		}
		//**************************************Drift Phonons***********************************************
		//**************************************************************************************************

		// //%%%%%%%%%%%%%%%%%%%%%%%%%%%Parallel version%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// Nint_before = Phonons.size() - Np_hot - Np_cold;
		// vector_distribute(ntask, int(Phonons.size()), sendcount, disp);
		// rbuf.resize(sendcount[rank]);
		// start = MPI_Wtime();
		// MPI_Scatterv(&Phonons[0], sendcount, disp, phonon_type, &rbuf[0], sendcount[rank], phonon_type, 0, MPI_COMM_WORLD);

		// local_escape = 0;
		// double local_cell_energy[Nz] = {0}, global_cell_Energy[Nz];
		// int local_cell_np[Nz] = {0}, global_cell_np[Nz];
		// // MPI_Barrier(MPI_COMM_WORLD);

		// for ( size_t i = 0; i < rbuf.size();)
		// {
		// 	// print_phonon(rbuf[i], init_file);
		// 	phonon_drift(rbuf[i]);
		// 	//If the phonon is in the cells at two ends, it is removed; otherwise update cell energy and phonon number
		// 	if ((rbuf[i].cell == 0) || (rbuf[i].cell == (Nz - 1)))
		// 	{
		// 		if (i != rbuf.size() - 1)
		// 		{
		// 			std::swap(rbuf[i], rbuf.back());
		// 		}
		// 		rbuf.pop_back();
		// 		local_escape += 1;
		// 		sendcount[rank]--;
		// 	}
		// 	else
		// 	{
		// 		local_cell_energy[rbuf[i].cell] +=  hbar * rbuf[i].omega;
		// 		local_cell_np[rbuf[i].cell] += 1;
		// 		i++;
		// 	}
		// }
		// // MPI_Barrier(MPI_COMM_WORLD);
		// if (rank == 0) std::printf("%d, time: %0.5f s\n", m, MPI_Wtime() - start);
		// MPI_Gatherv(&sendcount[rank], 1, MPI_INT, &sendcount[0], NUM, DISP, MPI_INT, 0, MPI_COMM_WORLD);
		// acc = 0;
		// disp[0] = 0;
		// if (ntask > 1)
		// {	for (i = 1; i < ntask; i++)
		// 	{
		// 		disp[i] = 0;
		// 		acc += sendcount[i - 1];
		// 		disp[i] += acc;
		// 	}
		// }
		// MPI_Reduce(&local_escape, &N_escape, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		// MPI_Reduce(&local_cell_energy, &global_cell_Energy, Nz, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		// MPI_Reduce(&local_cell_np, &global_cell_np, Nz, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		// for (i = 0; i < Nz; i++)
		// {
		// 	cells[i].Energy = global_cell_Energy[i];
		// 	cells[i].Np = global_cell_np[i];
		// 	// if (rank==0) printf("%0.3e\n",cells[i].Energy);
		// }
		// MPI_Bcast(&N_escape, 1, MPI_INT, 0, MPI_COMM_WORLD);
		// int tmp = Nint_before + Np_hot + Np_cold - N_escape;
		// Phonons.clear();
		// Phonons.resize(tmp);
		// MPI_Gatherv(&rbuf[0], sendcount[rank], phonon_type, &Phonons[0], sendcount, disp, phonon_type, 0, MPI_COMM_WORLD);
		// // FILE *drift_file = fopen(string(out_path + "ph_drift.dat").c_str(), "w");
		// // for (size_t i = 0; i < Phonons.size(); i++) {if (rank == 0) print_phonon(Phonons[i], drift_file);}
		// // fclose(drift_file);
		// // MPI_Barrier(MPI_COMM_WORLD);
		// end = MPI_Wtime();
		// double time_drift = end - start;
		// if (rank == 0) std::printf("%d The drift time: %0.5f s\n", m, time_drift);
		// // if (rank == 0) std::printf("The drift time: %0.8f s\n", time_drift);
		// // FILE *drift_file = fopen(string(out_path + "ph_drift.dat").c_str(), "w");
		// // for (size_t i = 0; i < Phonons.size(); i++) {if (rank == 0) print_phonon(Phonons[i], drift_file);}
		// // fclose(drift_file);
		// // int N_escape = 0;
		// // for (size_t i = 0; i < Phonons.size();)
		// // {
		// // 	//If the phonon is in the cells at two ends, it is removed; otherwise update cell energy and phonon number
		// // 	if ((Phonons[i].cell == 0) || (Phonons[i].cell == (Nz - 1)))
		// // 	{
		// // 		if (i != (Phonons.size() - 1))
		// // 		{
		// // 			std::swap(Phonons[i], Phonons.back());
		// // 		}
		// // 		Phonons.pop_back();
		// // 		N_escape += 1;
		// // 	}
		// // 	else
		// // 	{
		// // 		cells[Phonons[i].cell].Energy +=  hbar * Phonons[i].omega;
		// // 		cells[Phonons[i].cell].Np += 1;
		// // 		i++;
		// // 	}
		// // }
		// // std::cout << Phonons.size() << "\n";
		// //%%%%%%%%%%%%%%%%%%%%%%%%%%%Parallel version%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%Serial version%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		N_escape = 0;
		Nint_before = Phonons.size() - Np_hot - Np_cold;
		start = MPI_Wtime();
		for ( size_t i = 0; i < Phonons.size();)
		{
			phonon_drift(Phonons[i]);
			//If the phonon are in the cells at two ends, it is removed; otherwise update cell energy and phonon number
			if ((Phonons[i].cell == 0) || (Phonons[i].cell == (Nz - 1)))
			{
				if (i != (Phonons.size() - 1))
				{
					std::swap(Phonons[i], Phonons[Phonons.size() - 1]);
				}
				Phonons.pop_back();
				N_escape += 1;
			}
			else
			{
				cells[Phonons[i].cell].Energy +=  hbar * Phonons[i].omega;
				cells[Phonons[i].cell].Np += 1;
				i++;
			}
		}
		end = MPI_Wtime();
		time_drift = end - start;
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%Serial version%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		if (((m + 1) % Nprint == 0 || m == Nstep - 1 || m == 0) && rank == 0) std::fprintf(out_file, "There are %d phonons that enter the two end cells, therefore number of phonons in the interior cells reduces from %d to %d.\n", N_escape, Nint_before, int(Phonons.size()));




		// ***************************Calculate new temperature and update CPDFs of the interior cells ****************
		// ************************************************************************************************************
		MPI_Barrier(MPI_COMM_WORLD);
		start = MPI_Wtime();

		if (((m + 1) % Nprint == 0 || m == Nstep - 1 || m == 0) && rank == 0)
		{
			std::fprintf(T_file, "%10d%10.3f%10.2f", m + 1, dt * (m + 1) * 1E9, real_T_hot);
			std::fprintf(Nfile, "%10d%10.3f%10d", m + 1, dt * (m + 1) * 1E9, Np_hot);
		}

		Temp[m][0] = real_T_hot; Temp[m][Nz - 1] = real_T_cold;

		// int send_cell[ntask], disp_cell[ntask];
		// vector_distribute(ntask, Nz-2, send_cell, disp_cell);
		// double tmp_cellT[Nz-2],tmp_cellE[Nz-2],Tbuf[send_cell[rank]],Ebuf[send_cell[rank]],local_cellT[send_cell[rank]];
		// for (int s = 0; s < Nz - 2; s++)
		// {
		// 	tmp_cellT[s]=cells[s+1].temp;
		// 	tmp_cellE[s]=cells[s+1].Energy;
		// }

		// MPI_Scatterv(&tmp_cellT[0], send_cell, disp_cell, MPI_DOUBLE, &Tbuf[0], send_cell[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		// MPI_Scatterv(&tmp_cellE[0], send_cell, disp_cell, MPI_DOUBLE, &Ebuf[0], send_cell[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// for (int s = 0; s < send_cell[rank]; s++)
		// {
		// 	local_cellT[s] = calc_cell_temp(Tbuf[s] - 60 , Tbuf[s] + 60 , Ebuf[s], omega);
		// }
		// MPI_Gatherv(&local_cellT[0], send_cell[rank], MPI_DOUBLE, &tmp_cellT[0], send_cell, disp_cell, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// for (int s = 0; s < Nz - 2; s++)
		// {
		// 	cells[s+1].temp=tmp_cellT[s];
		// }
		double tmp_T, tmp_RT;
		for (int s = 1; s < Nz - 1; s++) //Only the temperature of interior cells need to be updated, the hot and cold cells are held at fixed temperature.
		{
			//Record number of phonons of each cell in "PhononNumber.dat"
			if (((m + 1) % Nprint == 0 || m == Nstep - 1 || m == 0) && rank == 0) std::fprintf(Nfile, "%10d", cells[s].Np);
			if (rank == 0) cells[s].temp = calc_cell_temp(cells[s].temp - 60 , cells[s].temp + 60 , cells[s].Energy, omega);
			MPI_Bcast(&cells[s].temp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			Temp[m][s] = cells[s].temp;
			if (((m + 1) % Nprint == 0 || m == Nstep - 1 || m == 0) && rank == 0) fprintf(T_file, "%10.2f", cells[s].temp);

			/*The RTs of the phonons in the cell have to be calculated at the cell temperature, which not neccessarily can be extracted from the existing RT data obtained from ShengBTE. So an interpolation between available data points is adopted to determine the RT at arbitrary temperature. First, the temperature range where the cell temperature is located is found. Then a simple linear interpolation is used to calculate the RTs at the specific cell temperature from the RTs data at the two temperature endpoints of the range*/
			k = 0;
			tmp_T = cells[s].temp;

			if (nT > 1) //If more than 1 temperature RT data file is available, the interpolation scheme is implemented.
			{
				if (tmp_T <= double(temperature[0]))
				{
					for (i = 0; i < Nq; i++)
					{
						for (j = 0; j < Nbranch; j++)
						{
							Psc[i][j] = (1 - exp(-dt / RT[0][i][j]));
						}
					}
				}
				else if (tmp_T >= double(temperature[nT - 1]))
				{
					for (i = 0; i < Nq; i++)
					{
						for (j = 0; j < Nbranch; j++)
						{
							Psc[i][j] = (1 - exp(-dt / RT[nT - 1][i][j]));
						}
					}
				} //if the temperature of the cell exceeds the temperature range of RT data, use the closest available data. BE CAUTIOUS! Is this really what you want?
				else while (tmp_T >= double(temperature[k]))
					{
						k += 1;
						//the index k stores the range: temperature[k-1]<cells.temp<temperature[k]

						for (i = 0; i < Nq; i++)
						{
							for (j = 0; j < Nbranch; j++)
							{
								tmp_RT = RT[k - 1][i][j] + ( tmp_T - double(temperature[k - 1])) * (RT[k][i][j] - RT[k - 1][i][j]) / (double(temperature[k]) - double(temperature[k - 1]));
								Psc[i][j] = (1 - exp(-dt / tmp_RT));
								// if (i==15&&j==2&&s==1)
								// {
								// 	std::cout<<tmp_RT<<" "<<RT[k - 1][i][j]<<" "<<RT[k][i][j];
								// }
							}
						}
					}
			}
			else //If there is only 1 RT data file, it is used as the RTs at different temperatures.
			{
				for (i = 0; i < Nq; i++)
				{
					for (j = 0; j < Nbranch; j++)
					{
						Psc[i][j] = (1 - exp(-dt / RT[0][i][j]));
					}
				}
			}
			cells[s].Psc = Psc; //assign scattering probability matrix "Psc" to the cell state variable "cells.Psc".
			calc_CPDFs(omega, Psc, cells[s].temp, cells[s], dummy); //update CPDFs of the cell using its state variables, the dummy variable returns the physical number of phonons in the cell. The dummy variable is only used in the initialization step which we do not need anymore.
		}
		if (((m + 1) % Nprint == 0 || m == Nstep - 1 || m == 0) && rank == 0)
		{
			std::fprintf(out_file, "Finished updating the temperature and CPDFs of interior cells.\n");
			std::fprintf(T_file, "%10.2f\n", real_T_cold);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		end = MPI_Wtime();
		double time_updatecell = end - start;



		// *********************************************Scatter phonons*********************************************
		// ************************************************************************************************************
		std::string scatter = "true";
		if (scatter == "true")
		{
			MPI_Barrier(MPI_COMM_WORLD);
			start = MPI_Wtime();
			double rnd;
			int Np_scatter = 0; //record how many phonons are scattered
			int Nphonons = int(Phonons.size());
			MPI_Bcast(&Nphonons, 1, MPI_INT, 0, MPI_COMM_WORLD);
			vector_distribute(ntask, Nphonons, sendcount, disp);
			rbuf.clear();
			rbuf.resize(sendcount[rank]);
			// if (rank == 0 && m == 0)
			// {
			// 	FILE *test_file = fopen(string(out_path + "phonon.dat").c_str(), "w");
			// 	for (size_t i = 0; i < Phonons.size() ; i++)
			// 	{fprintf(test_file, "%d", int(i) + 1); print_phonon(Phonons[i], test_file);}
			// 	fclose(test_file);
			// }
			MPI_Scatterv(&Phonons[0], sendcount, disp, phonon_type, &rbuf[0], sendcount[rank], phonon_type, 0, MPI_COMM_WORLD);
			if (rank == 1&&m==0)
			{
				FILE *kk_file = fopen(string(out_path + "phonon" + to_string(rank) + ".dat").c_str(), "w");
				for (size_t i = 0; i < rbuf.size() ; i++)
				{fprintf(kk_file, "%d", int(i) + 1); print_phonon(rbuf[i], kk_file);}
				fclose(kk_file);
			}

			int local_scatter = 0;

			// FILE *test_file = fopen(string(out_path + "rnd.dat").c_str(), "w");
			for (size_t i = 0; i < rbuf.size() ; i++)
			{
				double tmp[3];
				rnd = mtrand.rand(); //generate a random number
				// if (rank==0) fprintf(test_file,"%0.5f\n",rnd);
				rbuf[i].temp = cells[rbuf[i].cell].temp; //update temperature of phonon according to the cell state
				// If the random number rnd is smaller than the Scattering probability, the phonon is scattered. Therefore, the phonon's omega, kvector, branch, and velocity are resampled using the CPDFs of cell. The phonon's position is unchanged.

				if (rnd < cells[rbuf[i].cell].Psc[rbuf[i].kvector][rbuf[i].branch])
				{
					for (j = 0; j < 3; j++) tmp[j] = rbuf[i].position[j];
					set_phonon(rbuf[i], mtrand, cells[rbuf[i].cell], omega, velocity, rbuf[i].cell, 1);
					for (j = 0; j < 3; j++) rbuf[i].position[j] = tmp[j];
					local_scatter += 1;
				}
			}
			MPI_Gatherv(&rbuf[0], sendcount[rank], phonon_type, &Phonons[0], sendcount, disp, phonon_type, 0, MPI_COMM_WORLD);
			MPI_Reduce(&local_scatter, &Np_scatter, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			// for (size_t i = 0; i < Phonons.size() ; i++)
			// {
			// 	double tmp[3];
			// 	rnd = mtrand.rand(); //generate a random number
			// 	Phonons[i].temp = cells[Phonons[i].cell].temp; //update temperature of phonon according to the cell state
			// // 	If the random number rnd is smaller than the Scattering probability, the phonon is scattered. Therefore, the phonon's omega, kvector, branch, and velocity are resampled using the CPDFs of cell. The phonon's position is unchanged.

			// 	if (rnd < cells[Phonons[i].cell].Psc[Phonons[i].kvector][Phonons[i].branch])
			// 	{
			// 		for (j = 0; j < 3; j++) tmp[j] = Phonons[i].position[j];
			// 		set_phonon(Phonons[i], mtrand, cells[Phonons[i].cell], omega, velocity, Phonons[i].cell);
			// 		for (j = 0; j < 3; j++) Phonons[i].position[j] = tmp[j];
			// 		Np_scatter += 1;
			// 	}
			// }
			MPI_Barrier(MPI_COMM_WORLD);
			end = MPI_Wtime();
			double time_scatter = end - start;
			if (((m + 1) % Nprint == 0 || m == Nstep - 1 || m == 0) && rank == 0)
			{
				std::fprintf(out_file, "%d out of %d phonons are scattered.\n", Np_scatter, int(Phonons.size()));
				std::fprintf(out_file, "The drift time: %0.5f s\n", time_drift);
				std::fprintf(out_file, "The scatter time: %0.5f s\n", time_scatter);
				std::fprintf(out_file, "The updating cell state time: %0.5f s\n", time_updatecell);
			}
		}
		// FILE *scatter_file = fopen(string(out_path + "ph_scatter.dat").c_str(), "w");
		// for (size_t i = 0; i < Phonons.size(); i++) {if (rank == 0) print_phonon(Phonons[i], scatter_file);}
		// fclose(scatter_file);


		//**********************************Calculate Flux*****************************************************
		//*****************************************************************************************************
		flux[0] = 0.0E0; flux[1] = 0.0E0; flux[2] = 0.0E0; //clear the flux data from last timestep

		for (j = 0; j < 3; j++)
		{
			for (size_t i = 0; i < Phonons.size() ; i++)
			{
				if (Phonons[i].cell != 0 && Phonons[i].cell != Nz - 1) //Only include phonons in the interior cells for flux calculation
				{
					flux[j] += 1 / (Vtot - 2 * Vu) * W * hbar * Phonons[i].omega * Phonons[i].velocity[j]; //Heat flux per unit area along three directions
				}
			}
		}

		//**********************************Repopulate the hot and cold cells***********************************
		//******************************************************************************************************
		/*Fill in the hot and cold cells with phonons, since in the drift process all phonons in these cells are deleted. These phonons will be drifting into the interior cells at the next timestep, making the constant-temperature cells at two ends serve as energy reservoirs*/
		for (i = 0; i < Np_hot; i++)
		{
			Phonons.push_back(Phonon());
			set_phonon(Phonons.back(), mtrand, cells[0], omega, velocity, 0, m);
			cells[0].Energy += hbar * Phonons.back().omega;
		}

		for (i = 0; i < Np_cold; i++)
		{
			Phonons.push_back(Phonon());
			set_phonon(Phonons.back(), mtrand, cells[Nz - 1], omega, velocity, Nz - 1, m);
			cells[Nz - 1].Energy += hbar * Phonons.back().omega;
		}

		if (rank == 0)
		{
			if (((m + 1) % Nprint == 0 || m == 0 ) && m != Nstep - 1)
			{
				std::fprintf(out_file, "Finish repopulating the hot and cold cells.\n");
			}
			else if (m == Nstep - 1) std::fprintf(out_file, "%s\n", std::string(totalsize, '-').c_str());
		}
		if (((m + 1) % Nprint == 0 || m == Nstep - 1 || m == 0) && rank == 0) std::fprintf(Nfile, "%10d%10d\n", Np_cold, (int)Phonons.size()); //Be careful with the type cast (int) size_type. For a std::vector, .size() method will return a size_type or size_t which is usually an unsigned 64-bit/32-bit integer, depending on the machince architecture. There is a chance that this number will exceed the int type can hold: a signed 32-bit integer.




		kappa_z.push_back( flux[2] * Nz * Lz / (real_T_hot - real_T_cold)) ; //Thermal conductivity along the z-direction
		kappa_x.push_back( flux[0] * Lx / (real_T_hot - real_T_cold)) ; //Thermal conductivity along the x-direction
		kappa_y.push_back( flux[1] * Ly / (real_T_hot - real_T_cold)) ; //Thermal conductivity along the y-direction

		//Print flux data to "flux.dat"
		if (((m + 1) % Nprint == 0 || m == Nstep - 1 || m == 0) && rank == 0) std::fprintf(flux_file, "%10d%10.3f%15.3e%15.3e%15.3e\n", m + 1, dt * (m + 1) * 1E9, flux[0], flux[1], flux[2]);

		std::vector<double> tmp_kappa, tmp_Temp;
		double kappa_avg, kappa_std, temp_avg, temp_std;
		if (m == 0 && rank == 0)
		{
			std::fprintf(kappa_file, "%10d%10.3f%20.2f%10d%20.2f%10d%20.2f%10d\n", m + 1, dt * (m + 1) * 1E9, kappa_x[m], 0, kappa_y[m], 0, kappa_z[m], 0);

			std::fprintf(T_avg_file, "%10d%10.3f", m + 1, dt * (m + 1) * 1E9);
			for (s = 0; s < Nz; s++) std::fprintf(T_avg_file, "%10.2f", Temp[m][s]);
			std::fprintf(T_avg_file, "\n");
		}
		else if (((m + 1) % Nout == 0 || m == Nstep - 1) && rank == 0)
		{
			std::fprintf(kappa_file, "%10d%10.3f", m + 1, dt * (m + 1) * 1E9);

			tmp_kappa = std::vector<double> (kappa_y.end() - Navg, kappa_y.end());
			SD(kappa_avg, kappa_std, tmp_kappa);
			// kappa_avg_stream << std::setw(20) << kappa_avg << std::setw(10) << kappa_std;
			std::fprintf(kappa_file, "%20.2f%10.3f", kappa_avg, kappa_std);

			tmp_kappa = std::vector<double> (kappa_y.end() - Navg, kappa_y.end());
			SD(kappa_avg, kappa_std, tmp_kappa);
			// kappa_avg_stream << std::setw(20) << kappa_avg << std::setw(10) << kappa_std;
			std::fprintf(kappa_file, "%20.2f%10.3f", kappa_avg, kappa_std);

			tmp_kappa = std::vector<double> (kappa_z.end() - Navg, kappa_z.end());
			SD(kappa_avg, kappa_std, tmp_kappa);
			std::fprintf(kappa_file, "%20.2f%10.3f\n", kappa_avg, kappa_std);

			// Take average and standard deviation of temperature between timestep "m-Navg+1" and "m"
			std::fprintf(T_avg_file, "%10d%10.3f", m + 1, dt * (m + 1) * 1E9);
			for (s = 0; s < Nz; s++)
			{
				tmp_Temp.clear();
				for (i = m - Navg + 1; i < m + 1; i++)
				{
					tmp_Temp.emplace_back(Temp[i][s]);
				}
				SD(temp_avg, temp_std, tmp_Temp);
				std::fprintf(T_avg_file, "%10.2f", temp_avg);
			}
			std::fprintf(T_avg_file, "\n");
		}
		end = MPI_Wtime();
		Time = end - g_start;
		if (rank == 0) std::fprintf(out_file, "Total time spent: %0.2f s\n\n", Time);
	}
	if (rank == 0) std::fprintf(out_file, "\nSimulation done...\n");

//close the files.
	if (rank == 0)
	{
		std::fclose(Nfile);
		std::fclose(out_file);
		std::fclose(flux_file);
		std::fclose(kappa_file);
		std::fclose(T_file);
		std::fclose(T_avg_file);
	}

	MPI_Finalize();
}
