#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
// #include <fstream>
// #include <string>
// #include <cstdio>
// #include <cstdlib>
// #include "Input.h"
#include <mpi.h>
#include <vector>

using namespace std;
//***********************************Declare global variables so that all the function can use them**********************
// int i, j, k, s, Nir, Nq, Natom, Nbranch, Nfull, Nz,  Nstep, Nout, Navg, Nprint, Np_hot, Np_cold;
// double Lx, Ly, Lz, Vu, V_FC, W, dt, Vtot, T_hot, T_cold;

// double flux[3];

//***************************************************End global variables**********************************************

struct Phonon
{
  double temp;
  double omega;
  double velocity[3];
  int cell;
};

int main (int argc, char **argv)
{
  int size, rank, numtask;
  std::vector<Phonon> Phonons;

  // Initialize MPI.

  int ierr = MPI_Init ( &argc, &argv );

  if ( ierr != 0 )
  {
    cout << "MONTE_CARLO - Fatal error!\n";
    cout << "  MPI_Init returned ierr = " << ierr << "\n";
    exit ( 1 );
  }
  MPI_Comm_size(MPI_COMM_WORLD, &numtask); //get # of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); //call the processes
  std::string out_path = "./output"+to_string(rank);
  mkdir(out_path.c_str(), 0777);
  //create MPI datatype
  MPI_Datatype phonon_type;
  int lengths[6] = {1, 1, 1, 1, 1, 1};
  //Prepare inputs to the MPI datatype constructor
  MPI_Aint base_address;
  MPI_Aint displacements[6];
  MPI_Get_address(&Phonons[0], &base_address);
  MPI_Get_address(&Phonons[0].temp, &displacements[0]);
  MPI_Get_address(&Phonons[0].omega, &displacements[1]);
  MPI_Get_address(&Phonons[0].velocity[0], &displacements[2]);
  MPI_Get_address(&Phonons[0].velocity[1], &displacements[3]);
  MPI_Get_address(&Phonons[0].velocity[2], &displacements[4]);
  MPI_Get_address(&Phonons[0].cell, &displacements[5]);
  for (int i = 0; i < 5; i++) displacements[i] = MPI_Aint_diff(displacements[i], base_address);


  MPI_Datatype types[6] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
  MPI_Type_create_struct(6, lengths, displacements, types, &phonon_type);
  MPI_Type_commit(&phonon_type);

  int tsize;
  Phonon sbuf;
  // MPI_Aint ext;
  // MPI_Type_extent ( phonon_type, &ext);
  MPI_Type_size ( phonon_type, &tsize );
  // std::cout << ext << "\n";
  // std::cout << tsize << "\n";

  int Nz = 10;
  if (rank == 0)
  {

    for (int s = 1; s < Nz + 1; s++)
    {
      Phonons.push_back(Phonon());
      Phonons[Phonons.size() - 1].temp = 1 * s;
      Phonons[Phonons.size() - 1].omega = 2 * s;
      Phonons[Phonons.size() - 1].velocity[0] = double(s) / 2; Phonons[Phonons.size() - 1].velocity[1] = double(s) / 3; Phonons[Phonons.size() - 1].velocity[2] = double(s) / 4;
      Phonons[Phonons.size() - 1].cell = s;

    }
  }

  int num_per_task = Nz / numtask;
  int num_last = Nz % numtask;
  int sendcount[numtask], disp[numtask];
  std::vector<Phonon> rbuf(Nz);
  std::vector<Phonon> newph(Nz);
  for (int i = 0; i < numtask - 1; i++)
  {
    sendcount[i] = num_per_task;
    disp[i] = num_per_task * i;
  }
  sendcount[numtask - 1] = num_last + num_per_task;
  disp[numtask - 1] = num_per_task * (numtask - 1);
  MPI_Scatterv(&Phonons[0], sendcount, disp, phonon_type, &rbuf[0], sendcount[rank], phonon_type, 0, MPI_COMM_WORLD);
  std::printf("rank = %d,",rank);
  for (int i = 0; i < sendcount[rank]; i++) std::printf(" %0.1f",rbuf[i].temp);
  std::printf("\n");
for (int i = 0; i < sendcount[rank]; i++) rbuf[i].velocity[0]*=2;
  MPI_Gatherv(&rbuf[0], sendcount[rank], phonon_type, &Phonons[0], sendcount,disp, phonon_type, 0, MPI_COMM_WORLD);
  if (rank==0) {for (int i = 0; i < Nz; i++) std::printf(" %0.1f",*(&Phonons[i].omega));}
  MPI_Finalize();
}


