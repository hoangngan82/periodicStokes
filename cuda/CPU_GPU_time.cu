/*
 * =====================================================================================
 *
 *       Filename:  testLironPaper.cu
 *
 *    Description:  This tries to match computation results with those in
 *    Liron's paper.
 *
 *        Version:  1.0
 *        Created:  04/04/2014 07:58:39 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Hoang-Ngan Nguyen (), zhoangngan-gmail
 *   Organization:  
 *
 * =====================================================================================
 */

#include "periodicStokes.h"
#include <cstdlib>  /* srand, rand */
#include <ctime>    /* time */
#include <string>
#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;

int main( int argc, char *argv[] ) {
  MatrixOnHost L(3, 1, 1);

  //open binary file to save GPU and CPU times
  //char timeName[80];
  //sprintf(timeName, "CPU_GPU_time", filename);
  //ofstream outTime(timeName, ios::binary);

  //if (!outTime) {
    //std::cout << "Error: Could not open file \"" << timeName << "\"" 
      //<< ". Error occurs on line " << __LINE__ 
      //<< " in source file \"" << __FILE__ << "\"" << std::endl;;
    //exit(1);
  //}
  MatrixOnHost timeM(2, 4);

  int maxShell = 4;
  double d = 1.0/sqrt(atan(1)*4);
  double e = 0;
  
  clock_t start, end;
  cudaEvent_t eStart, eStop;
  HANDLE_ERROR( cudaEventCreate( &eStart ) );
  HANDLE_ERROR( cudaEventCreate( &eStop  ) );

  MatrixOnHost newM(2, 1), oldM(2, 1);
  float elapsedTime;
  size_t size = 10;
  size_t maxCol = 10;
  for (int i = 0; i < 4; i++) {
    cout << "Round " << i << endl;
    MatrixOnHost x(3, size), x0(3, maxCol);
    x.setRandom();
    x0 = x;
    //MatrixOnDevice dx = x, dx0 = x0;
    //MatrixOnDevice dA(3*dx.columns(), 3*dx0.columns());

    // record GPU time
    //for (int j = 0; j * maxCol < size; j++) {
      //for (int l = 0; l < 3; l++) {
        //for (int k = 0; k < maxCol; k++) {
          //x0(l, k) = x(l, j * maxCol + k);
        //}
      //}
      //dx0 = x0;
      //HANDLE_ERROR( cudaEventRecord( eStart, 0) );
      //imageStokeslet( dA, dx, dx0, d, e, maxShell, maxShell-1, L(0), L(1) ); 
      //HANDLE_ERROR( cudaEventRecord( eStop , 0) );
      //HANDLE_ERROR( cudaEventSynchronize( eStop ) );
      //HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, eStart, eStop ) );
      //timeM(0, i) = elapsedTime;
    //}

    //record CPU time
    MatrixOnHost    A(3*x.columns(), 3*x.columns());
    MatrixOnHost    absA = A, refSol = A;
    start = clock();
    newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
    realShells ( refSol, absA, x, x, newM, oldM, L, d, e );
    fourierShells(A, absA, x, x, newM, oldM, L, d, e);
    refSol = refSol + A;
    newM(0) = newM(1) = maxShell;
    realShells ( refSol, absA, x, x, newM, oldM, L, d, e );
    newM(0) = newM(1) = maxShell-1;
    fourierShells(refSol, absA, x, x, newM, oldM, L, d, e);
    end   = clock();
    timeM(1, i) = 1000.0 * ((double) (end - start)) / CLOCKS_PER_SEC ;
    size *= 8;
    cout << "maxShell = " << maxShell << endl;
    cout << "time is " << timeM(1, i) << "ms" << endl;
  }
  timeM.write("CPU_GPU_time");
  HANDLE_ERROR( cudaEventDestroy( eStart ) );
  HANDLE_ERROR( cudaEventDestroy( eStop  ) );
  //outTime.close();
  
	cout << "Done!!!!!!!!!!!!!!" << endl;

	return EXIT_SUCCESS;
}				// ----------  end of function main  ----------

