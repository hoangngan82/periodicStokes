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
  int N[3] = {1, 1, 1};
  int numL = 1, numDl = 10, numDg = 10;
  double dStart  =.2;  // the last value of d = dStart*d0;
  double dEnd    = 2;  // the last value of d = dEnd*d0;
  double h = .5, tol = 1e-15, e = 1e-5;
  double lStart = 1, lStep = 1;
  MatrixOnHost L(3, 1, 1);
  int i = 1;
  int maxShell = 100;
  char* option = "";
  char* filename = "output.bin";
    while (i < argc) {/* options parsing {{{*/
      option = argv[i];
      i++;
      if (i >= argc) {
        cout << endl;
        cout << "Invalid option(s) are found. Program ends!!!! " << endl;
        cout << endl;
        cout << "Usage: optirun --no-xorg ./periodicStokes [ ... ]" << endl;
        cout << endl;
        cout << "-e       epsilon (blob)      (default = " << e << ")" << endl;
        cout << "-t       tol =               (default = " << tol << ")" << endl;
        cout << "-f       filename            (default = optimal.bin)" << endl; 
        cout << "-l       number of different box sizes (Ly's since Lx is fixed at 1) (default = " << numL << ")" << endl;
        cout << "-lstart  starting value of Ly (default = " << lStart << ")" << endl;
        cout << "-lstep   difference between two consecutive box sizes (default = " << lStep << ")" << endl;
        cout << "-dstart  the smallest value of d = dStart*d0, dStart < 1 (default = " << dStart << ")" << endl;
        cout << "-dend    the largest value of d = dEnd*d0, dEnd > 1 (default = " << dEnd << ")" << endl;
        cout << "-n       number of sample points in each direction (value = "
          << N[0] << ")" << endl;
        cout << "-ndl     number of d-sample points less than d0 (default = "
          << numDl << ")" << endl;
        cout << "-ndg     number of d-sample points greater than d0 (default = "
          << numDg << ")" << endl;
        cout << "-s       maximum number of SHELLS (default = " << maxShell << "20)" << endl;
        cout << "-h       the height of the force to the wall (default = .5)" << endl;
        cout << "-Ly      = " << L(1) << endl;
        cout << "-Lz      = " << L(2) << endl;
        exit(EXIT_FAILURE);
      }
      if (strcmp(option, "-h") == 0) {
        h = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-Ly") == 0) {
        L(1) = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-Lz") == 0) {
        L(2) = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-e") == 0) {
        e = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-dstart") == 0) {
        dStart = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-dend") == 0) {
        dEnd = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-lstart") == 0) {
        lStart = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-lstep") == 0) {
        lStep = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-l") == 0) {
        numL = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-s") == 0) {
        maxShell = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-n") == 0) {
        N[0] = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-ndl") == 0) {
        numDl = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-ndg") == 0) {
        numDg = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-t") == 0) {
        tol = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-f") == 0) {
        filename = argv[i];
        i++; continue;
      }
      else {
        cout << endl;
        cout << "Invalid option(s) are found. Program ends!!!! " << endl;
        cout << endl;
        cout << "Usage: optirun --no-xorg ./periodicStokes [ ... ]" << endl;
        cout << endl;
        cout << "-e       epsilon (blob)      (default = " << e << ")" << endl;
        cout << "-t       tol =               (default = " << tol << ")" << endl;
        cout << "-f       filename            (default = optimal.bin)" << endl; 
        cout << "-l       number of different box sizes (Ly's since Lx is fixed at 1) (default = " << numL << ")" << endl;
        cout << "-lstart  starting value of Ly (default = " << lStart << ")" << endl;
        cout << "-lstep   difference between two consecutive box sizes (default = " << lStep << ")" << endl;
        cout << "-dstart  the smallest value of d = dStart*d0, dStart < 1 (default = " << dStart << ")" << endl;
        cout << "-dend    the largest value of d = dEnd*d0, dEnd > 1 (default = " << dEnd << ")" << endl;
        cout << "-n       number of sample points in each direction (default = "
          << N[0] << ")" << endl;
        cout << "-ndl     number of d-sample points less than d0 (default = "
          << numDl << ")" << endl;
        cout << "-ndg     number of d-sample points greater than d0 (default = "
          << numDg << ")" << endl;
        cout << "-s       maximum number of SHELLS (default = " << maxShell << ")" << endl;
        cout << "-h       the height of the force to the wall (default = .5)" << endl;
        cout << "-Ly      = " << L(1) << endl;
        cout << "-Lz      = " << L(2) << endl;
        exit(EXIT_FAILURE);
      }
    }
    cout << endl;
    cout << "========= Chosen options are ========= " << endl;
    cout << "-e       epsilon (blob)      (value = " << e << ")" << endl;
    cout << "-t       tol                 (value = " << tol << ")" << endl;
    cout << "-f       filename            (value = " << filename << endl; 
    cout << "-l       number of different box sizes (Ly's since Lx is fixed at 1) (value = " << numL << ")" << endl;
    cout << "-lstart  starting value of Ly (value = " << lStart << ")" << endl;
    cout << "-lstep   difference between two consecutive box sizes (value = " << lStep << ")" << endl;
    cout << "-dstart  the smallest value of d = dStart*d0 (value = " << dStart << ")" << endl;
    cout << "-dend    the largest value of d = dEnd*d0 (value = " << dEnd << ")" << endl;
    cout << "-n       number of sample points in each direction (value = "
      << N[0] << ")" << endl;
    cout << "-ndl     number of d-sample points less than d0 (value = "
      << numDl << ")" << endl;
    cout << "-ndg     number of d-sample points greater than d0 (value = "
      << numDg << ")" << endl;
    cout << "-s       maximum number of SHELLS (value = " << maxShell << ")" << endl;
    cout << "-h       the height of the force to the wall (default = .5)" << endl;
    cout << "-Lx      = " << L(0) << endl;
    cout << "-Ly      = " << L(1) << endl;
    cout << "-Lz      = " << L(2) << endl;
    cout << "======================================" << endl;/*}}}*/

  // open binary file to save data

  char timeName[80];
  char dValuesFile[80];
  sprintf(timeName, "time_%s", filename);
  sprintf(dValuesFile, "dValues_%s", filename);
  ofstream outM(filename, ios::binary);
  ofstream outTime(timeName, ios::binary);
  ofstream outD(dValuesFile, ios::binary);

  if (!outM) {
    std::cout << "Error: Could not open file \"" << filename << "\"" 
      << ". Error occurs on line " << __LINE__ 
      << " in source file \"" << __FILE__ << "\"" << std::endl;;
    exit(1);
  }
  if (!outTime) {
    std::cout << "Error: Could not open file \"" << timeName << "\"" 
      << ". Error occurs on line " << __LINE__ 
      << " in source file \"" << __FILE__ << "\"" << std::endl;;
    exit(1);
  }
  if (!outD) {
    std::cout << "Error: Could not open file \"" << dValuesFile << "\"" 
      << ". Error occurs on line " << __LINE__ 
      << " in source file \"" << __FILE__ << "\"" << std::endl;;
    exit(1);
  }
  MatrixOnHost x0(3, 1);
  x0(0) = L(0) / 2;
  x0(1) = L(1) / 2;
  x0(2) = h;
  double d, d0;
  int numD = numDl + numDg + 1;
  MatrixOnHost M(2, numD), timeM(2, numD);
  MatrixOnHost tempM(2, 1), tempTimeM(2, 1); 
  MatrixOnHost dValues(numD);
  N[2] = floor(L(2)/L(0)*N[0]);
  MatrixOnHost newM(2, 1), oldM(2, 1);
  for (i = 0; i < numL; i++) {
    L(1) = lStart + i*lStep ;
    x0(1) = L(1) / 2;
    N[1] = floor(L(1)/L(0)*N[0]);
    MatrixOnHost x(3, N[0]*N[1]*N[2]);
    double Lx = L(0), Ly = L(1), Lz = L(2);
    { // set up sample points
      double dx = Lx/N[0], dy = Ly/N[1], dz = Lz/N[2];
      for (int i = 0; i < N[0]; i++) {
        for (int j = 0; j < N[1]; j++) {
          for (int k = 0; k < N[2]; k++) {
            x(0, i + j * N[0] + k * N[0] * N[1]) = i * dx;
            x(1, i + j * N[0] + k * N[0] * N[1]) = j * dy;
            x(2, i + j * N[0] + k * N[0] * N[1]) = k * dz + 0.005;
          } 
        }
      }
    }
    MatrixOnHost A(3*x.columns(), 3*x0.columns());
    MatrixOnHost absA = A, newA = A;
    d0 =  sqrt(L(1)/PI);

    // compute the reference solution for this value of L
    d  = d0;
    newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
    realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
    for (int i = 0; i < maxShell; i++) {
      oldM(0) = newM(0); oldM(1) = newM(1);
      newM(0)++;
      newM(1) = ceil ( newM(0) * Lx/Ly );
      realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
    }
    A = newA;
    newA.setElements(0);
    newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
    fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
    for (int i = 0; i < maxShell; i++) {
      oldM(0) = newM(0); oldM(1) = newM(1);
      newM(1)++;
      newM(0) = ceil ( newM(1) * Lx/Ly );
      fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
    }
    A = A + newA;

    h  = (1 - dStart)*d0/numDl;
    d  = dStart*d0;
    cout << "d is " << d << endl;
    for ( int j = 0; j < numDl; j++) {
      dValues(j) = d;
      optimalNumShells (tempM, tempTimeM, L, x0, x, A, d, e, N, tol, maxShell);
      M(0, j) = tempM(0); M(1, j) = tempM(1);
      timeM(0, j) = tempTimeM(0); timeM(1, j) = tempTimeM(1);
      d += h;
    } // after this for loop d = d0, i.e, d = d0 at location i = numDl in octave

    h  = (dEnd - 1)*d0/numDg;
    for ( int j = numDl; j < numD; j++) {
      dValues(j) = d;
      optimalNumShells (tempM, tempTimeM, L, x0, x, A, d, e, N, tol, maxShell);
      M(0, j) = tempM(0); M(1, j) = tempM(1);
      timeM(0, j) = tempTimeM(0); timeM(1, j) = tempTimeM(1);
      d += h;
    }
    // write data to file
    M.append(outM, true);
    timeM.append(outTime, true);
    L.append(outD, false);
    L.print("L is");
    dValues.print("dvalues i");
    dValues.append(outD, true);
  }
  M.print("M is");
  timeM.print("timeM is");
  outM.close();
  outTime.close();
  
	cout << "Done!!!!!!!!!!!!!!" << endl;

	return EXIT_SUCCESS;
}				// ----------  end of function main  ----------

