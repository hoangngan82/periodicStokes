// =====================================================================================
// 
//       Filename:  periodicStokes.cu
// 
//    Description:  Main file for my periodicStokes
// 
//        Version:  1.0
//        Created:  10/07/13 07:02:16
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Hoang-Ngan Nguyen (), hnguyen87 at ucmerced dot edu
//        Company:  
// 
// =====================================================================================
// Usage:
// periodicStokes numberLoop delta Lx Ly
// TODO:
// - write a test code
// - write a kernel wrap
//
#include "periodicStokesProjects.h"
#include <cstdlib>  /* srand, rand */
#include <ctime>    /* time */
#include <string>
#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;

int main( int argc, char *argv[] ) {
  {
    uint numTimeStep = NUM_TIME_STEP;
    uint numLoopPerSec = NUM_LOOP_PER_SEC;
    uint numTimeStepPerLoop = NUM_TIME_STEP_PER_LOOP;
    uint numRealShells = NUM_REAL_SHELLS, numFourierShells = NUM_FOURIER_SHELLS;
    uint numPoints = NUM_POINTS;
    bool includeReg = true;
    double wsign = 1;
    double phi = PHI, theta = THETA, L = LENGTH, Lx = 1, Ly = 1;
    double d  = 0.5;
    double e  = L/(2*numPoints);
    double scale = 1;

    int i = 1;
    char* option = "";
    while (i < argc) {/* options parsing {{{*/
      option = argv[i];
      if (strcmp(option, "-r") == 0) {
        includeReg = false;
        i++; continue;
      }
      i++;
      if (i >= argc) {
        cout << endl;
        cout << "Invalid option(s) are found. Program ends!!!! " << endl;
        cout << endl;
        cout << "Usage: optirun --no-xorg ./periodicStokes [ ... ]" << endl;
        cout << endl;
        cout << "-T       numTimeStep         (default = " << NUM_TIME_STEP << ")" << endl;
        cout << "-loop    numLoopPerSec       (default = " << NUM_LOOP_PER_SEC << ")" << endl;
        cout << "-T-loop  numTimeStepPerLoop  (default = " << NUM_TIME_STEP_PER_LOOP << ")" << endl;
        cout << "-Sr      numRealShells       (default = " << NUM_REAL_SHELLS << ")" << endl;
        cout << "-Sf      numFourierShells    (default = " << NUM_FOURIER_SHELLS << ")" << endl;
        cout << "-L       cilia length        (default = " << LENGTH << ")" << endl;
        cout << "-C       cilia points        (default = " << NUM_POINTS << ")" << endl;
        cout << "-d       delta (Ewald)       (default = 0.5)" << endl;
        cout << "-e       epsilon (blob)      (default = L/(2*numPoints) = "
          << LENGTH / (2*NUM_POINTS) << ")" << endl;
        cout << "-Lx      box length          (default = 1)" << endl;
        cout << "-Ly      box length          (default = 1)" << endl;
        cout << "-t       tilt angle          (default = " << PI*THETA << " radian)" << endl;
        cout << "-p       cone angle          (default = " << PI*PHI << " radian)" << endl;
        cout << "-w       rotation direction  (default = 1, i.e, counter clock wise)"
          << endl;
        cout << "-r       turn off regularization term" << endl;
        cout << "-c       scale factor        (default = 1, Lx *= c, "
          "Ly *= c, d *= c" << endl;
        exit(EXIT_FAILURE);
      }
      if (strcmp(option, "-T") == 0) {
        numTimeStep = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-loop") == 0) {
        numLoopPerSec = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-T-loop") == 0) {
        numTimeStepPerLoop = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-Sr") == 0) {
        numRealShells = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-Sf") == 0) {
        numFourierShells = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-C") == 0) {
        numPoints = atoi(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-d") == 0) {
        d = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-e") == 0) {
        e = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-L") == 0) {
        L = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-Lx") == 0) {
        Lx = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-Ly") == 0) {
        Ly = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-p") == 0) {
        phi = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-t") == 0) {
        theta = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-c") == 0) {
        scale = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-w") == 0) {
        wsign = (atof(argv[i]) > 0) ? 1 : -1;
        i++; continue;
      }
      else {
        cout << endl;
        cout << "Invalid option(s) are found. Program ends!!!! " << endl;
        cout << endl;
        cout << "Usage: optirun --no-xorg ./periodicStokes [ ... ]" << endl;
        cout << endl;
        cout << "-T       numTimeStep         (default = " << NUM_TIME_STEP << ")" << endl;
        cout << "-loop    numLoopPerSec       (default = " << NUM_LOOP_PER_SEC << ")" << endl;
        cout << "-T-loop  numTimeStepPerLoop  (default = " << NUM_TIME_STEP_PER_LOOP << ")" << endl;
        cout << "-Sr      numRealShells       (default = " << NUM_REAL_SHELLS << ")" << endl;
        cout << "-Sf      numFourierShells    (default = " << NUM_FOURIER_SHELLS << ")" << endl;
        cout << "-L       cilia length        (default = " << LENGTH << ")" << endl;
        cout << "-C       cilia points        (default = " << NUM_POINTS << ")" << endl;
        cout << "-d       delta (Ewald)       (default = 0.5)" << endl;
        cout << "-e       epsilon (blob)      (default = L/(2*numPoints) = "
          << LENGTH / (2*NUM_POINTS) << ")" << endl;
        cout << "-Lx      box length          (default = 1)" << endl;
        cout << "-Ly      box length          (default = 1)" << endl;
        cout << "-t       tilt angle          (default = " << PI*THETA << " radian)" << endl;
        cout << "-p       cone angle          (default = " << PI*PHI << " radian)" << endl;
        cout << "-w       rotation direction  (default = 1, i.e, counter clock wise)"
          << endl;
        cout << "-r       turn off regularization term" << endl;
        cout << "-c       scale factor        (value = " << scale << ", Lx *= c, "
          "Ly *= c, d *= c" << endl;
        exit(EXIT_FAILURE);
      }
    }/*}}}*/
        d  *= scale;
        Lx *= scale;
        Ly *= scale;
    cout << endl;
    cout << "========= Chosen options are ========= " << endl;
    cout << "-T       numTimeStep         (value = " << numTimeStep << ")" << endl;
    cout << "-loop    numLoopPerSec       (value = " << numLoopPerSec << ")" << endl;
    cout << "-T-loop  numTimeStepPerLoop  (value = " << numTimeStepPerLoop << ")" << endl;
    cout << "-Sr      numRealShells       (value = " << numRealShells << ")" << endl;
    cout << "-Sf      numFourierShells    (value = " << numFourierShells << ")" << endl;
    cout << "-L       cilia length        (value = " << L << ")" << endl;
    cout << "-C       cilia points        (value = " << numPoints << ")" << endl;
    cout << "-d       delta (Ewald)       (value = " << d << ")" << endl;
    cout << "-e       epsilon (blob)      (value = " << e << ")" << endl;
    cout << "-Lx      box length          (value = " << Lx << ")" << endl;
    cout << "-Ly      box length          (value = " << Ly << ")" << endl;
    cout << "-t       tilt angle          (value = " << theta << "*PI radian)" << endl;
    cout << "-p       cone angle          (value = " << phi << "*PI radian)" << endl;
    cout << "-w       rotation direction  (value = 1, i.e, counter clock wise)"
      << endl;
    if (!includeReg)
      cout << "-r       regularization term is off " << endl;
    cout << "-c       scale factor        (value = " << scale << ", Lx *= c, "
          "Ly *= c, d *= c" << endl;
    cout << "======================================" << endl;

    double dt = 1.0 / numTimeStepPerLoop / numLoopPerSec;
    double w  = wsign * 2 * PI * numLoopPerSec;
//    phi *= PI;
    theta *= PI;
    MatrixOnHost vertex(3);
 nodalCilia ( w, dt, numTimeStep, d, e, vertex, numRealShells, numFourierShells
     , phi, theta, includeReg, L, Lx, Ly, numPoints
   ); 
//  velocityProfile ( 
//    d, e, numRealShells, numFourierShells, includeReg, Lx, Ly, L//Lx, Ly
//    );  

  }
  
	cout << "Done!!!!!!!!!!!!!!" << endl;

	return EXIT_SUCCESS;
}				// ----------  end of function main  ----------

