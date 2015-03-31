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
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>

using namespace std;

// interface for modified bessel function
double besselK0 (double x) {
  assert(x > 0);
  if ( x < 700 )
    return gsl_sf_bessel_K0 (x);
  else
    return 0.0;
}
double besselK1 (double x) {
  assert(x > 0);
  if ( x < 700 )
    return gsl_sf_bessel_K1 (x);
  else
    return 0.0;
}

// === LironFormula: FUNCTION  =========================================={{{ 
//         Name:  LironFormula
//  Description:  Compute periodic flows using Liron's formula.
// =============================================================================
void
LironFormula ( /* argument list: {{{*/
    MatrixOnHost & U,  
    const double & x1, const double & x2, const double & z
    , const double & h 
    , const int & maxL  = 100 
    , const int & maxN  = 100 
    , const double & a = 1, const double & b = 1// a = Lx and b = Ly
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* LironFormula implementation: {{{*/
  double alpha = z - h;
  double beta  = z + h;
  double sbb = sinh(PI*beta/b);
  double sab = sinh(PI*alpha/b);
  double sba = sinh(PI*beta/a);
  double saa = sinh(PI*alpha/a);
  double s2b = sin(PI*x2/b);
  double s1b = sin(PI*x1/b);
  double s2a = sin(PI*x2/a);
  double s1a = sin(PI*x1/a);
  double two  = log ( (sba*sba + s1a*s1a) / (saa*saa + s1a*s1a) ); 
  double five = alpha * sinh (2*PI*alpha/a) / (saa*saa + s1a*s1a) 
         - beta * sinh(2*PI*beta/a)  / (sba*sba + s1a*s1a);
  double two7 = 2*cosh(2*PI*beta/a) / ( sba*sba + s1a*s1a ) ;
  double temp = sinh(2*PI*beta/a) / ( sba*sba + s1a*s1a);
  two7-= temp*temp; 

  double three3 = 2*cos(2*PI*x1/a) / ( sba*sba + s1a*s1a );
     temp = sin(2*PI*x1/a) / ( sba*sba + s1a*s1a) ;
  three3 -= temp * temp ;
  double eight= 0;
  double two0 = 0, two5 = 0;
  double four3= 0, af3 = 0;
  double five1= 0, five2 = 0, five3 = 0;
  double rho1 = 0, rho2 = 0;
  double r1;
  for (int n = -maxN; n <= maxN; n++) {
    r1 = x1 + n*a;
    for (int l = 1; l < maxL; l++) {
      rho1 = sqrt (alpha*alpha + r1*r1);
      rho2 = sqrt (beta*beta + r1*r1);
      eight += cos (2*PI*l*x2/b) * (besselK0(2*PI*l*rho1/b) - besselK0(2*PI*l*rho2/b)); 
      two0  += l*l * cos (2*PI*l*x2/b) * besselK0 (2*PI*l*rho2/b);
      two5  += l * cos (2*PI*l*x2/b) * ( alpha*alpha/rho1 * besselK1(2*PI*l*rho1/b) - beta*beta/rho2*besselK1(2*PI*l*rho2/b));
      four3 += r1*r1 * l*cos(2*PI*l*x2/b) * ( besselK1(2*PI*l*rho1/b)/rho1 - besselK1(2*PI*l*rho2/b)/rho2 );
      af3   += x1*x1 * l*cos(2*PI*l*x2/b) * ( besselK1(2*PI*l*rho1/b)/rho1 - besselK1(2*PI*l*rho2/b)/rho2 );
      temp   = l*cos(2*PI*l*x2/b) * besselK1(2*PI*l*rho2/b) / rho2;
      five1 += temp;
      five2 += temp/(rho2*rho2);
      five3 += l*l*cos(2*PI*l*x2/b) * besselK0(2*PI*l*rho2/b) / (rho2*rho2) ;
    }
  }

  // Since in the numerical result, r01 = .3 -> alpha^2 + r01^2 > 0
  // we have
  U(0, 0) = 1/b * two + 4/b * eight - PI/(a*b) * five 
  + (8*PI)/(b*b) * four3 - 2*PI*PI*h*z/(a*a*b) * three3
  + 16*PI*h*z/(b*b) * five1 + 32*PI*PI*h*z/(b*b*b) * two0
  - 32*PI*PI*PI*h*z*beta*beta/(b*b*b) * five3 
  - 32*PI*h*z*beta*beta/(b*b) * five2;

  U(2, 2) = 1/b * two + 4/b * eight + PI/(a*b) * five + 8*PI/(b*b) * two5
  + 2*PI*PI*h*z/(a*a*b) * two7 + 16*PI*h*z/(b*b) * five1
  - 32*PI*PI*h*z*beta*beta/(b*b*b) * five3 
  - 32*PI*h*z*beta*beta/(b*b) * five2;
  if ((alpha*alpha + x2*x2) > 0) {
    double one, seven, nine, ten;
    one = seven = nine = ten = 0;
    one = log ( (sbb*sbb + s2b*s2b) / (sab*sab + s2b*s2b) );
    for (int n = -maxN; n <= maxN; n++) {
      for (int l = 1; l <= maxL; l++) {
        rho1 = sqrt (alpha*alpha + (x2 + n*b)*(x2 + n*b));
        rho2 = sqrt (beta*beta + (x2 + n*b)*(x2 + n*b));
        seven += cos(2*PI*l*x1/a) * (besselK0(2*PI*l*rho1/a) - besselK0(2*PI*l*rho2/a));
        nine  += l*l* cos(2*PI*l*x1/a) * besselK0(2*PI*l*rho2/a);
        ten   += l * cos(2*PI*l*x1/a) * (rho1 * besselK1(2*PI*l*rho1/a) - rho2*besselK1(2*PI*l*rho2/a));
      }
    }
    U(1, 1) = 2/a * one + 8/a * seven - 32*PI*PI*h*z/(a*a*a) * nine - 8*PI/(a*a) * ten; 
    //printf("diff at x = (%e, %e, %e) is %e\n", x1, x2, z, U(1, 1) - U(0, 0));
    U(0, 0) = U(1, 1);
  } else {
    printf("\n outside we have x = (%e, %e, %e)\n", x1, x2, z);
  }
  //printf("at x2 = %e, x3 = %e: diff using r1 and x1 is af3 - four3 = %e\n", x2, z, af3-four3 );

  //double s1 = 0, s2 = 0, ns2 = 0;
  //two = 0;
  //five1 = 0;
  //five2 = 0;
  //s1 = 1/b * log ( (sba*sba + s1a*s1a) / (saa*saa + s1a*s1a) );
  //for (int n = -maxN; n <= maxN; n++) {
    //r1 = x1 + n*a;
    //rho1 = sqrt (alpha*alpha + (x1 + n*a)*(x1 + n*a));
    //rho2 = sqrt (beta*beta + (x1 + n*a)*(x1 + n*a));
    //for (int l = 1; l <= maxL; l++) {
      //two += cos(2*PI*l*x2/b) * (besselK0(2*PI*l*rho1/b) - besselK0(2*PI*l*rho2/b));
      //five1 += l * cos(2*PI*l*x1/a) * (rho1 * besselK1(2*PI*l*rho1/a) - rho2 * besselK1(2*PI*l*rho2/a));
      //five2 += r1*r1 * l * cos(2*PI*l*x1/a) * ( 1/rho1 * besselK1(2*PI*l*rho1/a) - 1/rho2 * besselK1(2*PI*l*rho2/a) );
    //}
  //}
  //s1 += 4/b*two;
  //s2  = s1 - 8*PI/(a*a) * five1;
  //ns2 = PI/(a*b) * ( beta * sinh(2*PI*beta/a) / (sba*sba + s1a*s1a) - alpha * sinh(2*PI*alpha/a) / (saa*saa + s1a*s1a) );
  //ns2 += 8*PI/(b*b) * five2;
  //printf("ns2 - s2 = %e\n", ns2 - s2);

  //if (z < 1e-15)
    //printf("x = (%e, %e, %e),\n x0 = (0, 0, %e)\n two = %e, eight = %e, five = %e,\n "
        //"two5 = %e, two7 = %e \n four3 = %e, three3 = %e, "
        //"five1 = %e, five2 = %e, five3 = %e\n"
        //"U(2, 2) = %e\n", 
        //x1, x2, z, h, 
        //two, eight, five, two5, two7, four3,
        //three3, five1, five2, five3, U(2, 2));
  //double one, seven, six, two6, two8, five4, five5, five6;
  //one = seven = six = two6 = two8 = five4 = five5 = five6 = 0;
  //one = log ( (sbb*sbb + s2b*s2b) / (sab*sab + s2b*s2b) );
  //six = alpha * sinh (2*PI*alpha/b) / (sab*sab + s2b*s2b) 
    //- beta * sinh (2*PI*beta/b) / (sbb*sbb + s2b*s2b);
  //two8= 
  //for (int n = -maxN; n <= maxN; n++) {
    //rho1 = sqrt (alpha*alpha + (x2 + n*b)*(x2 + n*b));
    //rho2 = sqrt (beta*beta + (x2 + n*b)*(x2 + n*b));
    //for (int l = 1; l <= maxL; l++) {
      //seven += cos(2*PI*l*x1/a) * (besselK0(2*PI*l*rho1/a) - besselK0(2*PI*l*rho2/a) );
    //}
  //}
} /*}}}*/
/* ---------------  end of function LironFormula  -------------------- }}}*/


// === testLironPaper: FUNCTION  =========================================={{{ 
//         Name:  testLironPaper
//  Description:  Compute the velocity field due to the suspension of a sphere
//  near the wall.
// =============================================================================
void
testLironPaper ( /* argument list: {{{*/
    const double & d,  const double & e 
    , const int & numRealShells 
    , const int & numFourierShells 
    , const bool & includeReg = true
    , const double & Lx = 1, const double & Ly = 1
    , const double & x1 = 0.3
    , const double & sx2 = 0
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* testLironPaper implementation: {{{*/
//  std::ifstream fxs("xs.bin", std::ios::binary);
//  std::ifstream fvs("vs.bin", std::ios::binary);
  //gsl_error_handler_t* old_handler = gsl_set_error_handler_off ();
  int numX2 = 6;
  int numX3 = 201;
  double h3 = 2.5/(numX3 - 1);
  MatrixOnHost x0(3, 1), x(3, numX2*numX3);
  x0(2) = .5;
  for (int i = 0; i < numX2; i++) {
    for (int j = 0; j < numX3; j++) {
      x(0, i*numX3 + j) = x1;
      x(1, i*numX3 + j) = 0.1*i + sx2;
      x(2, i*numX3 + j) = h3*j;
    }
  }

  uint numPoints = numX2 * numX3;

  MatrixOnHost U(3*numPoints, 3);
  MatrixOnHost u(3, 3);

  //int maxL = 100, maxN = 100;

  for (int i = 0; i < x.columns(); i++) {
    u.setElements(0);
    LironFormula (u, x(0, i), x(1, i), x(2, i), x0(2));
    for (int j = 0; j < 3; j ++) {
      for (int k = 0; k < 3; k++) {
        U(3*i + j, k) = u(j, k);
      }
    }
  }
  //U.write("LironFormula.bin");
  MatrixOnHost newM(2), oldM(2);
  MatrixOnDevice dA0(3*numPoints, 3);
  MatrixOnDevice dx0 = x0, dx = x;
  imageStokeslet (dA0, dx, dx0, d, e, numRealShells, numFourierShells, Lx, Ly);
  MatrixOnHost A = dA0;
  A.write("testLironPaper.bin");
  A = A - (1.0/(8*PI))*U;
  char filename[80];
  sprintf(filename, "diffAt%4f.bin", x1);
  A.write(filename);
  //MatrixOnHost A(U.rows(), U.columns());
  //MatrixOnHost newA(A.rows(), A.columns());
  //MatrixOnHost absA(A.rows(), A.columns());
  //MatrixOnHost L(3);
  //L(0) = Lx;
  //L(1) = Ly;
  //newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
  //realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
  //newM(0) = numRealShells; newM(1) = ceil ( newM(0) * Lx/Ly );
  //realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
  //A = newA;
  //newA.setElements(0);
  //newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
  //fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
  //newM(1) = numFourierShells; newM(0) = ceil ( newM(1) * Lx/Ly );
  //fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
  //A = A + newA;
  //A.write("testLironPaper.bin");
  //gsl_set_error_handler (old_handler);
} /*}}}*/
/* ---------------  end of function testLironPaper  -------------------- }}}*/


int main( int argc, char *argv[] ) {
  {
    uint numRealShells = 6, numFourierShells = 6;
    bool includeReg = true;
    double d  = 0.5;
    double e  = 0.005;
    double Lx = 1, Ly = 1;
    double x1 = 0.3;
    double sx2= 0;

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
        cout << "-Sr      numRealShells       (default = 11)" << endl;
        cout << "-Sf      numFourierShells    (default = 9)" << endl;
        cout << "-d       delta (Ewald)       (default = 0.5)" << endl;
        cout << "-e       epsilon (blob)      (default = 0.005)" << endl;
        cout << "-x       value of x1 =       (default = 0.3)" << endl;
        cout << "-sx      value of sx2        (default = 0)" << endl;
        cout << "-Lx      box length          (default = 1)" << endl;
        cout << "-Ly      box length          (default = 1)" << endl;
        cout << "-r       turn off regularization term" << endl;
        cout << "-x       x1 =                (default = 0.3)" << endl;
        exit(EXIT_FAILURE);
      }
      if (strcmp(option, "-sx") == 0) {
        sx2 = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-x") == 0) {
        x1 = atof(argv[i]);
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
      if (strcmp(option, "-d") == 0) {
        d = atof(argv[i]);
        i++; continue;
      }
      if (strcmp(option, "-e") == 0) {
        e = atof(argv[i]);
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
      if (strcmp(option, "-x") == 0) {
        x1 = atof(argv[i]);
        i++; continue;
      }
      else {
        cout << endl;
        cout << "Invalid option(s) are found. Program ends!!!! " << endl;
        cout << endl;
        cout << "Usage: optirun --no-xorg ./periodicStokes [ ... ]" << endl;
        cout << endl;
        cout << "-Sr      numRealShells       (default = 11)" << endl;
        cout << "-Sf      numFourierShells    (default = 9)" << endl;
        cout << "-d       delta (Ewald)       (default = 1.5)" << endl;
        cout << "-e       epsilon (blob)      (default = 0.005)" << endl;
        cout << "-x       value of x1 =       (default = 0.3)" << endl;
        cout << "-sx      value of sx2        (default = 0)" << endl;
        cout << "-Lx      box length          (default = 1)" << endl;
        cout << "-Ly      box length          (default = 1)" << endl;
        cout << "-r       turn off regularization term" << endl;
        cout << "-x       x1 =                (default = 0.3)" << endl;
        exit(EXIT_FAILURE);
      }
    }/*}}}*/
    cout << endl;
    cout << "========= Chosen options are ========= " << endl;
    cout << "-Sr      numRealShells       (value = " << numRealShells << ")" << endl;
    cout << "-Sf      numFourierShells    (value = " << numFourierShells << ")" << endl;
    cout << "-d       delta (Ewald)       (value = " << d << ")" << endl;
    cout << "-e       epsilon (blob)      (value = " << e << ")" << endl;
    cout << "-x       value of x1 =       (value = " << x1 << ")" << endl;
    cout << "-sx      value of sx2        (default " << sx2 <<  ")" << endl;
    cout << "-Lx      box length          (value = " << Lx << ")" << endl;
    cout << "-Ly      box length          (value = " << Ly << ")" << endl;
    if (!includeReg)
      cout << "-r       regularization term is off " << endl;
    cout << "-x       x1 =                (value = " << x1 << ")" << endl;
    cout << "======================================" << endl;

    MatrixOnHost vertex(3);
    testLironPaper (d, e, numRealShells, numFourierShells, includeReg, Lx, Ly, x1, sx2);
  }
  
	cout << "Done!!!!!!!!!!!!!!" << endl;

	return EXIT_SUCCESS;
}				// ----------  end of function main  ----------


