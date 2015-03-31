/*
 * This file is the Ewald summation method for image system of a regularized
 * Stokeslet.
 */
#ifndef PERIODIC_STOKES_H_ 
#define PERIODIC_STOKES_H_
#include "matrix.h"
#include <cuda.h>
#include <math_functions.h>
//#include <cstdlib>
//#include <boost/timer/timer.hpp>
#include <iostream>
#include <fstream>
#include <ctime>

// Store 3x3 matrix in col-major to be compatible with lapack
#define IDX2D(row, col) (((col)*3) + row)

__device__ __host__ double 
exp_erf (double x, double z) {/*{{{*/
  assert (z > 0);
  double a = 0;
  if ((x < 0) || (x*x + z*z/4 < 701))
    a = exp(x*z) * erfc(z/2 + x);
  return a;
}/*}}}*/

// === blobFunc: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  blobFunc
//  Description:  Computing S - Sd
// =============================================================================
__device__ __host__ void
blobFunc ( /* argument list: {{{*/
    const double& r2g, const double& d, 
    double& H1, double& H2, double& D1, double& D2, 
    double& H1pr, double& H2pr, double& R, 
    const bool& onlyStokeslet = true
    ) /* ------------- end of argument list -------------------------------}}}*/ 
{ /* blobFunc implementation: {{{*/
  double r2    = r2g/(d*d);
  double expx2 = exp(-r2);
  double r     = sqrt(r2);
  double Erf   = erf(r);

// Stokeslet
  if (r > eps) {
    H1   = 1/d*(Erf/(8*PI*r) + expx2/(4*PI*SPI)*(5 - 8*r2 + 2*r2*r2));
    H2   = 1/(d*d*d)*(Erf/(8*PI*r2*r) - expx2/(4*PI*SPI)*(1/r2 - 6 + 2*r2));
  } else { /* if r is less than machine epsilon {{{*/
    H1 = 3/(2*PI*SPI*d);
    H2 = 5/(2*PI*SPI*d*d*d);
//  printf("inside blobFunc we have r = %e and H2 = %e with d = %e\n", r, H2, d);
  } /*}}}*/

  if (onlyStokeslet) return;

  if ( r > eps ) { /*  {{{*/
    //  Potential dipole
    D1   = 1/(d*d*d)*(-Erf / (4*PI*r*r2) 
        + expx2 / (2*PI*SPI) * (1/r2 + 14 - 20*r2 + 4*r2*r2));
    D2   = 1/(d*d*d*d*d)*( 3*Erf / (4*PI*r2*r2*r) 
        - expx2 * (6/r2/r2 + 4/r2 - 32 + 8*r2) / (4*PI*SPI));

    // Stokeslet doublet
    H1pr = 1/(d*d*d)*(- Erf / (8*PI*r*r2)          
        + expx2 / (4*PI*SPI)*(1/r2 - 26 + 24*r2 - 4*r2*r2));

    H2pr = 1/(d*d*d*d*d)*(- 3*Erf / (8*PI*r2*r2*r)         
        + expx2 / (8*PI*SPI) * (6/r2/r2 + 4/r2 - 32 + 8*r2)); 
    
  } /* }}} */
  else { /* {{{ */
    D1 = -20/(3*PI*SPI*d*d*d);
    D2 = -42/(5*PI*SPI*d*d*d*d*d);

    H1pr = -20/(3*PI*SPI*d*d*d);
    H2pr = -21/(5*PI*SPI*d*d*d*d*d);
  } /*}}}*/

// Rotlet
    R = -expx2*(10 - 11*r2 + 2*r2*r2 ) / (2*PI*SPI*d*d*d);
    
} /*}}}*/
/* ---------------  end of DEVICE function blobFunc  -------------- }}}*/

// === diffBlob: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  diffBlob
//  Description:  Computing Se - Sd where e is the blob parameter and d is the
//  Ewald splitting parameter.
//  We have Se - Sd = (Se - S) - (Sd - S)
// =============================================================================
__device__ __host__ void
diffBlob ( /* argument list: {{{*/
    const double& r2g
    , const double& epsilon 
    , const double& delta
    , double& H1, double& H2, double& D1, double& D2 
    , double& H1pr, double& H2pr, double& R 
    , const bool& onlyStokeslet = true
    ) /* ------------- end of argument list -------------------------------}}}*/ 
{ /* diffBlob implementation: {{{*/
  double d     = epsilon;
  double r2    = r2g/(d*d);
  double expx2 = exp(-r2);
  double r     = sqrt(r2);
  double Erf   = -erfc(r);

  if (epsilon < 1e-16) {
    r = sqrt(r2g);
    blobFunc ( r2g, delta, H1, H2, D1, D2, H1pr, H2pr, R, onlyStokeslet); 
    H1 = 1/(8*PI*r) - H1;
    H2 = 1/(8*PI*r2g*r) - H2;
    D1 = -1/(4*PI*r2g*r) - D1;
    D2 = 3/(4*PI*r2g*r2g*r) - D2;
    H1pr = -1/(8*PI*r2g*r) - H1pr;
    H2pr = -3/(8*PI*r2g*r2g*r) - H2pr;
    R  = -R;
    return;
  }

  if (onlyStokeslet) {
    // Se - S {{{
    if (r > eps) { 
      H1   = 1/d*(Erf/(8*PI*r) + expx2/(4*PI*SPI)*(5 - 8*r2 + 2*r2*r2));
      H2   = 1/(d*d*d)*(Erf/(8*PI*r2*r) - expx2/(4*PI*SPI)*(1/r2 - 6 + 2*r2));
    } 
    else { /* {{{ */
      H1 = 3/(2*PI*SPI*d);
      H2 = 5/(2*PI*SPI*d*d*d);
    } /*}}}*/
    /* }}} */
    
    d     = delta;
    r2    = r2g/(d*d);
    expx2 = exp(-r2);
    r     = sqrt(r2);
    Erf   = -erfc(r);

    // Se - Sd = (Se - S) - (Sd - S) {{{
    if (r > eps) { 
      H1  -= 1/d*(Erf/(8*PI*r) + expx2/(4*PI*SPI)*(5 - 8*r2 + 2*r2*r2));
      H2  -= 1/(d*d*d)*(Erf/(8*PI*r2*r) - expx2/(4*PI*SPI)*(1/r2 - 6 + 2*r2));
    } 
    else { /* {{{ */
      H1 -= 3/(2*PI*SPI*d);
      H2 -= 5/(2*PI*SPI*d*d*d);
    } /*}}}*/
    /* }}} */
    return;
  }

  // if we need more than just Stokeslet
  // Se - S {{{
  if (r > eps) { 
    // Stokeslet
    H1   = 1/d*(Erf/(8*PI*r) + expx2/(4*PI*SPI)*(5 - 8*r2 + 2*r2*r2));
    H2   = 1/(d*d*d)*(Erf/(8*PI*r2*r) - expx2/(4*PI*SPI)*(1/r2 - 6 + 2*r2));

    //  Potential dipole
    D1   = 1/(d*d*d)*(-Erf / (4*PI*r*r2) 
        + expx2 / (2*PI*SPI) * (1/r2 + 14 - 20*r2 + 4*r2*r2));
    D2   = 1/(d*d*d*d*d)*( 3*Erf / (4*PI*r2*r2*r) 
        - expx2 * (6/r2/r2 + 4/r2 - 32 + 8*r2) / (4*PI*SPI));

    // Stokeslet doublet
    H1pr = 1/(d*d*d)*(- Erf / (8*PI*r*r2)          
        + expx2 / (4*PI*SPI)*(1/r2 - 26 + 24*r2 - 4*r2*r2));

    H2pr = 1/(d*d*d*d*d)*(- 3*Erf / (8*PI*r2*r2*r)         
        + expx2 / (8*PI*SPI) * (6/r2/r2 + 4/r2 - 32 + 8*r2)); 
    
  } 
  else { /* {{{ */
    H1 = 3/(2*PI*SPI*d);
    H2 = 5/(2*PI*SPI*d*d*d);

    D1 = -20/(3*PI*SPI*d*d*d);
    D2 = -42/(5*PI*SPI*d*d*d*d*d);

    H1pr = -20/(3*PI*SPI*d*d*d);
    H2pr = -21/(5*PI*SPI*d*d*d*d*d);
  } /*}}}*/

  // Rotlet
  R = -expx2*(10 - 11*r2 + 2*r2*r2 ) / (2*PI*SPI*d*d*d);
  /* }}} */
  
  d     = delta;
  r2    = r2g/(d*d);
  expx2 = exp(-r2);
  r     = sqrt(r2);
  Erf   = -erfc(r);

  // Se - Sd = (Se - S) - (Sd - S) {{{
  if (r > eps) { 
    // Stokeslet
    H1  -= 1/d*(Erf/(8*PI*r) + expx2/(4*PI*SPI)*(5 - 8*r2 + 2*r2*r2));
    H2  -= 1/(d*d*d)*(Erf/(8*PI*r2*r) - expx2/(4*PI*SPI)*(1/r2 - 6 + 2*r2));

    //  Potential dipole
    D1  -= 1/(d*d*d)*(-Erf / (4*PI*r*r2) 
        + expx2 / (2*PI*SPI) * (1/r2 + 14 - 20*r2 + 4*r2*r2));
    D2  -= 1/(d*d*d*d*d)*( 3*Erf / (4*PI*r2*r2*r) 
        - expx2 * (6/r2/r2 + 4/r2 - 32 + 8*r2) / (4*PI*SPI));

    // Stokeslet doublet
    H1pr -= 1/(d*d*d)*(- Erf / (8*PI*r*r2)          
        + expx2 / (4*PI*SPI)*(1/r2 - 26 + 24*r2 - 4*r2*r2));

    H2pr -= 1/(d*d*d*d*d)*(- 3*Erf / (8*PI*r2*r2*r)         
        + expx2 / (8*PI*SPI) * (6/r2/r2 + 4/r2 - 32 + 8*r2)); 
    
  } 
  else { /* {{{ */
    H1 -= 3/(2*PI*SPI*d);
    H2 -= 5/(2*PI*SPI*d*d*d);

    D1 -= -20/(3*PI*SPI*d*d*d);
    D2 -= -42/(5*PI*SPI*d*d*d*d*d);

    H1pr -= -20/(3*PI*SPI*d*d*d);
    H2pr -= -21/(5*PI*SPI*d*d*d*d*d);
  } /*}}}*/

  // Rotlet
  R -= -expx2*(10 - 11*r2 + 2*r2*r2 ) / (2*PI*SPI*d*d*d);
  /* }}} */
    
} /*}}}*/
/* ---------------  end of DEVICE function diffBlob  -------------- }}}*/

// Compute I2n and dI2n ============= {{{
// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  evalI2m
//  Description:  
//  i2m (x, z) = PI^(1/2) * (z^4 + 4z^2 - 8 (6 + z^2) x^2 + 16x^4 + 12)
//  i2m (x, z) = i2m (x, z) * exp (-x^2);
//
//  Input: x^2, exp(-x^2), z^2
// =============================================================================
__device__ __host__ double
evalI2m ( double& x2, double& z2, double& expx2, double& expz2 ) {
    return SPI * expx2 * expz2 * 
        ( z2*( z2 + 4 ) - 8*( 6 + z2 - 2*x2 )*x2 + 12 );
}		// -----  end of DEVICE function evalI2m  ----- 

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  evalI1m
//  Description:  
//  i1m( x, z ) = PI^(1/2) * ( z^2 - 4x^2 + 2) * exp( -x^2) 
//
//  Input   : x^2, exp( -x^2 ), z^2
// =============================================================================
__device__ __host__ double
evalI1m ( double& x2, double& z2, double& expx2, double& expz2 ) {
    return SPI * ( z2 - 4*x2 + 2 ) * expx2 * expz2 ;
}		// -----  end of DEVICE function evalI1m  ----- 

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  evalI0
//  Description:  
//  i0( x, z ) = PI^(1/2) * exp( -x^2) 
//
//  Input   : exp( -x^2) 
// =============================================================================
__device__ __host__ double
evalI0 ( double& expx2, double& expz2 ) {
    return SPI * expx2 * expz2 ;
}		// -----  end of DEVICE function evalI0  ----- 

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  evalI1
//  Description:  
//  i1( x, z ) = exp( -zx ) * erfc( z/2 - x ) + exp( xz ) * erfc( z/2 + x) ;
//  i1( x, z ) = PI/4/z*exp( z^2/4 ) * i1( x, z ) ;
//
//  However, we only compute the part without the term exp( z^2/4 ) since it 
//  will be
//  cancelled by exp( -z^2/4 ) in the final formula.
//
//  Input   : z, experfp, experfm 
//      experfm = exp ( -xz ) * ( 1 - erf ( z/2 - x ) ) 
//      experfp = exp ( +xz ) * ( 1 - erf ( z/2 + x ) ) 
//  Output  : 
// =============================================================================
__device__ __host__ double
evalI1 ( double& z, double& experfp, double& experfm ) {
    return PI/4/z * ( experfp + experfm ) ;
}		// -----  end of DEVICE function evalI1  ----- 

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  evalI2
//  Description:  
//  i2 ( x, z ) = 1/4/z/z * exp ( z^2/4 ) * ( a + b) ;
//  a           =  exp ( -x^2 - z^2/4 ) * SPI;
//  b           =  ( 2 - 2xz - z^2 ) * i1 ( x, z ) ;
//
//  Input   : x, z
//          : expx2 = exp ( -x^2 ) 
//          : expz2 = exp ( -z^2/4 ) 
//          : i1 = i1 ( x, z ) ;    -- we removed exp ( z^2/4 ) from i1
// =============================================================================
__device__ __host__ double
evalI2 ( double& x, double& z, double& z2, double& expx2, double& expz2, 
        double& experfp, double& experfm ) {
    double temp = PI * ( experfm * ( 2 + 2*x*z - z2 ) - experfp * ( -2 + 2*x*z
                + z2 ) ) ;
    temp = temp + 4 * expx2 * expz2 * z * SPI ;
    return temp / 16 / z2 / z ;
}		// -----  end of DEVICE function evalI2  ----- 

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  evalDI0
//  Description:  
// =============================================================================
__device__ __host__ double
evalDI0 ( double& x, double& expx2, double& expz2 ) {
    return -2 * expx2 * expz2 * SPI * x ;
}		// -----  end of DEVICE function evalDI0  ----- 

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  evalDI2m
//  Description:  
// =============================================================================
__device__ __host__ double
evalDI2m ( double& x2, double& z2, double& di0 ) {
    return di0*( 60 + 8*x2*( 2*x2 - 10 - z2 ) + z2*( 12 + z2 ) ) ;
}		// -----  end of DEVICE function evalDI2m  ----- 

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  evalDI1m
//  Description:  
// =============================================================================
__device__ __host__ double
evalDI1m ( double& x2, double& z2, double& di0 ) {
    return di0 * ( z2 - 4 * x2 + 6 ) ;
}		// -----  end of DEVICE function evalDI1m  ----- 

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  evalDI1
//  Description:  
// =============================================================================
__device__ __host__ double
evalDI1 ( double& experfp, double& experfm ) {
    return PI/4*( experfp - experfm ) ;
}		// -----  end of DEVICE function evalDI1  ----- 

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  evalDI2
//  Description:  
// =============================================================================
__device__ __host__ double
evalDI2 ( double& x, double& z, double experfp, double& experfm ) {
    return PI/16/z*((-2 * x + z)*experfm - (2 * x + z)*experfp) ;
}		// -----  end of DEVICE function evalDI2  ----- 
/* -------- end of computing I2n and dI2n ---------------}}} */

// === imageReal: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  imageReal
//  Description:  Compute the image system for point force.
// =============================================================================
__device__ __host__ void
imageReal ( /* argument list: {{{*/
    double* A, double* xh, 
    const double& h, const double& d, const double& e 
    ) /* ------------- end of argument list -----------------------------------}}}*/ 
{ /* imageReal implementation: {{{*/
  double r2, H1, H2, D1, D2, H1pr, H2pr, R;
  
  //  Contribution from original point.
  r2 = dot(xh, xh);
  diffBlob  (r2, e, d, H1, H2, D1, D2, H1pr, H2pr, R);
//  printf("r2 = %e e = %e H1 = %e H2 = %e D1 = %e D2 = %e H1pr = %e H2pr = %e\
//      R = %e\n", r2, e, H1, H2, D1, D2, H1pr, H2pr, R);

  // Compute Sij = dij * H1 + xixj * H2;
  for ( int i = 0; i < 3; i++ ) { /* loop by row {{{*/
    for ( int j = 0; j < 3; j++ ) { /* loop by columns {{{*/
      A[IDX2D(i, j)] = xh[i]*xh[j]*H2;
    }                      /*---------- end of for loop ----------------}}}*/
    A[IDX2D(i, i)] += H1;
  }                        /*---------- end of for loop ----------------}}}*/
//  printf("A[0, 0] = %e A[0, 1] = %e A[0, 2] = %e \n\
//      A[1, 0] = %e A[1, 1] = %e A[1, 2] = %e \n\
//      A[2, 0] = %e A[2, 1] = %e A[2, 2] = %e \n", 
//      A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8]);

//  Image point
  xh[2] += 2*h;
  double z = xh[2];
  r2 = dot (xh, xh);
  diffBlob  (r2, e, d, H1, H2, D1, D2, H1pr, H2pr, R, false);

  // Compute Sij = dij * H1 + xixj * H2;
  for ( int i = 0; i < 3; i++ ) { /* loop by row {{{*/
    for ( int j = 0; j < 3; j++ ) { /* loop by columns {{{*/
      A[IDX2D(i, j)] -= xh[i]*xh[j]*H2;
    }                      /*---------- end of for loop ----------------}}}*/
    A[IDX2D(i, i)] -= H1;
  }                        /*---------- end of for loop ----------------}}}*/

//  # Compute PDij = dij*D1 + xixj*D2 
//  # Note that the new force is diag([-1, -1, 1]) times the old force.
//  # So the first two columns must be multiplied by (-1).
  D1 *= h*h;
  D2 *= h*h;
  for ( int j = 0; j < 2; j++ ) { /* loop by columns {{{*/
    A[IDX2D(j, j)] -= D1;
    for ( int i = 0; i < 3; i++ ) { /* loop by rows {{{*/
      A[IDX2D(i, j)] -= xh[i]*xh[j]*D2;
    }                      /*---------- end of for loop ----------------}}}*/
  }                        /*---------- end of for loop ----------------}}}*/
  // The last column
  A[IDX2D(2, 2)] += D1;
  for ( int i = 0; i < 3; i++ ) { /* loop by rows {{{*/
    A[IDX2D(i, 2)] += xh[i]*z*D2;
  }                        /*---------- end of for loop ----------------}}}*/

//# Stokeslet doublet at the image point
//  # Compute SDij = (xi*dj3 + x3*dij)*H2 + x3*xixj*H2pr + di3*xj*H1pr;
//  # Note that the new force is diag([-1, -1, 1]) times the old force.
//  # So the first two columns must be multiplied by (-1).
  H1 *= 2*h;
  H2 *= 2*h;
  H1pr *= 2*h;
  H2pr *= 2*h*z;
  for ( int j = 0; j < 2; j++ ) { /* loop by columns {{{*/
    A[IDX2D(j, j)] -= z*H2;
    A[IDX2D(2, j)] -= xh[j]*H1pr;
    for ( int i = 0; i < 3; i++ ) { /* loop by rows {{{*/
      A[IDX2D(i, j)] -= xh[i]*xh[j]*H2pr;
    }                      /*---------- end of for loop ----------------}}}*/
  }                        /*---------- end of for loop ----------------}}}*/

  // The last column
  A[IDX2D(2, 2)] += z*(H2 + H1pr);
  for ( int i = 0; i < 3; i++ ) { /* loop by rows {{{*/
    A[IDX2D(i, 2)] += xh[i]*H2 + xh[i]*z*H2pr;
  }                        /*---------- end of for loop ----------------}}}*/

//# Rotlet at the image point
//  # Compute Rij = ( di3*xj - x3*dij ) * R
  R *= 2*h;
  for ( int j = 0; j < 3; j++ ) { /* loop by columns {{{*/
    A[IDX2D(2, j)] += xh[j] * R;
    A[IDX2D(j, j)] -= z * R;
  }                        /*---------- end of for loop ----------------}}}*/

} /*}}}*/
/* ---------------  end of DEVICE function imageReal  -------------- }}}*/

// === imageFourier: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  imageFourier
//  Description:  Compute the image system corresponding to one wave number.
// =============================================================================
__device__ __host__ void
imageFourier ( /* argument list: {{{*/
    double* A, const double* const xh, double* l, 
    const double& h, const double& d 
    , const double& Lx = 1, const double& Ly = 1
    ) /* ------------- end of argument list -----------------------------------}}}*/ 
{ /* imageFourier implementation: {{{*/
  for (int i = 0; i < 9; i++) {
    A[i] = 0;
  }
//  l[0] *= 2*PI/Lx;
//  l[1] *= 2*PI/Ly;
  // Compute the contribution from the image point
//  printf("imageFourier: l = %e %e %e\n", l[0], l[1], l[2]);
  double x        = ( xh[2] + 2*h ) / d;
  double x2       = x * x ;
  double z2       = dot (l, l, 2) * d * d ;
  double z        = sqrt (z2) ;
  double expx2    = exp (-x2) ;
  double expz2    = exp (-z2/4) ;
  double experfm  = exp_erf(-x, z);
  double experfp  = exp_erf( x, z);

  double i2m      = evalI2m ( x2, z2, expx2, expz2 ) ;
  double i1m      = evalI1m ( x2, z2, expx2, expz2 ) ;
  double i0       = evalI0  ( expx2, expz2 ) ;
  double i1       = evalI1  ( z, experfp, experfm ) ;
  double i2       = evalI2  ( x, z, z2, expx2, expz2, experfp, experfm ) ;
  /*  First derivative of the in-th functions with respecto to x */
  double di0      = evalDI0 ( x, expx2, expz2 ) ;
  double di2m     = evalDI2m( x2, z2, di0 ) ;
  double di1m     = evalDI1m( x2, z2, di0 ) ;
  double di1      = evalDI1 ( experfp, experfm ) ;
  double di2      = evalDI2 ( x, z, experfp, experfm ) ;
//    printf("at image: x = %e z = %e expx2 = %e expz2 = %e experfm = %e experfp = %e\n", 
//        x, z, expx2, expz2, experfm, experfp);

  /* Second derivatives: d2 i(n) = -i(n-1) + z^2 i(n) 
   * Third  derivatives: d3 i(n) = -d i(n-1) + d z^2 i(n)  */

  // Compute cos ( l \cdot yhat ) and sin
  x   = dot (l, xh, 2);
  z   = sin (x) ;
  x   = cos (x) ;

// Compute the contribution from Stokeslet at the image point/*{{{*/
  double Sn       = i2 + i1/4 - i0/16 + i1m/32 ;
  double Sm       = i1 + i0/4 - i1m/16 + i2m/32 ;
  double dSn      = di2 + di1/4 - di0/16 + di1m/32 ;

/*     Formula for Stokeslet at the image point {{{
 *     f1 = -f[0]; f2 = -f[1]; f3 = -f[2];
 *     temp    = x * ( -l1 * d * fDotL * Sn + f1 * Sm ) - z * f3 * l1 * d * dSn ;
 *     u1      = 2 * d * temp ;
 *     temp    = x * ( -l2 * d * fDotL * Sn + f2 * Sm ) - z * f3 * l2 * d * dSn ;
 *     u2      = 2 * d * temp ;    
 *     temp    = x * f3 * z2 * Sn - z * fDotL * dSn ;
 *     u3      = 2 * d * temp ;
 }}} */

  double temp = 2 * d;
  for ( int i = 0; i < 2; i++ ) { /* the first two rows {{{*/
    A[IDX2D(i, i)] -= temp * x * Sm;
    A[IDX2D(i, 2)] += temp * z * l[i] * d * dSn;
    for ( int j = 0; j < 2; j++ ) { /* the first two columns {{{*/
      A[IDX2D(i, j)] += temp * d * d * x * Sn * l[i] * l[j];
    }         /*---------- end of for loop ----------------}}}*/
  }         /*---------- end of for loop ----------------}}}*/
  // The last row
  A[IDX2D(2, 2)] -= temp * x * z2 * Sn;
  for ( int j = 0; j < 2; j++ ) { /* the last row {{{*/
    A[IDX2D(2, j)] += temp * z * l[j] * d * dSn;
  }         /*---------- end of for loop ----------------}}}*/
/*}}}*/

  // Stokeslet doublet/*{{{*/
/*     Formula for Stokeslet doublet {{{
 *     f1 = -f[0]; f2 = -f[1]; 
 *     f3 = +f[2];
 *     temp    = -fDotL * l1 * d * dSn * x - z * f3 * l1 * d * ( z2 * Sn - Sm ) ;
 *     u1     += 4 * h * temp ;
 *     temp    = -fDotL * l2 * d * dSn * x - z * f3 * l2 * d * ( z2 * Sn - Sm ) ;
 *     u2     += 4 * h * temp ;
 *     temp    = -f3 * dSn * x + z * fDotL * Sn ;
 *     u3     -= 4 * h * z2 * temp ;
 *     }}}*/
  temp = 4 * h;
  // The first two rows
  for ( int i = 0; i < 2; i++ ) { /* the first two rows {{{*/
    for ( int j = 0; j < 2; j++ ) { /* the first two columns {{{*/
      A[IDX2D(i, j)] -= -temp * d * d * dSn * x * l[i] * l[j];
    }         /*---------- end of for loop ----------------}}}*/
    // The last column
    A[IDX2D(i, 2)] -= temp * z * l[i] * d * ( z2 * Sn - Sm );
  }         /*---------- end of for loop ----------------}}}*/
  // The last row
  for ( int j = 0; j < 2; j++ ) { /* all columns {{{*/
    A[IDX2D(2, j)] += temp * z2 * l[j] * d * z * Sn;
  }         /*---------- end of for loop ----------------}}}*/
  A[IDX2D(2, 2)] += temp * z2 * dSn * x;
  /*}}}*/

    // Compute the contribution from potential dipole at the image point/*{{{*/
    Sn      = i1 + i0/4 + i1m/8 ;
    Sm      = i0 + i1m/4 + i2m/8 ;
    dSn     = di1 + di0/4 + di1m/8 ;

/*     Formula for the potential dipole  {{{
 *     f1 = -f[0]; f2 = -f[1]; 
 *     f3 = +f[2];
 *     temp    = x * ( -l1 * d * fDotL * Sn + f1 * Sm ) - z * f3 * l1 * d * dSn ;
 *     u1     += 2 * h * h / d * temp ;
 *     temp    = x * ( -l2 * d * fDotL * Sn + f2 * Sm ) - z * f3 * l2 * d * dSn ;
 *     u2     += 2 * h * h / d * temp ;    
 *     temp    = x * f3 * z2 * Sn - z * fDotL * dSn ;
 *     u3     += 2 * h * h / d * temp ;
 **********************************************}}}*/
  temp = 2 * h * h / d;
  for ( int i = 0; i < 2; i++ ) { /* the first two rows {{{*/
    A[IDX2D(i, i)] -= temp * x * Sm;
    A[IDX2D(i, 2)] -= temp * z * l[i] * d * dSn;
    for ( int j = 0; j < 2; j++ ) { /* the first two columns {{{*/
      A[IDX2D(i, j)] += temp * d * d * x * Sn * l[i] * l[j];
    }         /*---------- end of for loop ----------------}}}*/
  }         /*---------- end of for loop ----------------}}}*/
  // The last row
  A[IDX2D(2, 2)] += temp * x * z2 * Sn;
  for ( int j = 0; j < 2; j++ ) { /* the last row {{{*/
    A[IDX2D(2, j)] += temp * z * l[j] * d * dSn;
  }         /*---------- end of for loop ----------------}}}*/
/*}}}*/
//
    // Contribution from the rotlet at the image point/*{{{*/
    Sn      = i1m - i2m/6 ;
    dSn     = di1m - di2m/6 ;
/*     u1     += 3 * h / 4 * x * f1 * dSn ;
 *     u2     += 3 * h / 4 * x * f2 * dSn ;
 *     u3     += 3 * h / 4 * z * fDotL * Sn ;
 */
    temp = 3 * h / 4;
    for ( int i = 0; i < 2; i++ ) { /*  {{{*/
      A[IDX2D(i, i)] += temp * x * dSn;
      A[IDX2D(2, i)] += temp * z * l[i] * d * Sn;
    }         /*---------- end of for loop ----------------}}}*/
/*}}}*/

// Compute the contribution from the original point/*{{{*/
    x        = xh[2] / d;
    x2       = x * x;
    z2       = dot (l, l, 2) * d * d;
    z        = sqrt (z2) ;
    expx2    = exp (-x2) ;
    expz2    = exp (-z2/4) ;
    experfm  = exp_erf (-x, z);
    experfp  = exp_erf ( x, z);

    i2m      = evalI2m ( x2, z2, expx2, expz2 ) ;
    i1m      = evalI1m ( x2, z2, expx2, expz2 ) ;
    i0       = evalI0  ( expx2, expz2 ) ;
    i1       = evalI1  ( z, experfp, experfm ) ;
    i2       = evalI2  ( x, z, z2, expx2, expz2, experfp, experfm ) ;
    /*  First derivative of the in-th functions with respecto to x */
    di0      = evalDI0 ( x, expx2, expz2 ) ;
    di2m     = evalDI2m( x2, z2, di0 ) ;
    di1m     = evalDI1m( x2, z2, di0 ) ;
    di1      = evalDI1 ( experfp, experfm ) ;
    di2      = evalDI2 ( x, z, experfp, experfm ) ;
//    printf("at original: x = %e z = %e expx2 = %e expz2 = %e experfm = %e experfp = %e\n", 
//        x, z, expx2, expz2, experfm, experfp);

    /* Second derivatives: d2 i(n) = -i(n-1) + z^2 i(n) 
     * Third  derivatives: d3 i(n) = -d i(n-1) + d z^2 i(n)  */

// Compute cos ( l \cdot y ) and sin
    x   = dot ( l, xh, 2 );
    z   = sin (x) ;
    x   = cos (x) ;

// Compute the contribution from Stokeslet at the original point/*{{{*/
  Sn       = i2 + i1/4 - i0/16 + i1m/32 ;
  Sm       = i1 + i0/4 - i1m/16 + i2m/32 ;
  dSn      = di2 + di1/4 - di0/16 + di1m/32 ;
//  printf("Sn = %+20.14e Sm = %+20.14e dSn = %+20.14e\n", Sn, Sm, dSn);

/*     Formula for Stokeslet at the original point {{{
 *     temp    = x * ( -l1 * d * fDotL * Sn + f1 * Sm ) - z * f3 * l1 * d * dSn ;
 *     u1      = 2 * d * temp ;
 *     temp    = x * ( -l2 * d * fDotL * Sn + f2 * Sm ) - z * f3 * l2 * d * dSn ;
 *     u2      = 2 * d * temp ;    
 *     temp    = x * f3 * z2 * Sn - z * fDotL * dSn ;
 *     u3      = 2 * d * temp ;
 }}} */

  temp = 2 * d;
  for ( int i = 0; i < 2; i++ ) { /* the first two rows {{{*/
    A[IDX2D(i, i)] += temp * x * Sm;
    A[IDX2D(i, 2)] -= temp * z * l[i] * d * dSn;
    for ( int j = 0; j < 2; j++ ) { /* the first two columns {{{*/
      A[IDX2D(i, j)] -= temp * d * d * x * Sn * l[i] * l[j];
    }         /*---------- end of for loop ----------------}}}*/
  }         /*---------- end of for loop ----------------}}}*/
// The last row
  A[IDX2D(2, 2)] += temp * x * (z2 * Sn);
  for ( int j = 0; j < 2; j++ ) { /* the last row {{{*/
    A[IDX2D(2, j)] -= temp * z * l[j] * d * dSn;
  }         /*---------- end of for loop ----------------}}}*/
/*}}}*/
/*}}}*/
/*     u1 /=  ( 2*PI*Lx*Ly ) ;
 *     u2 /=  ( 2*PI*Lx*Ly ) ;
 *     u3 /=  ( 2*PI*Lx*Ly ) ;
 */
//    x        = xh[2] / d;
//    x2       = x * x;
//    z2       = dot (l, l, 2) * d * d;
//    z        = sqrt (z2) ;
//    expx2    = exp (-x2) ;
//    expz2    = exp (-z2/4) ;
//    experfm  = exp (-x*z) * erfc ( z/2 - x ) ;
//    experfp  = exp ( x*z) * erfc ( z/2 + x ) ;
  for (int i = 0; i < 9; i++) {
//    if (isnan(A[i]))
//       printf("at original: x = %e z = %e expx2 = %e expz2 = %e experfm = %e experfp = %e\n", 
//        x, z, expx2, expz2, experfm, experfp);
    A[i] /= ( 2*PI*Lx*Ly );
  }
//  if ((abs(l[0]) > 8*2*PI) || (abs(l[1]) > 8*2*PI))
//  printf("A[0, 0] = %e A[0, 1] = %e A[0, 2] = %e \n\
//      A[1, 0] = %e A[1, 1] = %e A[1, 2] = %e \n\
//      A[2, 0] = %e A[2, 1] = %e A[2, 2] = %e \n\
//      and l = %e %e %e\n", 
//      A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], l[0], l[1], l[2]);

} /*}}}*/
/* ---------------  end of DEVICE function imageFourier  -------------- }}}*/

// === imageZeroReg: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  imageZeroReg
//  Description:  Compute the zero-th order term and the regularization term.
// =============================================================================
__device__ __host__ void
imageZeroReg ( /* argument list: {{{*/
    double* A, const double* const xh, const double & h, const double & d 
    , const bool& includeReg = true
    , const double & Lx = 1, const double & Ly = 1
    ) /* ------------- end of argument list -------------------------------}}}*/ 
{ /* imageZeroReg implementation: {{{*/
    // contribution from original point
  double  x       = abs (xh[2] / d);
  double  x2      = x * x ;
  double  temp    = SPI / 2 * x2 * ( 2*x2 - 5 ) * exp ( -x2 ) 
    + x * PI * erfc ( x ) ;
  temp   *= d ;
  A[IDX2D(0, 0)] += temp;
  A[IDX2D(1, 1)] += temp;

    // contribution from the image point
  x       = ( xh[2] + 2*h ) / d;
  x2      = x * x ;
    // rotlet
  temp    = 2 * h * SPI * x * ( x2 * ( 2*x2 - 7 ) + 3 ) * exp ( -x2 ) ;
  A[IDX2D(0, 0)] += temp;
  A[IDX2D(1, 1)] += temp;

    // potential dipole
  temp    = -2 * exp ( -x2 ) * SPI * ( x2 * ( 2*x2 -7 ) +3 ) ;
  temp   *= h * h / d ;
  A[IDX2D(0, 0)] += temp;
  A[IDX2D(1, 1)] += temp;

    // Stokeslet
    temp    = SPI / 2 * x2 * ( 2*x2 - 5 ) * exp ( -x2 ) + x * PI * erfc ( x ) ;
    temp   *= d ;
  A[IDX2D(0, 0)] -= temp;
  A[IDX2D(1, 1)] -= temp;

  A[IDX2D(0, 0)] /=  ( 2*PI*Lx*Ly ) ;
  A[IDX2D(1, 1)] /=  ( 2*PI*Lx*Ly ) ;
  A[IDX2D(2, 2)] /=  ( 2*PI*Lx*Ly ) ;

  // Adding regularization terms according to Pozrikidis' paper
  if (includeReg) {
    x       =   abs( xh[2] ) - ( xh[2] + 2*h ) ;
    x      *=  .5/Lx/Ly ;
    A[IDX2D(0, 0)] -= x;
    A[IDX2D(1, 1)] -= x;
  }
  
} /*}}}*/
/* ---------------  end of DEVICE function imageZeroReg  -------------- }}}*/

// === kernelImageReal: CUDA KERNEL ========================================{{{
//         Name:  kernelImageReal
//  Description:  Compute the action of one (1) point force on one (1) field
//  point.
//  Each block will compute the matrix corresponding to one (1) field point and
//  one (1) point force.
//  BlockSize must be a power of 2.
// =============================================================================
__global__ void
kernelImageReal ( /* argument list: {{{*/
    MatrixOnDevice dA
    , MatrixOnDevice dx
    , MatrixOnDevice dx0 
    , double d, double e 
    , int Mx // number of real shells in x direction
    , int My // number of real shells in y direction
    , double Lx, double Ly
    , uint bitSize
    ) /* ------------- end of argument list ------------------------------}}}*/ 
{ /* kernelImageReal implementation: {{{*/
  uint blockSize = 1 << bitSize;
  assert ( blockSize == blockDim.x * blockDim.y );
  extern __shared__ double U[];
  __shared__ double xh0[3];
  double A[9];

  uint xid   =  blockIdx.x;
  uint yid   =  blockIdx.y;
  uint xSize =  gridDim.x;
  uint ySize =  gridDim.y;
  uint tid;
  int l1, l2;

  double h;
  double xh[3];
  while ((xid < dA.rows()/3)) { 
    yid = blockIdx.y;
    while ((yid < dA.columns()/3)) {
      h = dx0(2, yid);

      tid =  threadIdx.x + threadIdx.y * blockDim.x;

      for ( int i = 0; i < 9; i++ ) { /* initialize data */
        U[tid + i*blockSize] = 0;
        A[i] = 0;
      }         /*---------- end of for loop ----------------*/
      
      if ( tid == 0 ) { /* compute a common xh for all threads in a block */
        xh0[0]  = dx(0, xid) - dx0(0, yid);
        xh0[1]  = dx(1, xid) - dx0(1, yid);
        xh0[2]  = dx(2, xid) - dx0(2, yid);
      }         /*---------- end of if ----------------------*/

      __syncthreads();

      while ( tid < (2*Mx + 1) * (2*My + 1) ) {
        l1 = (tid % (2*Mx + 1)) - Mx;
        l2 = (tid / (2*Mx + 1)) - My;

        xh[0] = xh0[0] - l1 * Lx;
        xh[1] = xh0[1] - l2 * Ly;
        xh[2] = xh0[2];
  
        imageReal ( A, xh, h, d, e ); 
  
        for ( int i = 0; i < 9; i++ ) { /*  */
          U[(tid & (blockSize - 1)) + (i << bitSize)] += A[i];
          A[i] = 0;
        }         /*---------- end of for loop ----------------*/
        tid += blockSize;
      }

      __syncthreads();
      
      tid =  threadIdx.x + threadIdx.y * blockDim.x;
      uint index = 0;
    
      for ( uint i = 0; i < 9; i++ ) { /* running sum reduction on data */
        index = (tid & (blockSize - 1)) + (i << bitSize);
        // Now do the partial reduction
        if ( blockSize > 512 ) { if ( tid < 512 ) { U[index] += U[index + 512]; } __syncthreads (); } 
        if ( blockSize > 256 ) { if ( tid < 256 ) { U[index] += U[index + 256]; } __syncthreads (); } 
        if ( blockSize > 128 ) { if ( tid < 128 ) { U[index] += U[index + 128]; } __syncthreads (); }
        if ( blockSize >  64 ) { if ( tid <  64 ) { U[index] += U[index +  64]; } __syncthreads (); }
        if ( tid < 32 ) {
          warpReduce ( &U[blockSize*i], tid, blockSize );
        }
      }         /*---------- end of for loop ----------------*/

      __syncthreads();

      while (tid < 9) {
        dA ((tid % 3) + xid * 3, (tid / 3) + yid * 3) += U[tid << bitSize];
        tid += blockSize;
      }
    
      yid += ySize;
      __syncthreads();
    }
    xid += xSize;
    __syncthreads();
  }
} /*}}}*/
/* ----------------  end of CUDA kernel kernelImageReal  ----------------- }}}*/

// === kernelImageFourier: CUDA KERNEL ========================================{{{
//         Name:  kernelImageFourier
//  Description:  Compute the action of one (1) point force on one (1) field
//  point.
//  Each block will compute the matrix corresponding to one (1) field point and
//  one (1) point force.
//  BlockSize must be a power of 2.
// =============================================================================
__global__ void
kernelImageFourier ( /* argument list: {{{*/
    MatrixOnDevice dA
    , MatrixOnDevice dx
    , MatrixOnDevice dx0 
    , double d, double e 
    , int Nx // number of Fourier shells in l1 direction
    , int Ny // number of Fourier shells in l2 direction
    , double Lx, double Ly
    , uint bitSize
    ) /* ------------- end of argument list ------------------------------}}}*/ 
{ /* kernelImageFourier implementation: {{{*/
  uint blockSize = 1 << bitSize;
  assert ( blockSize == blockDim.x * blockDim.y );
  extern __shared__ double U[];
  __shared__ double xh[3];
  double A[9];

  uint xid   =  blockIdx.x;
  uint yid   =  blockIdx.y;
  uint xSize =  gridDim.x;
  uint ySize =  gridDim.y;
  uint tid;
  int l1, l2;

  double h;
  double dl[3] = {0, 0, 0};
  while ((xid < dA.rows()/3)) { 
    yid = blockIdx.y;
    while ((yid < dA.columns()/3)) {
      h = dx0(2, yid);

      tid =  threadIdx.x + threadIdx.y * blockDim.x;

      for ( int i = 0; i < 9; i++ ) { /* initialize data */
        U[tid + i*blockSize] = 0;
        A[i] = 0;
      }         /*---------- end of for loop ----------------*/

      if ( tid == 0 ) { /* compute xh common to all threads in a block */
        xh[0]   = dx(0, xid) - dx0(0, yid);
        xh[1]   = dx(1, xid) - dx0(1, yid);
        xh[2]   = dx(2, xid) - dx0(2, yid);
      }         /*---------- end of if ----------------------*/

      __syncthreads();

      while ( tid < (2*Nx + 1) * (2*Ny + 1) ) {
        l1 = (tid % (2*Nx + 1)) - Nx;
        l2 = (tid / (2*Nx + 1)) - Ny;
        if ((l1 == 0) && (l2 == 0)) {
          imageZeroReg ( A, xh, h, d, true, Lx, Ly ); 
//         printf("At tid = %d and l = (%d, %d) inside GPU\n"
//            "%+20.15e %+20.15e %+20.15e \n"
//            "%+20.15e %+20.15e %+20.15e \n"
//            "%+20.15e %+20.15e %+20.15e \n", tid, l1, l2, A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]);
  
          for ( int i = 0; i < 9; i++ ) { /*  */
            U[(tid & (blockSize - 1)) + (i << bitSize)] += A[i];
            A[i] = 0;
          }         /*---------- end of for loop ----------------*/
          tid += blockSize;
          continue;
        }

        dl[0] = 2*PI*l1/Lx;
        dl[1] = 2*PI*l2/Ly;
        
        imageFourier ( A, xh, dl, h, d, Lx, Ly ); 
//         printf("At tid = %d and l = (%d, %d) inside GPU\n "
//            "%+20.15e %+20.15e %+20.15e \n"
//            "%+20.15e %+20.15e %+20.15e \n"
//            "%+20.15e %+20.15e %+20.15e \n", tid, l1, l2, A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]);

        for ( int i = 0; i < 9; i++ ) { /*  */
          U[(tid & (blockSize - 1)) + (i << bitSize)] += A[i];
          A[i] = 0;
        }         /*---------- end of for loop ----------------*/
        tid += blockSize;
      }

      __syncthreads();
      
      tid =  threadIdx.x + threadIdx.y * blockDim.x;
      uint index;
      
      for ( int i = 0; i < 9; i++ ) { /* running sum reduction on data */
        index = (tid & (blockSize - 1)) + (i << bitSize);
        // Now do the partial reduction
        if ( blockSize > 512 ) { if ( tid < 512 ) { U[index] += U[index + 512]; } __syncthreads (); } 
        if ( blockSize > 256 ) { if ( tid < 256 ) { U[index] += U[index + 256]; } __syncthreads (); } 
        if ( blockSize > 128 ) { if ( tid < 128 ) { U[index] += U[index + 128]; } __syncthreads (); }
        if ( blockSize >  64 ) { if ( tid <  64 ) { U[index] += U[index +  64]; } __syncthreads (); }
        if ( tid < 32 ) {
          warpReduce ( &U[blockSize*i], tid, blockSize );
        }
      }         /*---------- end of for loop ----------------*/

      __syncthreads();

      while (tid < 9) {
        dA ((tid % 3) + xid * 3, (tid / 3) + yid * 3) += U[tid << bitSize];
        tid += blockSize;
      }
//  
//      if (tid == 0)
//        for ( int j = 0; j < 3; j++ ) { /*  */
//          for ( int i = 0; i < 3; i++ ) { /*  */
//            dA( i + xid * 3, j + yid * 3) += U[IDX2D(i, j)*blockSize];
//          }         /*---------- end of for loop ----------------*/
//        }         /*---------- end of for loop ----------------*/
      yid += ySize;
      __syncthreads();
    }
    xid += xSize;
    __syncthreads();
  }
} /*}}}*/
/* ----------------  end of CUDA kernel kernelImageFourier  ----------------- }}}*/

// === imageStokeslet: FUNCTION  =========================================={{{ 
//         Name:  imageStokeslet
//  Description:  Calculate the real space sum
// =============================================================================
void
imageStokeslet ( /* argument list: {{{*/
    MatrixOnDevice dA, MatrixOnDevice dx, MatrixOnDevice dx0
    , const double & d, const double & e
    , const int & M // numRealShells
    , const int & N // numFourierShells
    , const double & Lx, const double & Ly 
    , const int & maxBlockSize = 256
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* imageStokeslet implementation: {{{*/
  assert ((dA.rows() == 3 * dx.columns()) 
      &&  (dA.columns() == 3 * dx0.columns())
      &&  (dx.rows() == 3) && (dx0.rows() == 3));
  assert ( maxBlockSize > 0 );
  dA.setElements(0);
  uint maxBitSize = 0;
  while ((1 << maxBitSize) < maxBlockSize) {
    maxBitSize++;
  }
  if ( (1 << maxBitSize) != maxBlockSize ) {
    printf("maxBlockSize must be a power of 2 and less than 2^9 = 512 for device with compute capability <= 3.0\n");
    maxBitSize--;
    printf("maxBlockSize is reduced to 2^%d = %d\n", maxBitSize, (1 << maxBitSize));
  }
//  assert ((dx.columns() % dimBlockx == 0) && (dx0.columns() % dimBlocky == 0));
  int Mx = M, My = M, Nx = N, Ny = N;
  if (Lx > Ly) {
    My = floor(M*Lx/Ly) + 1;    // for real space sum, we multiply by Lx
    Nx = floor(N*Lx/Ly) + 1;    // for Fourier space sum, we divide by Lx
  }

  if ( Lx < Ly ) { /*  */
    Mx = floor(M*Ly/Lx) + 1;
    Ny = floor(N*Ly/Lx) + 1;
  }         /*---------- end of if ----------------------*/

  uint dimGridx   = chooseSize(dx.columns());
  uint dimGridy   = chooseSize(dx0.columns());
  dim3 dimGrid (dimGridx, dimGridy);

  uint blockSize  = chooseSize(Nx*Ny, maxBitSize);
  dim3 dimBlock (blockSize);
  uint bitSize = 0;
  while ((1 << bitSize) < blockSize) {
    bitSize++;
  }
  if (N >= 0)
  kernelImageFourier <<< dimGrid, dimBlock, blockSize * 9 * sizeof(double) >>> 
    (dA, dx, dx0, d, e, Nx, Ny, Lx, Ly, bitSize);

  blockSize = chooseSize(Mx*My, maxBitSize);
  dimBlock  = dim3(blockSize);
  
  bitSize   = 0;
  while ((1 << bitSize) < blockSize) {
    bitSize++;
  }
  if (M >= 0)
  kernelImageReal <<< dimGrid, dimBlock, blockSize * 9 * sizeof(double) >>> 
    (dA, dx, dx0, d, e, Mx, My, Lx, Ly, bitSize);
} /*}}}*/
/* ---------------  end of function imageStokeslet  -------------------- }}}*/

// === realShells: FUNCTION  =========================================={{{ 
//         Name:  realShells
//  Description:  Compute the outer shell in real space.
//  Assume: L(1) = Lx <= Ly = L(2)
//  Since L(1) < L(2), we can increase M(1) each round and choose 
//  M(2) = ceil(Lx/Ly*M(1)).
// =============================================================================
void
realShells ( /* argument list: {{{*/
    MatrixOnHost & A, MatrixOnHost & absA
    , const MatrixOnHost & x, const MatrixOnHost & x0
    , const MatrixOnHost & M, const MatrixOnHost & oldM
    , const MatrixOnHost & L // Assume that L(1) = Lx < Ly = L(2)
    , const double & d, const double & e
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* realShells implementation: {{{*/
  assert((3*x.columns() == A.rows()) && (3*x0.columns() == A.columns()));
  MatrixOnHost xh(3, 1), xh0(3, 1);
  MatrixOnHost tA(3, 3);
  if ((M(0)==oldM(0)) && (M(1)==oldM(1))) { // when there is NO shell, we use the original force location
//    std::cout << "In side initial realShells at M = " << M(0) << std::endl;
    for (int i = 0; i < x.columns(); i++) {
      for (int j = 0; j < x0.columns(); j++) {
        for (int k = 0; k < 3; k++) 
          xh(k) = x(k, i) - x0(k, j);
        imageReal ( tA, xh, x0(2, j), d, e ); 
        for (int x1 = 0; x1 < 3; x1++) {
          for (int x2 = 0; x2 < 3; x2++) {
            A(3*i + x1, 3*j + x2) = tA(x1, x2);
            absA(3*i + x1, 3*j + x2) = abs(tA(x1, x2));
          }
        }
      }
    }
    return;
  }

  // If there is a shell of finite thickness, 
  for (int i = 0; i < x.columns(); i++) {
    for (int j = 0; j < x0.columns(); j++) {
      for (int k = 0; k < 3; k++) xh0(k) = x(k, i) - x0(k, j);
      // compute left and right cells, assuming that Lx < Ly
      //      Ly
      // |--|--|--|--|
      // | .| .| .| .|
      // |  |  |  |  |
      // |--|--|--|--|  Lx 
      // | .| .| .| .|
      // |  |  |  |  |
      // |--|--|--|--|
      for (int l1 = oldM(0); l1 < M(0); l1++) {
        for (int l2 = -oldM(1); l2 <= oldM(1); l2++) {
          xh(0) = xh0(0) - (l1 + 1)*L(0);
          xh(1) = xh0(1) - l2*L(1);
          xh(2) = xh0(2);
          imageReal ( tA, xh, x0(2, j), d, e ); 
          for (int x1 = 0; x1 < 3; x1++) {
            for (int x2 = 0; x2 < 3; x2++) {
              A(3*i + x1, 3*j + x2) += tA(x1, x2);
              absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
            }
          }

          xh(0) = xh0(0) + (l1 + 1)*L(0);
          xh(2) = xh0(2);
          imageReal ( tA, xh, x0(2, j), d, e ); 
          for (int x1 = 0; x1 < 3; x1++) {
            for (int x2 = 0; x2 < 3; x2++) {
              A(3*i + x1, 3*j + x2) += tA(x1, x2);
              absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
            }
          }
        }
      }

      // Compute top and bottom cells
      for (int l2 = oldM(1); l2 < M(1); l2++) {
        for (int l1 = -M(0); l1 <= M(0); l1++) {
          xh(0) = xh0(0) - l1*L(0);
          xh(1) = xh0(1) - (l2 + 1)*L(1);
          xh(2) = xh0(2);
          imageReal ( tA, xh, x0(2, j), d, e ); 
          for (int x1 = 0; x1 < 3; x1++) {
            for (int x2 = 0; x2 < 3; x2++) {
              A(3*i + x1, 3*j + x2) += tA(x1, x2);
              absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
            }
          }

          xh(1) = xh0(1) + (l2 + 1)*L(1);
          xh(2) = xh0(2);
          imageReal ( tA, xh, x0(2, j), d, e ); 
          for (int x1 = 0; x1 < 3; x1++) {
            for (int x2 = 0; x2 < 3; x2++) {
              A(3*i + x1, 3*j + x2) += tA(x1, x2);
              absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
            }
          }
        }
      }
    }
  }
} /*}}}*/
/* ---------------  end of function realShells  -------------------- }}}*/

// === fourierShells: FUNCTION  =========================================={{{ 
//         Name:  fourierShells
//  Description:  Compute the outer shell in real space.
//  Assume: L(1) = Lx <= Ly = L(2)
//  Since L(1) < L(2), we increase M(2) each round and choose 
//  M(1) = ceil(Lx/Ly*M(2)).
// =============================================================================
void
fourierShells ( /* argument list: {{{*/
    MatrixOnHost & A, MatrixOnHost & absA
    , const MatrixOnHost & x, const MatrixOnHost & x0
    , const MatrixOnHost & M, const MatrixOnHost & oldM
    , const MatrixOnHost & L // Assume that L(1) = Lx < Ly = L(2)
    , const double & d, const double & e
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* fourierShells implementation: {{{*/
  assert((3*x.columns() == A.rows()) && (3*x0.columns() == A.columns()));
  MatrixOnHost xh(3, 1);
  MatrixOnHost tA(3, 3);
  MatrixOnHost l(3, 1);
  if ((M(0)==oldM(0)) && (M(1)==oldM(1))) { // when there is NO shell, we compute zero order term
    for (int i = 0; i < x.columns(); i++) {
      for (int j = 0; j < x0.columns(); j++) {
        for (int k = 0; k < 3; k++) xh(k) = x(k, i) - x0(k, j);
        tA.setElements(0);
        imageZeroReg(tA, xh, x0(2, j), d, true, L(0), L(1));
        for (int x1 = 0; x1 < 3; x1++) {
          for (int x2 = 0; x2 < 3; x2++) {
            A(3*i + x1, 3*j + x2) = tA(x1, x2);
            absA(3*i + x1, 3*j + x2) = abs(tA(x1, x2));
          }
        }
      }
    }
    return;
  }
  for (int i = 0; i < x.columns(); i++) {
    for (int j = 0; j < x0.columns(); j++) {
      for (int k = 0; k < 3; k++) xh(k) = x(k, i) - x0(k, j);

      // Compute top and bottom cells, assuming Lx < Ly
      // In Fourier space, the length is proportional to 1/Lx > 1/Ly
      // => M(0) < M(1).
      //      Ly
      // |--|--|--|--|
      // | .| .| .| .|
      // |  |  |  |  |
      // |--|--|--|--|  Lx 
      // | .| .| .| .|
      // |  |  |  |  |
      // |--|--|--|--|
      for (int l2 = oldM(1) + 1; l2 <= M(1); l2++) {
        for (int l1 = -oldM(0); l1 <= oldM(0); l1++) {
          l(0) = 2*PI*l1/L(0);
          l(1) = 2*PI*l2/L(1);
          imageFourier ( tA, xh, l, x0(2, j), d, L(0), L(1) ); 
//          printf("At tid = %d and l = (%d, %d)\n", tid, l1, l2);
//          tA.print(" and tA is");
          for (int x1 = 0; x1 < 3; x1++) {
            for (int x2 = 0; x2 < 3; x2++) {
              A(3*i + x1, 3*j + x2) += tA(x1, x2);
              absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
            }
          }

          l(1) = -l(1);
          imageFourier ( tA, xh, l, x0(2, j), d, L(0), L(1) ); 
//          printf("At tid = %d and l = (%d, %d)\n", tid, l1, -l2);
//          tA.print(" and tA is");
          for (int x1 = 0; x1 < 3; x1++) {
            for (int x2 = 0; x2 < 3; x2++) {
              A(3*i + x1, 3*j + x2) += tA(x1, x2);
              absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
            }
          }
        }
      }

      // compute left and right cells, assuming that Lx < Ly
      for (int l1 = oldM(0) + 1; l1 <= M(0); l1++) {
        for (int l2 = -M(1); l2 <= M(1); l2++) {
          l(0) = 2*PI*l1/L(0);
          l(1) = 2*PI*l2/L(1);
          imageFourier ( tA, xh, l, x0(2, j), d, L(0), L(1) ); 
//          printf("At tid = %d and l = (%d, %d)\n", tid, l1, l2);
//          tA.print(" and tA is");
          for (int x1 = 0; x1 < 3; x1++) {
            for (int x2 = 0; x2 < 3; x2++) {
              A(3*i + x1, 3*j + x2) += tA(x1, x2);
              absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
            }
          }

          l(0) = -l(0);
          imageFourier ( tA, xh, l, x0(2, j), d, L(0), L(1) ); 
//          printf("At tid = %d and l = (%d, %d)\n", tid, -l1, l2);
//          tA.print(" and tA is");
          for (int x1 = 0; x1 < 3; x1++) {
            for (int x2 = 0; x2 < 3; x2++) {
              A(3*i + x1, 3*j + x2) += tA(x1, x2);
              absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
            }
          }
        }
      }
    }
  }
} /*}}}*/
/* ---------------  end of function fourierShells  -------------------- }}}*/

// === testGPU: CUDA KERNEL ========================================{{{
//         Name:  testGPU
//  Description: test the computational result between GPU and CPU
// =============================================================================
__global__ void
testGPU ( /* argument list: {{{*/
    MatrixOnDevice A, MatrixOnDevice x, MatrixOnDevice x0
    , double d, double Lx, double Ly
    ) /* ------------- end of argument list ------------------------------}}}*/ 
{ /* testGPU implementation: {{{*/
  double xh[3];
  double h = x0(2);
  for (int i = 0; i < 3; i++) xh[i] = x(i) - x0(i);
  imageZeroReg ( A, xh, h, d, true, Lx, Ly ); 
} /*}}}*/
/* ----------------  end of CUDA kernel testGPU  ----------------- }}}*/

// === optimalNumShells: FUNCTION  =========================================={{{ 
//         Name:  optimalNumShells
//  Description: Compute the optimal number of cells in x- and y-directions.
//  Input      : box size L = (Lx, Ly)
//             : blob parameter e
//             : splitting parameter d
//             : tolerance tol
//  Output     : number of cells in each direction M = (Mx, Nx)
//             : computing time for each sum  timeM  = (realSum, fourierSum)
// =============================================================================
void
optimalNumShells ( /* argument list: {{{*/
    MatrixOnHost & M, MatrixOnHost & timeM
    , const MatrixOnHost & L
    , const MatrixOnHost & x0
    , const MatrixOnHost & x
    , const MatrixOnHost & refSol
    , const double & d, const double & e
    , int * numPoints // number of sample points in each direction
    , const double & tol = 1e-15
    , const int & maxShell = 10 
//    , const double & MatrixOnHost x0 = MatrixOnHost(3, 1, 0)
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* optimalNumShells implementation: {{{*/
  MatrixOnHost newM(2, 1), oldM(2, 1);
  double Lx = L(0), Ly = L(1), Lz = L(2);

//  MatrixOnHost x(3, numPoints[0]*numPoints[1]*numPoints[2]);
//  { // set up sample points
//    double dx = Lx/numPoints[0], dy = Ly/numPoints[1], dz = Lz/numPoints[2];
//    for (int i = 0; i < numPoints[0]; i++) {
//      for (int j = 0; j < numPoints[1]; j++) {
//        for (int k = 0; k < numPoints[2]; k++) {
//          x(0, i + j * numPoints[0] + k * numPoints[0] * numPoints[1]) = i * dx;
//          x(1, i + j * numPoints[0] + k * numPoints[0] * numPoints[1]) = j * dy;
//          x(2, i + j * numPoints[0] + k * numPoints[0] * numPoints[1]) = k * dz + 0.005;
//        } 
//      }
//    }
//  }

  MatrixOnHost newA(3*x.columns(), 3*x0.columns());
  MatrixOnHost absA = newA;

  // Compute the exact value using a large number of cells in each directions
//  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
//  realShells ( refSol, absA, x, x0, newM, oldM, L, d, e );
//  fourierShells(A, absA, x, x0, newM, oldM, L, d, e);
//  refSol = refSol + A;
//  newM(0) = newM(1) = maxShell;
//  realShells ( refSol, absA, x, x0, newM, oldM, L, d, e );
//  fourierShells(refSol, absA, x, x0, newM, oldM, L, d, e);
//  newM(0) = newM(1) = maxShell;
//  realShells ( refSol, absA, x, x0, newM, oldM, L, d, e );
//  fourierShells(refSol, absA, x, x0, newM, oldM, L, d, e);

  double err = 10;
  int numLoop = 1;

//  boost::timer::cpu_timer timer;
//  boost::timer::cpu_times elapsed;
  
  clock_t start, end;
  // compute the real space sum
//  timer.start();
//  start = clock();
//  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
//  realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//  for (int i = 0; i < maxShell; i++) {
//    oldM(0) = newM(0); oldM(1) = newM(1);
//    newM(0)++;
//    newM(1) = ceil ( newM(0) * Lx/Ly );
//    realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//  }
//  A = newA;
//  newA.setElements(0);
//  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
//  fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//  for (int i = 0; i < maxShell; i++) {
//    oldM(0) = newM(0); oldM(1) = newM(1);
//    newM(1)++;
//    newM(0) = ceil ( newM(1) * Lx/Ly );
//    fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//  }
//  A = A + newA;
  
  newA.setElements(0);
  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
  realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
  err = 10;
  numLoop = 1;
  while ( (err > tol) && (numLoop < maxShell) ) {
    oldM(0) = newM(0); oldM(1) = newM(1);
    newM(0)++;
    newM(1) = ceil ( newM(0) * Lx/Ly );
    absA.setElements(0);
    realShells ( newA, absA, x, x0, newM, oldM, L, d, e );

    // compute error
    for ( int i = 0; i < absA.length(); i++ ) {
      if ((abs(refSol(i)) > eps) || (absA(i) > eps)) {
        absA(i) = absA(i)/abs(refSol(i));
      }
    }
    err = absA.max();
//    err = 1;
    numLoop++;
  }

  M(0) = oldM(0);     // record time and number of shells for real sum
  std::cout << " Real err is " << err << std::endl;
  
  start = clock();
  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
  realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
  newM(0) = M(0); newM(1) = ceil ( newM(0) * Lx/Ly );
  realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
  end = clock();
//  refSol.print("exact sol is");

//  elapsed = timer.elapsed();
  timeM(0) = 1000.0 * ((double) (end - start)) / CLOCKS_PER_SEC / x.columns();

//  A = newA;
//  MatrixOnHost B = newA;

  // compute the fourier space sum
//  timer.start();
//  start = clock();
  newA.setElements(0);
  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
  fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
  err = 10;
  numLoop = 1;
      double tmp = 0;
  while ( (err > tol) && (numLoop < maxShell) ) {
    oldM(0) = newM(0); oldM(1) = newM(1);
    newM(1)++;
    newM(0) = ceil ( newM(1) * Lx/Ly );
    absA.setElements(0);
    fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );

    // compute error
    for ( int i = 0; i < absA.length(); i++ ) {
      tmp = absA(i);
      if ((abs(refSol(i)) > eps) || (absA(i) > eps)) {
        absA(i) = absA(i)/abs(refSol(i));
      }
      //if (absA(i) > 100) {
        //std::cout << "At numLoop = " << numLoop 
          //<< " and newM = (" << newM(0) << ", " << newM(1) << "): "
          //<< "err(" << i << ") = " << absA(i) 
          //<< " for refSol = " << refSol(i) 
          //<< " and absA = " << tmp << std::endl;
      //}
    }
    err = absA.max();
//    err = 1;
    numLoop++;
  }

  M(1) = oldM(1);  // record time and number of shells for real sum
  std::cout << " Fourier err is " << err << std::endl;
  
  start = clock();
  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
  fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
  newM(1) = M(1); newM(0) = ceil ( newM(1) * Lx/Ly );
  fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
  end = clock();
  
//  elapsed = timer.elapsed();
  timeM(1) = 1000.0 * ((double) (end - start)) / CLOCKS_PER_SEC / (double)x.columns();

//  refSol.print("Exact sol is");
//  A = A + newA;
//  A.print("approx is ");

//  B = B + newA - A;
////  MatrixOnHost B = refSol - A;
// std::cout << "max(refSol - approx.) = " << B.abs().max() << std::endl;
//   for (int i = 0; i < B.length(); i++) {
//     if ((A(i) > es) || (B(i) > eps)) 
//       B(i) = abs(B(i))/abs(A(i));
//   }
// err = B.max();
// std::cout << "error is " << err << std::endl;
// B.print("error is ");

} /*}}}*/
/* ---------------  end of function optimalNumShells  -------------------- }}}*/


#endif
