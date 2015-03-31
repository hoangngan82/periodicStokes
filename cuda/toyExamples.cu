/*
 * =====================================================================================
 *
 *       Filename:  toyExamples.cu
 *
 *    Description:  This file contains CUDA programming technique that I have
 *    learned and found interesting.
 *
 *        Version:  1.0
 *        Created:  04/01/2014 08:19:19 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Hoang-Ngan Nguyen (), zhoangngan-gmail
 *   Organization:  
 *
 * =====================================================================================
 */

// === warpReduce: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  warpReduce
//  Description:  Sum reduction within a warp.
// =============================================================================
__device__ __host__ void
warpReduce ( /* argument list: {{{*/
    volatile double* data, const int& tid, const int& blockSize 
    ) /* ------------- end of argument list -----------------------------------}}}*/ 
{ /* warpReduce implementation: {{{*/
  if ( blockSize >= 64 ) data [ tid ] += data [ tid + 32 ] ;
  if ( blockSize >= 32 ) data [ tid ] += data [ tid + 16 ] ;
  if ( blockSize >= 16 ) data [ tid ] += data [ tid +  8 ] ;
  if ( blockSize >=  8 ) data [ tid ] += data [ tid +  4 ] ;
  if ( blockSize >=  4 ) data [ tid ] += data [ tid +  2 ] ;
  if ( blockSize >=  2 ) data [ tid ] += data [ tid +  1 ] ;
} /*}}}*/
/* ---------------  end of DEVICE function warpReduce  -------------- }}}*/

// ===  CUDA KERNEL ============================================================
//         Name:  sumReduction
//  Description:  To do reduction on all of the three components of the velocity
//  Requirement: -Every size must be a power of 2.
//               -We only have one block -> gridDim.x = 1 and gridDim.y = 1.
//               -N = 2^m * blockSize where m is some positive integer.
// =============================================================================
__global__ void
sumReduction ( double* out, double* u1g, double* u2g, double* u3g, unsigned int N ) {/*{{{*/
    extern __shared__ double  u[] ;
    unsigned int tid        =   threadIdx.x ;
    unsigned int blockSize  =   blockDim.x * blockDim.y ;   
    unsigned int i          =   blockIdx.x * blockSize * 2 + tid ;
    unsigned int gridSize   =   blockSize * 2 * gridDim.x ;
    double*  u1 = (double*) u ;
    double*  u2 = (double*) &u1 [blockSize] ;
    double*  u3 = (double*) &u2 [blockSize] ;

    u1 [ tid ] = 0 ;
    u2 [ tid ] = 0 ;
    u3 [ tid ] = 0 ;
    

    while ( i < N ) {
        u1 [ tid ]+= u1g [ i ] + u1g [ i + blockSize ] ;
        u2 [ tid ]+= u2g [ i ] + u2g [ i + blockSize ] ;
        u3 [ tid ]+= u3g [ i ] + u3g [ i + blockSize ] ;
        i += gridSize ;
    }
    __syncthreads () ;

    // Now do the partial reduction
    if ( blockSize >= 1024 ) if ( tid < 512 ) {
        u1 [ tid ] += u1 [ tid + 512 ] ;
        u2 [ tid ] += u2 [ tid + 512 ] ;
        u3 [ tid ] += u3 [ tid + 512 ] ;
        __syncthreads () ;
    } 
    if ( blockSize >= 512 ) if ( tid < 256 ) {
        u1 [ tid ] += u1 [ tid + 256 ] ;
        u2 [ tid ] += u2 [ tid + 256 ] ;
        u3 [ tid ] += u3 [ tid + 256 ] ;
        __syncthreads () ;
    } 
    if ( blockSize >= 256 ) if ( tid < 128 ) {
        u1 [ tid ] += u1 [ tid + 128 ] ;
        u2 [ tid ] += u2 [ tid + 128 ] ;
        u3 [ tid ] += u3 [ tid + 128 ] ;
        __syncthreads () ;
    } 
    if ( blockSize >= 128 ) if ( tid < 64 ) {
        u1 [ tid ] += u1 [ tid + 64 ] ;
        u2 [ tid ] += u2 [ tid + 64 ] ;
        u3 [ tid ] += u3 [ tid + 64 ] ;
        __syncthreads () ;
    } 
    if ( tid < 32 ) {
        warpReduce ( u1, tid, blockSize ) ;
        warpReduce ( u2, tid, blockSize ) ;
        warpReduce ( u3, tid, blockSize ) ;
    }

    // Write result to output
    if ( tid == 0 ) {
        out [0] = u1 [0] ;
        out [1] = u2 [0] ;
        out [2] = u3 [0] ;
    }
}		// -----  end of CUDA kernel sumReduction  ----- /*}}}*/

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  singularImage
//  Description:  We will compute the contribution from the image system for the
//  singular Stokeslet corresponding to at point force of strength f located at
//  the point x0 subject to the no-slip boundary condition on the wall z = 0.
// =============================================================================
__device__ __host__ void 
singularImage ( double& u1, double& u2, double& u3, /*{{{*/
        double* xv, double* x0, double* f ) {
    double h = x0[2] ;
    double x = xv[0] - x0[0] ;
    double y = xv[1] - x0[1] ;
    double z = xv[2] - h ;
    double r = sqrt ( x*x + y*y + z*z ) ;
    u1       = 0 ;
    u2       = 0 ;
    u3       = 0 ;
    if ( r < eps ) return ;

    double f1 = f[0] ;
    double f2 = f[1] ;
    double f3 = f[2] ;
    double fDotX = f1*x + f2*y + f3*z ;
    // Stokeslet at the original point
    double H1   = 1 / ( 8 * PI * r ) ;
    double H2   = 1 / ( 8 * PI * r * r * r ) ;
    u1          = H1 * f1 + fDotX * x * H2 ;
    u2          = H1 * f2 + fDotX * y * H2 ;
    u3          = H1 * f3 + fDotX * z * H2 ;

    // Image point
    z           = xv[2] + h ;
    double r2   = sqrt ( x*x + y*y + z*z ) ;
    f1          = -f1 ; 
    f2          = -f2 ; 
    f3          = -f3 ; 
    fDotX       = f1 * x + f2 * y + f3 * z ;
    // Stokeslet at the image point
    H1          = 1 / ( 8 * PI * r ) ;
    H2          = 1 / ( 8 * PI * r * r * r ) ;
    u1         += H1 * f1 + fDotX * x * H2 ;
    u2         += H1 * f2 + fDotX * y * H2 ;
    u3         += H1 * f3 + fDotX * z * H2 ;
    // Stokeslet doublet at the image point
    f3          = -f3 ;
    double H1pr = - 1 / ( 8 * PI * r * r * r ) ;            /* H'(r)/r */
    double H2pr = - 3 / ( 8 * PI * r * r * r * r * r ) ;    /* H'(r)/r */
    fDotX       = f1 * x + f2 * y + f3 * z ;
    double temp = ( f3 * x + z * f1 ) * H2 + z * fDotX * x * H2pr ;
    u1         += 2 * h * temp ;
    temp        = ( f3 * y + z * f2 ) * H2 + z * fDotX * y * H2pr ;
    u2         += 2 * h * temp ;
    temp        = ( f3 * z + z * f3 ) * H2 + z * fDotX * z * H2pr
        + fDotX * H1pr ; 
    u3         += 2 * h * temp ;
    // Potential dipole at the image point
    H1          = -1 / ( 4 * PI * r * r * r ) ;
    H2          =  3 / ( 4 * PI * r * r * r * r * r ) ;
    temp        = f1 * H1 + fDotX * x * H2 ;
    u1         += h * h * temp ;
    temp        = f2 * H1 + fDotX * y * H2 ;
    u2         += h * h * temp ;
    temp        = f3 * H1 + fDotX * z * H2 ;
    u3         += h * h * temp ;
}		// -----  end of DEVICE function singularImage  ----- /*}}}*/

// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  regularizedImage
//  Description:  We will compute the contribution from the image system for the
//  regularized Stokeslet corresponding to at point force of strength f located 
//  at the point x0 subject to the no-slip boundary condition on the wall z = 0.
// =============================================================================
__device__ __host__ void
regularizedImage ( double& u1, double& u2, double& u3,/*{{{*/
        double* xv, double* x0, double* f, const double& d ) {
    double h    = x0[2] ;
    double x    = xv[0] - x0[0] ;
    double y    = xv[1] - x0[1] ;
    double z    = xv[2] - h ;
    double r2   = ( x*x + y*y + z*z ) / d / d ;
    double r    = sqrt ( r2 ) ;
    double expx2= exp ( -r2 ) ;
    double f1   = f[0]  ;
    double f2   = f[1]  ;
    double f3   = f[2]  ;
    double fDotX = f1*x + f2*y + f3*z ;
    // Stokeslet at the original point
    double Erf  = erf ( r ) ;
    double H1   = Erf / ( 8 * PI * r ) + expx2 / 4 / PI / SPI 
        * ( 5 - 8*r2 + 2*r2*r2 ) ;
    H1         /= d ;
    double H2   = Erf / ( 8 * PI * r * r * r ) - expx2 / 4 / PI / SPI
        / r / r * ( 1 - 6*r2 + 2*r2*r2 ) ;
    H2         /= ( d*d*d ) ;
    u1          = H1 * f1 + fDotX * x * H2 ;
    u2          = H1 * f2 + fDotX * y * H2 ;
    u3          = H1 * f3 + fDotX * z * H2 ;

    // Image point
    z           = xv[2] + h ;
    r2          = ( x*x + y*y + z*z ) / d / d  ;
    r           = sqrt ( r2 ) ;
    expx2       = exp ( -r2 ) ;
    Erf         = erf ( r ) ;
    f1          = -f1 ; 
    f2          = -f2 ; 
    f3          = -f3 ; 
    fDotX       = f1 * x + f2 * y + f3 * z ;
    // Stokeslet at the image point
    H1          = Erf / ( 8 * PI * r ) + expx2 / 4 / PI / SPI 
        * ( 5 - 8*r2 + 2*r2*r2 ) ;
    H1         /= d ;
    H2         /= ( d*d*d ) ;
    u1         += H1 * f1 + fDotX * x * H2 ;
    u2         += H1 * f2 + fDotX * y * H2 ;
    u3         += H1 * f3 + fDotX * z * H2 ;
    // Stokeslet doublet at the image point
    f3          = -f3 ;
    double H1pr = - Erf / ( 8 * PI * r2 * r ) ;            /* H'(r)/r */
    H1pr       += expx2 / 4 / PI / SPI / r2 * ( 1 - 26*r2 + 24*r2*r2
            - 4*r2*r2*r2 ) ;
    H1pr       /= ( d*d*d ) ;
    double H2pr = - 3 * Erf / ( 8 * PI * r2 * r2 * r ) ;    /* H'(r)/r */
    H2pr       += expx2 / 8 / PI / SPI * ( 6 + 4*r2 - 32*r2*r2 + 8*r2*r2*r2 ) 
        / ( r2*r2 ) ;
    H2pr       /= ( d*d*d*d*d ) ;
    fDotX       = f1 * x + f2 * y + f3 * z ;
    double temp = ( f3 * x + z * f1 ) * H2 + z * fDotX * x * H2pr ;
    u1         += 2 * h * temp ;
    temp        = ( f3 * y + z * f2 ) * H2 + z * fDotX * y * H2pr ;
    u2         += 2 * h * temp ;
    temp        = ( f3 * z + z * f3 ) * H2 + z * fDotX * z * H2pr
        + fDotX * H1pr ; 
    u3         += 2 * h * temp ;
    // Potential dipole at the image point
    H1          = -Erf / ( 4 * PI * r * r * r ) ;
    H1         += expx2 / 2 / PI / SPI / r / r * ( 1 + 14*r2 - 20*r2*r2
            + 4*r2*r2*r2 ) ;
    H1         /= ( d*d*d ) ;
    H2          =  3 * Erf / ( 4 * PI * r * r * r * r * r ) ;
    H2         -= expx2 * ( 6 + 4*r2 - 32*r2*r2 + 8*r2*r2*r2 ) / PI / SPI
        / 4 / ( r2*r2 ) ;
    H2         /= ( d*d*d*d*d ) ;
    temp        = f1 * H1 + fDotX * x * H2 ;
    u1         += h * h * temp ;
    temp        = f2 * H1 + fDotX * y * H2 ;
    u2         += h * h * temp ;
    temp        = f3 * H1 + fDotX * z * H2 ;
    u3         += h * h * temp ;
    // rotlet at the image point
    temp        = -expx2 * ( 10 - 11*r2 + 2*r2*r2 ) / PI / SPI / ( d*d*d ) / 2 ;
    f1          = f1 ;
    f2          = f2 ;
    u1         += 2 * h * temp * ( -z ) * f1 ;
    u2         += 2 * h * temp * ( -z ) * f2 ;
    u3         += 2 * h * temp * ( x*f1 + y*f2 ) ;
}		// -----  end of DEVICE function regularizedImage  ----- /*}}}*/


// === CUDA DEVICE AND HOST FUNCTION ===========================================
//         Name:  sumFourier
//  Description:  This will compute the sum in Fourier space of the image system
//  of the regularized point force of strength f located at x0.
// =============================================================================
__device__ __host__ void
sumFourier ( double& u1, double& u2, double& u3, double* xv, double* x0, /*{{{*/
        double* f, const int & L1, const int& L2, 
        const double& Lx, const double& Ly, const double& d ) {
    double l1       = 2 * PI * L1 / Lx ;
    double l2       = 2 * PI * L2 / Ly ;
    double h        = x0[2] ;
    // Compute the contribution from the image point
    double x        = ( xv[2] + h ) / d;
    double x2       = x * x ;
    double z2       = ( l1*l1 + l2*l2 ) * d * d ;
    double z        = sqrt (z2) ;
    double expx2    = exp (-x2) ;
    double expz2    = exp (-z2/4) ;
    double experfm  = exp (-x*z) * erfc ( z/2 - x ) ;
    double experfp  = exp ( x*z) * erfc ( z/2 + x ) ;

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

    /* Second derivatives: d2 i(n) = -i(n-1) + z^2 i(n) 
     * Third  derivatives: d3 i(n) = -d i(n-1) + d z^2 i(n)  */

    // Compute cos ( l \cdot yhat ) and sin
    x   = l1*( xv[0] - x0[0] ) + l2*( xv[1] - x0[1] ) ;
    z   = sin (x) ;
    x   = cos (x) ;

    // Compute the contribution from Stokeslet at the image point
    // Compute f1z1 + f2z2
    double f1       = -f[0] ;
    double f2       = -f[1] ;
    double f3       = -f[2] ;
    double Sn       = i2 + i1/4 - i0/16 + i1m/32 ;
    double Sm       = i1 + i0/4 - i1m/16 + i2m/32 ;
    double dSn      = di2 + di1/4 - di0/16 + di1m/32 ;

    double fDotL    = f1 * l1 * d + f2 * l2 * d ;

    double temp ;

    temp    = x * ( -l1 * d * fDotL * Sn + f1 * Sm ) - z * f3 * l1 * d * dSn ;
    u1      = 2 * d * temp ;
    temp    = x * ( -l2 * d * fDotL * Sn + f2 * Sm ) - z * f3 * l2 * d * dSn ;
    u2      = 2 * d * temp ;    
    temp    = x * f3 * z2 * Sn - z * fDotL * dSn ;
    u3      = 2 * d * temp ;

    // Stokeslet doublet 
    f3      = -f3 ;
    temp    = -fDotL * l1 * d * dSn * x - z * f3 * l1 * d * ( z2 * Sn - Sm ) ;
    u1     += 4 * h * temp ;
    temp    = -fDotL * l2 * d * dSn * x - z * f3 * l2 * d * ( z2 * Sn - Sm ) ;
    u2     += 4 * h * temp ;
    temp    = -f3 * dSn * x + z * fDotL * Sn ;
    u3     -= 4 * h * z2 * temp ;


    // Compute the contribution from potential dipole at the image point
    Sn      = i1 + i0/4 + i1m/8 ;
    Sm      = i0 + i1m/4 + i2m/8 ;
    dSn     = di1 + di0/4 + di1m/8 ;

    temp    = x * ( -l1 * d * fDotL * Sn + f1 * Sm ) - z * f3 * l1 * d * dSn ;
    u1     += 2 * h * h / d * temp ;
    temp    = x * ( -l2 * d * fDotL * Sn + f2 * Sm ) - z * f3 * l2 * d * dSn ;
    u2     += 2 * h * h / d * temp ;    
    temp    = x * f3 * z2 * Sn - z * fDotL * dSn ;
    u3     += 2 * h * h / d * temp ;


    // Contribution from the rotlet at the image point
    f1      = -f1 ; 
    f2      = -f2 ; 
    Sn      = i1m - i2m/6 ;
    dSn     = di1m - di2m/6 ;
    u1     += 3 * h / 4 * x * f1 * dSn ;
    u2     += 3 * h / 4 * x * f2 * dSn ;
    u3     -= 3 * h / 4 * z * fDotL * Sn ;

    // Compute the contribution from the original point
    x        = ( xv[2] - h ) / d;
    x2       = x * x;
    z2       = ( l1*l1 + l2*l2 ) * d * d;
    z        = sqrt (z2) ;
    expx2    = exp (-x2) ;
    expz2    = exp (-z2/4) ;
    experfm  = exp (-x*z) * erfc ( z/2 - x ) ;
    experfp  = exp ( x*z) * erfc ( z/2 + x ) ;

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

    /* Second derivatives: d2 i(n) = -i(n-1) + z^2 i(n) 
     * Third  derivatives: d3 i(n) = -d i(n-1) + d z^2 i(n)  */

    // Compute cos ( l \cdot y ) and sin
    x   = l1*( xv[0] - x0[0] ) + l2*( xv[1] - x0[1] ) ;
    z   = sin (x) ;
    x   = cos (x) ;

    // Compute the contribution from Stokeslet at the original point
    fDotL    = f1 * l1 * d + f2 * l2 * d ;
    Sn      = i2 + i1/4 - i0/16 + i1m/32 ;
    Sm      = i1 + i0/4 - i1m/16 + i2m/32 ;
    dSn     = di2 + di1/4 - di0/16 + di1m/32 ;
    temp    = x * ( -l1 * d * fDotL * Sn + f1 * Sm ) 
        - z * f3 * l1 * d * dSn ;
    u1     += 2 * d * temp ;
    temp    = x * ( -l2 * d * fDotL * Sn + f2 * Sm ) - z * f3 * l2 * d * dSn ;
    u2     += 2 * d * temp ;    
    temp    = x * f3 * z2 * Sn - z * fDotL * dSn ;
    u3     += 2 * d * temp ;

    // Adding zeroth-order terms
    // contribution from original point
    x       = ( xv[2] - h ) / d;
    x       = abs ( x ) ;
    x2      = x * x ;
    temp    = SPI / 2 * x2 * ( 2*x2 - 5 ) * exp ( -x2 ) + x * PI * erfc ( x ) ;
    temp   *= d ;
    u1     += f1 * temp ;
    u2     += f2 * temp ;

    // contribution from the image point
    x       = ( xv[2] + h ) / d;
    x2      = x * x ;
    // rotlet
    temp    = 2 * h * SPI * x * ( x2 * ( 2*x2 - 7 ) + 3 ) * exp ( -x2 ) ;
    u1     += f1 * temp ;
    u2     += f2 * temp ;

    // potential dipole
    f3      = -f3 ;
    temp    = -2 * exp ( -x2 ) * SPI * ( x2 * ( 2*x2 -7 ) +3 ) ;
    temp   *= h * h / d ;
    u1     += f1 * temp ;
    u2     += f2 * temp ;

    // Stokeslet
    f1      = -f1 ;
    f2      = -f2 ;
    temp    = SPI / 2 * x2 * ( 2*x2 - 5 ) * exp ( -x2 ) + x * PI * erfc ( x ) ;
    temp   *= d ;
    u1     += f1 * temp ;
    u2     += f2 * temp ;

    u1 /=  ( 2*PI*Lx*Ly ) ;
    u2 /=  ( 2*PI*Lx*Ly ) ;
    u3 /=  ( 2*PI*Lx*Ly ) ;

    // Adding regularization terms according to Pozrikidis' paper
    x       =   abs( xv[2] - h ) - ( xv[2] + h ) ;
    x      *=  4*PI/Lx/Ly ;
    u1     -=  f1*x;
    u2     -=  f2*x;

}		// -----  end of DEVICE function sumFourier  ----- /*}}}*/

// ===  CUDA KERNEL ============================================================
//         Name:  periodicImage
//  Description:  This function computes the periodic image system. Each thread
//  will compute the value in Fourier sum for each pair (l1, l2) and the sum in
//  the real space for each pair (n1, n2).
// =============================================================================
__global__ void
periodicImage ( double* u1g, double* u2g, double* u3g, /*{{{*/
 //       double* xvg, double* x0g, double* fg,
        double* xv, double* x0, double* f,
        double gLx, double gLy, double delta ) {
    __shared__ double u1[1024] ;
    __shared__ double u2[1024] ;
    __shared__ double u3[1024] ;

/*    
    __shared__ double f [3] ;
    __shared__ double xv[3] ;
    __shared__ double x0[3] ;
    for ( int i = 0; i < 3; ++i ) {
        f[i]    = fg[i] ;
        xv[i]   = xvg[i] ;
        x0[i]   = x0g[i] ;
    }
*/

    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x ;
    unsigned int blockSize = blockDim.x * blockDim.y ;

    // initialize velocity to zero's
    while ( tid < 1024 ) { 
        u1 [tid] = 0 ;
        u2 [tid] = 0 ;
        u3 [tid] = 0 ;
        tid     += blockSize ;
    }
    tid = threadIdx.x + threadIdx.y * blockDim.x ;

    double d    = delta ;
    double Lx   = gLx ;
    double Ly   = gLy ;

    int l1   = threadIdx.x - blockDim.x / 2 ;
    int l2   = threadIdx.y + blockIdx.y * blockDim.y - blockDim.x / 2 ;

    double u1Temp, u2Temp, u3Temp ;

    sumFourier ( u1Temp, u2Temp, u3Temp, xv, x0, f, l1, l2, Lx,  Ly, d ) ;
    u1 [ tid ]  = u1Temp ;
    u2 [ tid ]  = u2Temp ;
    u3 [ tid ]  = u3Temp ;
    if ( ( l1 == 0 ) && ( l2 == 0 ) ) {
        u1 [tid] = 0 ;
        u2 [tid] = 0 ;
        u3 [tid] = 0 ;
    }

    x0 [0] += l1 * Lx ; 
    x0 [1] += l2 * Ly ; 
    singularImage ( u1Temp, u2Temp, u3Temp, xv, x0, f ) ;
    u1 [tid] += u1Temp ;
    u2 [tid] += u2Temp ;
    u3 [tid] += u3Temp ;
    regularizedImage ( u1Temp, u2Temp, u3Temp, xv, x0, f, d ) ;
    u1 [tid] -= u1Temp ;
    u2 [tid] -= u2Temp ;
    u3 [tid] -= u3Temp ;


    __syncthreads () ;

    // Now do the partial reduction
    if ( blockSize >= 512 ) if ( tid < 512 ) {
        u1 [ tid ] += u1 [ tid + 512 ] ;
        u2 [ tid ] += u2 [ tid + 512 ] ;
        u3 [ tid ] += u3 [ tid + 512 ] ;
        __syncthreads () ;
    } 
    if ( blockSize >= 256 ) if ( tid < 256 ) {
        u1 [ tid ] += u1 [ tid + 256 ] ;
        u2 [ tid ] += u2 [ tid + 256 ] ;
        u3 [ tid ] += u3 [ tid + 256 ] ;
        __syncthreads () ;
    } 
    if ( blockSize >= 128 ) if ( tid < 128 ) {
        u1 [ tid ] += u1 [ tid + 128 ] ;
        u2 [ tid ] += u2 [ tid + 128 ] ;
        u3 [ tid ] += u3 [ tid + 128 ] ;
        __syncthreads () ;
    } 
    if ( blockSize >= 64 ) if ( tid < 64 ) {
        u1 [ tid ] += u1 [ tid + 64 ] ;
        u2 [ tid ] += u2 [ tid + 64 ] ;
        u3 [ tid ] += u3 [ tid + 64 ] ;
        __syncthreads () ;
    } 
    if ( tid < 32 ) {
        warpReduce ( u1, tid, blockSize ) ;
        warpReduce ( u2, tid, blockSize ) ;
        warpReduce ( u3, tid, blockSize ) ;
    }
    if ( tid == 0 ) {
        u1g [ blockIdx.y ] = u1 [ 0 ] ;
        u2g [ blockIdx.y ] = u2 [ 0 ] ;
        u3g [ blockIdx.y ] = u3 [ 0 ] ;
    }
}		// -----  end of CUDA kernel periodicImage  ----- /*}}}*/

// === periodicMatrix: CUDA KERNEL ========================================{{{
//         Name:  periodicMatrix
//  Description:  Compute the matrix corresponding to N point forces
//  If we have N point forces then the size of x0 is N-by-3 and the size of A is
//  3N-by-3N.
//  For M shells, the number of cells (periodic copies) need to be added is
//  (2*M + 1)^2. This function will only compute all periodic copies of x0's
//  except x0 itself (cell zeroth). The computation for x0 will be taken together with the
//  computation of zeroth-order term in Fourier space and the regularization
//  term. Therefore, the total number of periodic copies for 1 point force in
//  this function are only 
//  (2*M + 1)^2 - 1 = 2M * (2M + 2)  ---- which is divisible by 8.
//  We will compute each 3x3 submatrix on one block of size 
//  blockDim.x = 2*M;
//  blockDim.y = 2*M + 2;
//  by computing 2M * (2M + 2) copies of 3x3 submatrices then use sum reduction
//  on each block to get a single 3x3 matrix.
//  x = threadIdx.x - M + threadIdx.x/M;  ---- x runs through [(-M:-1), (1:M)]
//  y = threadIdx.y - M;                  ---- y runs through (-M:M).
//  if (threadIdx.y = 2M + 1 = blockDim.y - 1) 
//  then y = threadIdx.x - M + threadIdx.x/M
//  and  x = 0. To get a unified operation for all threads, we do as follow
//  {
//    x = threadIdx.x - M + threadIdx.x/M;  ---- x runs through [(-M:-1), (1:M)]
//    y = threadIdx.y - M;                  ---- y runs through (-M:M).
//    temp = y/(M+1);       ---- temp = 1 only if threadIdx.y = 2M + 1.
//    y = temp * x + (1 - temp) * y;
//    x = (1 - temp) * x;
//  }
// =============================================================================
__global__ void
periodicMatrix ( /* argument list: {{{*/
    MatrixOnDevice dA, MatrixOnDevice dx, MatrixOnDevice dx0, 
    double d, double e, 
    int N 
    ) /* ------------- end of argument list -----------------------------------}}}*/ 
{ /* periodicMatrix implementation: {{{*/
  // Test size compatible between A ad x0
/*   if (x0.rows() != 3 || x0.columns() != N 
 *       || A.rows() != 3*N || A.columns() != 3*N) {
 *     printf("Size of x0 must be 3-by-N\n Size of A must be 3N-by-3N\n");
 *     return;
 *   }
 */
//  extern __shared__ double smemPool[];
//  uint tid = threadIdx.x + threadIdx.y * blockDim.x;
//  uint blockSize = blockDim.x * blockDim.y;
//  double* A = (double*) & smemPool[9*tid];
//  double* xh = (double*) & smemPool[9*blockSize + 3*tid];
//  __shared__ double A[9];  // A is a row-major matrix of size 3*N.
//  __shared__ double xh[3];
//  __shared__ double h;
  double A[9];
  double xh[3];
  uint xid = threadIdx.x + blockIdx.x * blockDim.x;
  uint yid = threadIdx.y + blockIdx.y * blockDim.y;
//  printf ("xid = %u and yid = %u : blockIdx.x = %u, blockIdx.y = %u : \
//blockDim.x = %u, blockDim.y = %u, gridDim.x = %u, gridDim.y = %u\n", 
//  xid, yid, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
  double h = dx0(2, yid);
  
//  for ( int l1 = -M; l1 <= M; l1++ ) { /*  {{{*/
//    for ( int l2 = -M; l2 <= M; l2++ ) { /*  {{{*/
//      if ( l1*l1 + l2*l2 <= (M + 2)*(M + 2) ) {
        // Only compute the effect of a periodic copies if its distance from the
        // field point is close enough.
        for ( uint i = 0; i < 3; i++ ) { /*  {{{*/
          xh[i] = dx(i, xid) - dx0(i, yid);
        }         /*---------- end of for loop ----------------}}}*/
//        xh[0] += l1 * Lx;
//        xh[1] += l2 * Ly;

        imageReal ( A, xh, h, d, e ); 
  
        for ( int j = 0; j < 3; j++ ) { /*  {{{*/
          for ( int i = 0; i < 3; i++ ) { /*  {{{*/
            dA( i + xid * 3, j + yid * 3) += A[IDX2D(i, j)];
          }         /*---------- end of for loop ----------------}}}*/
        }         /*---------- end of for loop ----------------}}}*/
//      }
//    }         /*---------- end of for loop ----------------}}}*/
//  }         /*---------- end of for loop ----------------}}}*/
//  printf ("Input: \n");
//  printf ("%+20.14e %+20.14e %+20.14e \n", dl[0], dl[1], dl[2]);
//  dx.print();
//  dx0.print();
//  printf ("Output: \n");
//  dA.print();
} /*}}}*/
/* ----------------  end of CUDA kernel periodicMatrix  ----------------- }}}*/

