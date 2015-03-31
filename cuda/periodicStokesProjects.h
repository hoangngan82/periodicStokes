/*
 * This file contain all the periodicStokes-related-projects-code.
 */
#ifndef TEST_H_ 
#define TEST_H_
#include "periodicStokes.h"
#include <cuda.h>
#include <math_functions.h>
//#include <cstdlib>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>

// The following parameter will be used in nodal cilia problem.
#define NUM_TIME_STEP         1024
#define NUM_LOOP_PER_SEC         4
#define NUM_TIME_STEP_PER_LOOP 128
#define NUM_REAL_SHELLS          6
#define NUM_FOURIER_SHELLS       6
#define NUM_POINTS              64
#define PHI               (1.0/24)
#define THETA             (1.0/6)
#define LENGTH                 0.5

// Store 3x3 matrix in col-major to be compatible with lapack
#define IDX2D(row, col) (((col)*3) + row)

/*   // Find the intersection of the plane bottom of the cone and the cone axis.
 *   // With the point A(x0, y0, z0) and the plane Ax + By + Cz + D = 0, and H is
 *   // the intersection point, we have
 *   // HA = tn -> t = (Ax0 + By0 + Cz0 + D) / (A^2 + B^2 + C^2);
 *   // -> H(x, y, z) = (x0 - tA, y0 - tB, z0 - tC);
 *   double t = dot (axisUnitVec, (vertex - endPoint));
 *   MatrixOnHost H(3);
 *   for (int i = 0; i < 3; i++) {
 *     H(i) = vertex(i) - t * axisUnitVec (i);
 *   }
 */


// === markerPoints: FUNCTION  =========================================={{{ 
//         Name:  markerPoints
//  Description:  Compute the position of the point on the rod rotating around a
//  tilted axis and the corresponding velocities.
// =============================================================================
void
markerPoints ( /* argument list: {{{*/
    MatrixOnHost & x
    , const MatrixOnHost & e1, const MatrixOnHost & e2, const MatrixOnHost & e3 
    , const MatrixOnHost & H // intersection between axisUnitVec and bottom
    , const double & r 
    , const uint & numPoints = 64 
    , const MatrixOnHost & vertex = MatrixOnHost(3)
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* markerPoints implementation: {{{*/
  assert ((e1.length() == 3) && (e2.length() == 3)
      && (H.length() == 3) 
      && (numPoints > 1));

  uint end = x.columns()/4;
  double dx, dy, dz; 

  dx = (H(0) - vertex(0)) / (numPoints );
  dy = (H(1) - vertex(1)) / (numPoints );
  dz = (H(2) - vertex(2)) / (numPoints );
  
  int i = 0;

  x(0, i) = vertex(0) + (i + 1) * dx;
  x(1, i) = vertex(1) + (i + 1) * dy;
  x(2, i) = vertex(2) + (i + 1) * dz;

  for ( int j = 0; j < 3; j++ ) { /* create three points on the circle centered at x {{{*/
    x(0, j+1) = x(0, 0) + r*cos(2*PI/3*j);
    x(1, j+1) = x(1, 0) + r*sin(2*PI/3*j);
    x(2, j+1) = x(2, 0);
  }         /*---------- end of for loop ----------------}}}*/

  dx = (H(0) + 3*dx - x(0, 0)) / (end-1);
  dy = (H(1) + 3*dy - x(1, 0)) / (end-1);
  dz = (H(2) + 3*dz - x(2, 0)) / (end-1);

  for (i = 1; i < end; i++) {
    x(0, i*4) = x(0, 0) + (i) * dx;
    x(1, i*4) = x(1, 0) + (i) * dy;
    x(2, i*4) = x(2, 0) + (i) * dz;

    for ( int j = 0; j < 3; j++ ) { /* create three points on the circle centered at x {{{*/
      x(0, i*4 + j+1) = x(0, i*4) + r*cos(2*PI/3*j);
      x(1, i*4 + j+1) = x(1, i*4) + r*sin(2*PI/3*j);
      x(2, i*4 + j+1) = x(2, i*4);
    }         /*---------- end of for loop ----------------}}}*/
  }

} /*}}}*/
/* ---------------  end of function markerPoints  -------------------- }}}*/


// === conePoints: FUNCTION  =========================================={{{ 
//         Name:  conePoints
//  Description:  Compute the position of the point on the rod rotating around a
//  tilted axis and the corresponding velocities.
// =============================================================================
void
conePoints ( /* argument list: {{{*/
    MatrixOnHost & x, MatrixOnHost & v
    , const MatrixOnHost & e1, const MatrixOnHost & e2, const MatrixOnHost & e3 
    , const MatrixOnHost & H // intersection between axisUnitVec and bottom
    , const double & r, const double & w, const double & t
    , const uint & numPoints = 64 
    , const MatrixOnHost & vertex = MatrixOnHost(3)
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* conePoints implementation: {{{*/
  assert ((e1.length() == 3) && (e2.length() == 3)
      && (H.length() == 3) 
      && (numPoints > 1)
      && (x.columns() == numPoints) && (v.columns() == numPoints));

  uint end = numPoints - 1;
  MatrixOnHost endPoint(3);
  double dx = cos(w * t + .5), dy = sin(w * t+ .5), dz; 
  for (int i = 0; i < 3; i++) {
    endPoint(i) = H(i) + r * (e1(i) * dx + e2(i) * dy);
    x(i, 0) = vertex(i);
    x(i, end) = endPoint(i);
  }

  dx = (endPoint(0) - vertex(0)) / (numPoints );
  dy = (endPoint(1) - vertex(1)) / (numPoints );
  dz = (endPoint(2) - vertex(2)) / (numPoints );
  for (int i = 0; i < end; i++) {
    x(0, i) = vertex(0) + (i + 1) * dx;
    x(1, i) = vertex(1) + (i + 1) * dy;
    x(2, i) = vertex(2) + (i + 1) * dz;
  }
  for ( int i = 0; i < 3; i++ ) { /*  {{{*/
    endPoint(i) = endPoint(i) - H(i);
  }         /*---------- end of for loop ----------------}}}*/

// Compute the corresponding velocity
  // Cross-product of e3 and endPoint to get velocity of the end point.
  v(0, end) = w * (e3(1) * endPoint(2) - e3(2) * endPoint(1));
  v(1, end) = w * (e3(2) * endPoint(0) - e3(0) * endPoint(2));
  v(2, end) = w * (e3(0) * endPoint(1) - e3(1) * endPoint(0));

  for (int i = 0; i < end; i++) {
    v(0, i) = (double)(i+1) / numPoints * v(0, end);
    v(1, i) = (double)(i+1) / numPoints * v(1, end);
    v(2, i) = (double)(i+1) / numPoints * v(2, end);
  }

} /*}}}*/
/* ---------------  end of function conePoints  -------------------- }}}*/


// === setBox: FUNCTION  =========================================={{{ 
//         Name:  setBox
//  Description:  
//  This will create a box of size nx * ny * nz centered at the center (default
//  to be the origin) with distance between adjacent grid points are dx, dy, 
//  and dz. There will be more points on the 'negative' side of the center if
//  the number of points on that direction is an even number.
// =============================================================================
MatrixOnHost
setBox ( /* argument list: {{{*/
    const uint & nx, const uint & ny, const uint & nz
    , const double & dx, const double & dy, const double & dz
    , MatrixOnHost center = MatrixOnHost(3)
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* setBox implementation: {{{*/
  MatrixOnHost xm(3, nx * ny * nz);
  for ( uint z = 0; z < nz; z++ ) { /* loop through z-layer {{{*/
    for (uint x = 0; x < nx; x++) {
      for (uint y = 0; y < ny; y++) {
        xm(0, x + y*nx + z*nx*ny) = x * dx - (nx >> 1) * dx + center(0);
        xm(1, x + y*nx + z*nx*ny) = y * dy - (ny >> 1) * dy + center(1);
        xm(2, x + y*nx + z*nx*ny) = z * dz - (nz >> 1) * dz + center(2);
      }
    }
  }         /*---------- end of for loop ----------------}}}*/
  return xm;
} /*}}}*/
/* ---------------  end of function setBox  -------------------- }}}*/


// === setBottom: FUNCTION  =========================================={{{ 
//         Name:  setBottom
//  Description:  Create points on the bottom to check no-slip boundary
//  condition.
// =============================================================================
MatrixOnHost
setBottom ( /* argument list: {{{*/
    const double & Lx, const double & Ly 
    , const uint & nx, const uint & ny
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* setBottom implementation: {{{*/
  MatrixOnHost bottom(3, nx * ny);
  double dx = Lx / (nx + 1);
  double dy = Ly / (ny + 1);
  for (uint x = 0; x < nx; x++) {
    for (uint y = 0; y < ny; y++) {
      bottom(0, x + y*nx) = x * dx - Lx/2 + dx;
      bottom(1, x + y*nx) = y * dy - Ly/2 + dx;
    }
  }
  return bottom;
} /*}}}*/
/* ---------------  end of function setBottom  -------------------- }}}*/


// === speedMarker: FUNCTION  =========================================={{{ 
//         Name:  speedMarker
//  Description:  Compute the velocities of marker points.
// =============================================================================
void
speedMarker ( /* argument list: {{{*/
    MatrixOnHost & v, const MatrixOnDevice & dx, const MatrixOnDevice & dx0
    , const MatrixOnHost & f
    , MatrixOnDevice & dA, MatrixOnHost & A
    , const int & numRealShells
    , const int & numFourierShells
    , const double & d, const double & e
    , const double & Lx, const double & Ly
    , const bool & includeReg = true
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* speedMarker implementation: {{{*/
  dA.setElements (0);
  imageStokeslet (dA, dx, dx0, d, e, numRealShells, numFourierShells, Lx, Ly);
  A = dA;
//  printf("imageFourSum: dA is nan = %d and abs(dA).max() = %+20.14e\n", A.isNaN(), abs(dA).max());
  Eigen::Map <Eigen::MatrixXd> ev(v, v.length(), 1);
  Eigen::Map <Eigen::MatrixXd> ef(f, f.length(), 1);
  Eigen::Map <Eigen::MatrixXd> eA(A, A.rows(), A.columns());

  ev = eA * ef;
  
} /*}}}*/
/* ---------------  end of function speedMarker  -------------------- }}}*/


// === moveMarker: FUNCTION  =========================================={{{ 
//         Name:  moveMarker
//  Description:  Advance the marker set according to the current velocity field
//  v and time step dt.
// =============================================================================
void
moveMarker ( /* argument list: {{{*/
    MatrixOnHost & x, const MatrixOnHost v, double dt 
    , const double & Lx = 1, const double & Ly = 1
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* moveMarker implementation: {{{*/
  uint nr = x.rows();
  uint nc = x.columns();
  assert ((nr == 3) && (nr == v.rows()) && (nc == v.columns()));
  double p;
  for (uint i = 0; i < nc; i++) {
    x(0, i) += v(0, i) * dt;
    p = x(0, i) / Lx; 
    p = p + 2*floor(abs(p)) + 1.5;
    p = p - floor(p) - .5;
    x(0, i) = p * Lx;
    x(1, i) += v(1, i) * dt;
    p = x(1, i) / Ly; 
    p = p + 2*floor(abs(p)) + 1.5;
    p = p - floor(p) - .5;
    x(1, i) = p * Ly;
    x(2, i) += v(2, i) * dt;
    if (abs(x(2, i)) < eps)
      x(2, i) = 0;
  }
} /*}}}*/
/* ---------------  end of function moveMarker  -------------------- }}}*/


// === nodalCilia: FUNCTION  =========================================={{{ 
//         Name:  nodalCilia
//  Description:  Model the nodal cilia in a 2D-periodic box of size [Lx, Ly]
//  ranging from -Lx/2 to Lx/2 and from -Ly/2 to Ly/2. The cilia are attached to
//  the wall (z = 0) at the cone vertex. The discretization points on each
//  cilium and their velocities are created using conePoints function.
//  The velocity at any point is computed by evaluating the line integrals of
//  point force distribution on the cilia. Using the trapezoidal rule, since the
//  cilia are rigid, the line integrals can be approximated by a discrete
//  distribution of point forces with modified force strength, i.e, at interior
//  points, forces are multiplied by dx, and at the two end points, forces are
//  multiplied by dx/2. From the code view point, they are still forces.
// =============================================================================
void
nodalCilia ( /* argument list: {{{*/
    const double & w, const double & dt, const uint & numTimeStep 
    , const double & d, const double & e
    , const MatrixOnHost & vertex = MatrixOnHost(3)
    , const int & numRealShells = 8
    , const int & numFourierShells = 8
    , const double & phi = PI/6   // cone angle in radian
    , const double & theta = PI/4 // tilted angle in radian
    , const bool & includeReg = true
    , const double & L = .5       // cilia length
    , const double & Lx = 1, const double & Ly = 1
    , const uint & numPoints = 64
    , const uint & numTimeStepPerLoop = NUM_TIME_STEP_PER_LOOP
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* nodalCilia implementation: {{{*/
  MatrixOnHost e1(3), e2(3), e3(3), H;
  e2(1) = -1;
  e3(0) = sin(theta);
  e3(2) = cos(theta);
  e1(0) = -e3(2);
  e1(2) =  e3(0);
  H     = L * cos(phi) * e3;
  double r = L * sin(phi);
  double t = 0;

  MatrixOnHost x0(3, numPoints), v0(3, numPoints);
  MatrixOnHost A0(3*numPoints, 3*numPoints);
//  MatrixOnHost f0(3*numPoints);
  MatrixOnDevice dx0(3, numPoints), dA0(3*numPoints, 3*numPoints);
  MatrixOnHost* f0 = new MatrixOnHost[numTimeStepPerLoop];

//  uint m_nx = 8 , m_ny = 8 , m_nz = 8;
 uint b_nx = 1, b_ny = 1;
//  double m_dx = .05, m_dy = .05, m_dz =  L / m_nz;
//  MatrixOnHost center(3);
//  center(0) = .3;
//  center(2) = 3*L/4 + m_dz;
//  MatrixOnHost xm = setBox(m_nx, m_ny, m_nz, m_dx, m_dy, m_dz, center);
  MatrixOnHost xm(3, 64);
  markerPoints (xm, e1, e2, e3, H, r, numPoints);
  moveMarker (xm, xm, 0, Lx, Ly);
  
//  center(0) = 0;
  MatrixOnHost xb = setBottom(Lx, Ly, b_nx, b_ny);
  MatrixOnHost Am(3*xm.columns(), 3*numPoints);
  MatrixOnHost Ab(3*xb.columns(), 3*numPoints);
  MatrixOnHost vm(3, xm.columns());
  MatrixOnHost vb(3, xb.columns());
  MatrixOnDevice dxm(3, xm.columns()), dxb(3, xb.columns());
  MatrixOnDevice dAm(3*xm.columns(), 3*numPoints);
  MatrixOnDevice dAb(3*xb.columns(), 3*numPoints);

  std::ofstream bottom("bottom.bin", std::ios::binary);
  std::ofstream marker("marker.bin", std::ios::binary);
  std::ofstream cilia ("cilia.bin" , std::ios::binary);
  std::ofstream vbottom("vbottom.bin", std::ios::binary);
  std::ofstream vmarker("vmarker.bin", std::ios::binary);
  std::ofstream vcilia ("vcilia.bin" , std::ios::binary);
//  std::ofstream abottom("abottom.bin", std::ios::binary);
//  std::ofstream amarker("amarker.bin", std::ios::binary);
//  std::ofstream acilia ("acilia.bin" , std::ios::binary);
  std::ofstream force ("force.bin" , std::ios::binary);

  uint nrows, ncols;
  char *s1 = "==================================================";
  char *s2 = "--------------------------------------------------";
  int loc = 0;
  for ( uint i = 0; i < numTimeStepPerLoop; i++ ) {
    loc = (i+1)*100/numTimeStepPerLoop;
    printf("\rCreating force list: [%-.*s%-.*s%02d%%]", (loc+1)/2, s1, 50-(loc+1)/2, s2, loc);

    t = dt * i; 
    conePoints (x0, v0, e1, e2, e3, H, r, w, t, numPoints, vertex);
//    moveMarker (x0, v0,  0, Lx, Ly);
//    printf ("max of v0 is %+20.14e\n", abs(v0).max());
//    printf ("max of x0 is %+20.14e\n", abs(x0).max());
    dx0 = x0;
    x0.append(cilia);
    v0.append(vcilia);
//    x0.print("value of x0 is");
    dA0.setElements(0);
    imageStokeslet (dA0, dx0, dx0, d, e, numRealShells, numFourierShells, Lx, Ly);
//    A0 = dA0;
//    A0.append(acilia);
//    A0.print("The value of A0  is ");
//    printf ("max of dA0 is %+20.14e\n", abs(dA0).max());
//    A0.write("A0.bin");
//    v0.write("v0.bin");

    nrows = v0.rows();
    ncols = v0.columns();
    f0[i] = dA0.solve(v0.reshape(0, 1));
    f0[i].append(force);
//    if (f0.isNaN()) f0.print("NaN appears on f0");
//    printf ("max of f0 is %+20.14e\n", abs(f0).max());
    v0.reshape(nrows, ncols);
  }

//  uint dimBlockx = chooseSize(numPoints, 4);
//  uint dimBlocky = chooseSize(numPoints, 4);
  std::cout << std::endl;
  uint force_index;
  for (uint i = 0; i < numTimeStep; i++) {
    loc = (i+1)*100/numTimeStep;
    force_index = i % numTimeStepPerLoop;
    printf("\rNodal Cilia: [%-.*s%-.*s%02d%%]", (loc+1)/2, s1, 50-(loc+1)/2, s2, loc);

    // advance marker set
    dxm = xm;
    xm.append(marker);
    speedMarker (vm, dxm, dx0, f0[force_index], dAm, Am, numRealShells, numFourierShells
      , d, e, Lx, Ly, includeReg
    ); /* ------------- end of argument list -----------------------------*/ 
    vm.append(vmarker);
//    Am.append(amarker);
//    if (xm.isNaN()) xm.print("NaN appears on xm");

    moveMarker (xm, vm, dt, Lx, Ly);

    // advance marker set
    dxb = xb;
//    if (xb.isNaN()) xb.print("NaN appears on xb");
//    printf ("max of xb is %+20.14e\n", abs(xb).max());
    xb.append(bottom);
//    printf("bottom speed \n");
    speedMarker (vb, dxb, dx0, f0[force_index], dAb, Ab, numRealShells, numFourierShells
      , d, e, Lx, Ly, includeReg
    ); /* ------------- end of argument list -----------------------------*/ 
    vb.append(vbottom);
//    Ab.append(abottom);
    
    moveMarker (xb, vb, dt, Lx, Ly);
  }
std::cout << std::endl;
  bottom.close();
  marker.close();
  cilia.close();
  vbottom.close();
  vmarker.close();
  vcilia.close();
//  abottom.close();
//  amarker.close();
//  acilia.close() ;
  force.close();
  delete [] f0;
} /*}}}*/
/* ---------------  end of function nodalCilia  -------------------- }}}*/


// === velocityProfile: FUNCTION  =========================================={{{ 
//         Name:  velocityProfile
//  Description:  Compute the velocity on xy-, yz-, and zx-plane for one (1)
//  point force located at the center of the box.
// =============================================================================
void
velocityProfile ( /* argument list: {{{*/
    const double & d,  const double & e 
    , const int & numRealShells 
    , const int & numFourierShells 
    , const bool & includeReg 
    , const double & Lx, const double & Ly, const double & Lz
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* velocityProfile implementation: {{{*/
  uint M = 2;
  uint nx = 16*M, ny = 16*M, nz = 1;
  MatrixOnHost xy, yz, zx;
  MatrixOnHost center(3);
  MatrixOnDevice dx(3, nx*ny), dA(3*nx*ny, 3);
  MatrixOnHost A(3*nx*ny, 3), v(3, nx*ny);
  double Dx, Dy, Dz;
  

  // xy-plane
  nx = 16*M; ny = 16*M; nz = 1;
  Dx = Lx*2*M / (nx - 1);
  Dy = Ly*2*M / (ny - 1);
  Dz = 0;
  center(2) = Lz / 2;
  xy = setBox(nx, ny, nz, Dx, Dy, Dz, center);
  xy.write("xy.bin");

  center(0) = center(1) = Dx/2;
  // yz-plane
  nx = 1; ny = 16*M; nz = 16*M;
  Dz = min(Lz, Lx*4*M) / (nz - 1);
  Dy = Ly*4*M / (ny - 1);
  Dx = 0;
  center(2) = Lz / 2 + Dz;
  yz = setBox(nx, ny, nz, Dx, Dy, Dz, center);
  yz.write("yz.bin");

  // zx-plane
  nx = 16*M; ny = 1; nz = 16*M;
  Dx = Lx*4*M / (nx - 1);
  Dz = min(Lz, Lx*4*M) / (nz - 1);
  Dy = 0;
  center(2) = Lz / 2 + Dz;
  zx = setBox(nx, ny, nz, Dx, Dy, Dz, center);
  zx.write("zx.bin");

  nx = 16*M; ny = 16*M;

//  center(2) = Lz / 2;

//  center(0) = center(1) = 0;
  MatrixOnDevice dx0 = center;
  MatrixOnHost f(3);
  f(0) = 1;
  Eigen::Map <Eigen::MatrixXd> ev(v, v.length(), 1);
  Eigen::Map <Eigen::MatrixXd> ef(f, f.length(), 1);
  Eigen::Map <Eigen::MatrixXd> eA(A, A.rows(), A.columns());
  dA.setElements (0);
  dx = xy;
  imageStokeslet (dA, dx, dx0, d, e, numRealShells, numFourierShells, Lx, Ly);
  A = dA;
  A.print("A is ");

  ev = eA * ef;
  v.write("vxy.bin");

  dA.setElements (0);
  dx = yz;
  imageStokeslet (dA, dx, dx0, d, e, numRealShells, numFourierShells, Lx, Ly);
  A = dA;

  ev = eA * ef;
  v.write("vyz.bin");

  dA.setElements (0);
  dx = zx;
  imageStokeslet (dA, dx, dx0, d, e, numRealShells, numFourierShells, Lx, Ly);
  A = dA;

  ev = eA * ef;
  v.write("vzx.bin");

} /*}}}*/
/* ---------------  end of function velocityProfile  -------------------- }}}*/


// === fallingSphere: FUNCTION  =========================================={{{ 
//         Name:  fallingSphere
//  Description:  Compute the velocity field due to the suspension of a sphere
//  near the wall.
// =============================================================================
void
fallingSphere ( /* argument list: {{{*/
    const double & r,  const double & h
    , const double & d,  const double & e 
    , const int & numRealShells 
    , const int & numFourierShells 
    , const bool & includeReg 
    , const double & Lx, const double & Ly
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* fallingSphere implementation: {{{*/
//  std::ifstream fxs("xs.bin", std::ios::binary);
//  std::ifstream fvs("vs.bin", std::ios::binary);
  MatrixOnHost sphere("pointList.bin"), vs("vs.bin"), quadList("quadList.bin");
  assert ((vs.rows() == 3) || (sphere.rows() == 3));
//  sphere.print("sphere is");
//  vs.print("vs is");
//  fxs.close();
//  fvs.close();
  uint numPoints = sphere.columns();
  sphere = r*sphere;
  quadList = r*r*quadList; 
  double tmp = sphere(0)*sphere(0) + sphere(1)*sphere(1) + sphere(2)*sphere(2);
  tmp = sqrt(tmp);
  printf("radius is %e\n", tmp);
  MatrixOnHost center(3);
  center(2) = h;

  MatrixOnDevice dA0(3*numPoints, 3*numPoints);
  MatrixOnDevice dx0 = sphere;
  printf("after dx0\n");
  imageStokeslet (dA0, dx0, dx0, d, e, numRealShells, numFourierShells, Lx, Ly);

//    nrows = vs.rows();
//    ncols = vs.columns();
  MatrixOnHost  f = dA0.solve(vs.reshape(0, 1));
  printf("after f\n");
  if (f.isNaN())
    printf("f is nan!!!\n");
  f.write("sforce.bin");
  double drag = 0;
  for (int i = 0; i < f.length(); i++) {
    drag += f(i) * quadList(i);
  }
  printf("total force is %e\n", drag);
//  MatrixOnHost A0; A0 = dA0;
//  Eigen::Map <Eigen::MatrixXd> evs(vs, vs.length(), 1);
//  Eigen::Map <Eigen::MatrixXd> ef(f, f.length(), 1);

//  Eigen::Map <Eigen::MatrixXd> eA0(A0, A0.rows(), A0.columns());

//  double Dx, Dy, Dz;
//
//  // yz-plane
//  uint M = 32, nx, ny, nz;
//  nx = 1; ny = M >> 1; nz = M;
//  Dz = min(1.0, h) / (nz - 1);
//  Dy = 1.0 / (ny - 1);
//  Dx = 0;
//  center(2) += 2*Dz;
//  MatrixOnHost yz = setBox(nx, ny, nz, Dx, Dy, Dz, center);
//  yz.write("syz.bin");
//  printf("after yz\n");
//  MatrixOnHost A(3*ny*nz, 3*numPoints), v(3, ny*nz);
//  MatrixOnDevice dA(3*ny*nz, 3*numPoints);
//  MatrixOnDevice dx = yz;
//  imageRealSum (dA, dx, dx0, d, e, numRealShells, dimBlockx, dimBlocky, Lx, Ly);
//  imageFourierSum (dA, dx, dx0, d, numFourierShells, dimBlockx, dimBlocky
//    , includeReg, Lx, Ly
//  ); /* ------------- end of argument list -----------------------------*/ 
//  A = dA;
//  Eigen::Map <Eigen::MatrixXd> eA(A, A.rows(), A.columns());
//  Eigen::Map <Eigen::MatrixXd> ev(v, v.length(), 1);
//
//  ev = eA * ef;
//  evs = eA0 * ef;
//  A0.write("A0.bin");
//  printf("max da0 = %e %e\n", eA0.maxCoeff(), eA0.minCoeff());
//  printf("max evs is %+20.14e\n", ((-1)*vs).max());
//  v.write("vsyz.bin");
//  sphere.reshape(3, 3);
//  vs.reshape(3, 3);
//  sphere.print("sphere is");
//  vs.print("vs is ");
//  A0.reshape(9, 9);
//  f.reshape(3, 3);
//  A0.print("A0 is ");
//  f.print("f is ");
} /*}}}*/
/* ---------------  end of function fallingSphere  -------------------- }}}*/

#endif
