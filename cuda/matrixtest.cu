/*
 * =====================================================================================
 *
 *       Filename:  matrixtest.cu
 *
 *    Description:  test matrix.h and mystdlib.h
 *
 *        Version:  1.0
 *        Created:  12/26/2013 05:51:39 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Hoang-Ngan Nguyen (), zhoangngan-gmail
 *   Organization:  
 *
 * =====================================================================================
 */

#include "matrix.h"
#include <cstdlib>  /* srand, rand */
#include <ctime>    /* time */
#include <cstring>
#include <fstream>
#include <iostream>
//#include <eigen3/Eigen/Core>
#include <sys/times.h>
using namespace std;
//using namespace Eigen;

// === constructTest: CUDA KERNEL ========================================{{{
//         Name:  constructTest
//  Description: 
// =============================================================================
__global__ void
constructTest ( /* argument list: {{{*/
    MatrixOnDevice A, MatrixOnDevice B, MatrixOnDevice C 
    ) /* ------------- end of argument list ------------------------------}}}*/ 
{ /* constructTest implementation: {{{*/
  uint tid = threadIdx.x;
  A(tid) = B(tid) + C(tid);
  printf("tid = %d\n", tid);
} /*}}}*/
/* ----------------  end of CUDA kernel constructTest  ----------------- }}}*/

int main( int argc, char *argv[] ) {
  
  cout << "construcTest" << endl;
  MatrixOnHost A(3, 1, 20), B(3);
  A.print("A is");
  MatrixOnDevice dB(3), dC(3, 1, 1);
  B(1) = 1; B(2) = 2;
  dB = B;
  B.print("B is");
  cout << "before constructTest " << endl;
  constructTest <<< 1, 3 >>> (dB, dB, dC);
  cout << "after constructTest " << endl;
  B = dB;
  B.print("B is");
  A.print("A is");
  MatrixOnHost C = B;
  C.print("C is");
  cout << "address of C is " << (long)&C(0) << " and B is " << (long)&B(0) << endl;

	cout << "End of program!!!!!!!!!!!!!!" << endl;

	return EXIT_SUCCESS;
}				// ----------  end of function main  ----------

