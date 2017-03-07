#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int ni, int nj, int nk, int nl, double *alpha, double *beta, double A[1600][2200], double B[2200][1800], double C[1800][2400], double D[1600][2400]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (double)((i * j + 1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (double)(i * (j + 1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (double)((i * (j + 3) + 1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (double)(i * (j + 2) % nk) / nk;
}

static __attribute__((noinline)) void kernel_2mm(int ni, int nj, int nk, int nl, double alpha, double beta, double tmp[1600][1800], double A[1600][2200], double B[2200][1800], double C[1800][2400], double D[1600][2400]) {
  int i, j, k;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      tmp[i][j] = 0.0;
      for (k = 0; k < nk; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
    }
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      D[i][j] *= beta;
      for (k = 0; k < nj; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
    }
}

int main() {
  int ni = 1600;
  int nj = 1800;
  int nk = 2200;
  int nl = 2400;
  double alpha;
  double beta;
  double(*tmp)[1600][1800];
  double(*A)[1600][2200];
  double(*B)[2200][1800];
  double(*C)[1800][2400];
  double(*D)[1600][2400];

  posix_memalign((void**)&tmp, 64, (1600) * (1800) * sizeof(double));
  posix_memalign((void**)&A, 64, (1600) * (2200) * sizeof(double));
  posix_memalign((void**)&B, 64, (2200) * (1800) * sizeof(double));
  posix_memalign((void**)&C, 64, (1800) * (2400) * sizeof(double));
  posix_memalign((void**)&D, 64, (1600) * (2400) * sizeof(double));

  init_array(ni, nj, nk, nl, &alpha, &beta, *A, *B, *C, *D);

  DefaultTimer t;
  t.start();
  kernel_2mm(ni, nj, nk, nl, alpha, beta, *tmp, *A, *B, *C, *D);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("D", &(*D)[0][0], ni * nl);

  free((void *)tmp);
  free((void *)A);
  free((void *)B);
  free((void *)C);
  free((void *)D);
  return 0;
}
