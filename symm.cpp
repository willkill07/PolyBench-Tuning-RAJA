#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double *alpha, double *beta, double C[2000][2600], double A[2000][2000], double B[2000][2600]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (double)((i + j) % 100) / m;
      B[i][j] = (double)((n + i - j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <= i; j++)
      A[i][j] = (double)((i + j) % 100) / m;
    for (j = i + 1; j < m; j++)
      A[i][j] = -999;
  }
}

static __attribute__ ((noinline)) void kernel_symm(int m, int n, double alpha, double beta, double C[2000][2600], double A[2000][2000], double B[2000][2600]) {
  int i, j, k;
  double temp2;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      temp2 = 0;
      for (k = 0; k < i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
  }
}

int main() {
  int m = 2000;
  int n = 2600;
  double alpha;
  double beta;
  double(*C)[2000][2600];
  double(*A)[2000][2000];
  double(*B)[2000][2600];

  posix_memalign((void**)&C, 64, (2000) * (2600) * sizeof(double));
  posix_memalign((void**)&A, 64, (2000) * (2000) * sizeof(double));
  posix_memalign((void**)&B, 64, (2000) * (2600) * sizeof(double));

  init_array(m, n, &alpha, &beta, *C, *A, *B);

  DefaultTimer t;
  t.start();
  kernel_symm(m, n, alpha, beta, *C, *A, *B);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("C", &(*C)[0][0], m * n);

  free((void *)C);
  free((void *)A);
  free((void *)B);
  return 0;
}
