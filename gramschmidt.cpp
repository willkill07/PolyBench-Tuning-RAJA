#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double A[2000][2600], double R[2600][2600], double Q[2000][2600]) {
  int i, j;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((double)((i * j) % m) / m) * 100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}

static __attribute__ ((noinline)) void kernel_gramschmidt(int m, int n, double A[2000][2600], double R[2600][2600], double Q[2000][2600]) {
  int i, j, k;
  double nrm;
  for (k = 0; k < n; k++) {
    nrm = 0.0;
    for (i = 0; i < m; i++)
      nrm += A[i][k] * A[i][k];
    R[k][k] = sqrt(nrm);
    for (i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];
    for (j = k + 1; j < n; j++) {
      R[k][j] = 0.0;
      for (i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (i = 0; i < m; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }
}

int main() {
  int m = 2000;
  int n = 2600;
  double(*A)[2000][2600];
  double(*R)[2600][2600];
  double(*Q)[2000][2600];

  posix_memalign((void**)&A, 64, (2000) * (2600) * sizeof(double));
  posix_memalign((void**)&R, 64, (2600) * (2600) * sizeof(double));
  posix_memalign((void**)&Q, 64, (2000) * (2600) * sizeof(double));

  init_array(m, n, *A, *R, *Q);

  DefaultTimer t;
  t.start();
  kernel_gramschmidt(m, n, *A, *R, *Q);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("A", &(*A)[0][0], m * n);
  dump("R", &(*R)[0][0], n * n);
  dump("Q", &(*Q)[0][0], m * n);

  free((void *)A);
  free((void *)R);
  free((void *)Q);
  return 0;
}
