#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double A[4000], double B[4000]) {
  int i;
  for (i = 0; i < n; i++) {
    A[i] = ((double)i + 2) / n;
    B[i] = ((double)i + 3) / n;
  }
}

static __attribute__ ((noinline)) void kernel_jacobi_1d(int tsteps, int n, double A[4000], double B[4000]) {
  int t, i;
  for (t = 0; t < tsteps; t++) {
    for (i = 1; i < n - 1; i++)
      B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
    for (i = 1; i < n - 1; i++)
      A[i] = 0.33333 * (B[i - 1] + B[i] + B[i + 1]);
  }
}

int main() {
  int n = 4000;
  int tsteps = 1000;
  double(*A)[4000];
  double(*B)[4000];

  posix_memalign((void**)&A, 64, 4000 * sizeof(double));
  posix_memalign((void**)&B, 64, 4000 * sizeof(double));

  init_array(n, *A, *B);

  DefaultTimer t;
  t.start();
  kernel_jacobi_1d(tsteps, n, *A, *B);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("A", &(*A)[0], n);

  free((void *)A);
  free((void *)B);
  return 0;
}
