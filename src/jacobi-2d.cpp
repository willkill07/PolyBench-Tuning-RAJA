#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double A[2800][2800], double B[2800][2800]) {
  int i, j;

  if (load_init("A", A[0], n * n) &&
      load_init("B", B[0], n * n))
    return;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = ((double)i * (j + 2) + 2) / n;
      B[i][j] = ((double)i * (j + 3) + 3) / n;
    }

  dump_init("A", A[0], n * n);
  dump_init("B", B[0], n * n);
}

static __attribute__ ((noinline)) void kernel_jacobi_2d(int tsteps, int n, double A[2800][2800], double B[2800][2800]) {
  for (int t = 0; t < tsteps; t++) {
    RAJA::forallN<Pol_Id_0_Size_2_Parent_null>(RAJA::RangeSegment{1, n - 1}, RAJA::RangeSegment{1, n - 1}, [=] (int i, int j) {
      B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
    });
    RAJA::forallN<Pol_Id_1_Size_2_Parent_null>(RAJA::RangeSegment{1, n - 1}, RAJA::RangeSegment{1, n - 1}, [=] (int i, int j) {
      A[i][j] = 0.2 * (B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] + B[i - 1][j]);
    });
  }
}

int main() {
  int n = 2800;
  int tsteps = 1000;
  double(*A)[2800][2800];
  double(*B)[2800][2800];

  posix_memalign((void**)&A, 64, (2800) * (2800) * sizeof(double));
  posix_memalign((void**)&B, 64, (2800) * (2800) * sizeof(double));

  init_array(n, *A, *B);

  DefaultTimer t;
  t.start();
  kernel_jacobi_2d(tsteps, n, *A, *B);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("A", &(*A)[0][0], n * n);

  free((void *)A);
  free((void *)B);
  return 0;
}
