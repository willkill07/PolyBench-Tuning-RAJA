#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double A[4000][4000]) {
  int i, j;

  if (load_init("A", A[0], n * n))
    return;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((double)i * (j + 2) + 2) / n;

  dump_init("A", A[0], n * n);
}

static __attribute__ ((noinline)) void kernel_seidel_2d(int tsteps, int n, double A[4000][4000]) {
  RAJA::forallN<Pol_Id_0_Size_3_Parent_null>(RAJA::RangeSegment{0, tsteps}, RAJA::RangeSegment{1, n - 1}, RAJA::RangeSegment{1, n - 1}, [=] (int, int i, int j) {
    A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
  });
}

int main() {
  int n = 4000;
  int tsteps = 1000;
  double(*A)[4000][4000];

  posix_memalign((void**)&A, 64, (4000) * (4000) * sizeof(double));

  init_array(n, *A);

  DefaultTimer t;
  t.start();
  kernel_seidel_2d(tsteps, n, *A);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("A", &(*A)[0][0], n * n);

  free((void *)A);
  return 0;
}
