#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double x1[4000], double x2[4000], double y_1[4000], double y_2[4000], double A[4000][4000]) {
  int i, j;

  if (load_init("x1", x1, n) &&
      load_init("x2", x2, n) &&
      load_init("y_1", y_1, n) &&
      load_init("y_2", y_2, n) &&
      load_init("A", A[0], n * n))
    return;

  for (i = 0; i < n; i++) {
    x1[i] = (double)(i % n) / n;
    x2[i] = (double)((i + 1) % n) / n;
    y_1[i] = (double)((i + 3) % n) / n;
    y_2[i] = (double)((i + 4) % n) / n;
    for (j = 0; j < n; j++)
      A[i][j] = (double)(i * j % n) / n;
  }

  dump_init("x1", x1, n);
  dump_init("x2", x2, n);
  dump_init("y_1", y_1, n);
  dump_init("y_2", y_2, n);
  dump_init("A", A[0], n * n);
}

static __attribute__ ((noinline)) void kernel_mvt(int n, double x1[4000], double x2[4000], double y_1[4000], double y_2[4000], double A[4000][4000]) {
  RAJA::forallN<Pol_Id_0_Size_2_Parent_null>(RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
    x1[i] = x1[i] + A[i][j] * y_1[j];
  });
  RAJA::forallN<Pol_Id_1_Size_2_Parent_null>(RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
    x2[i] = x2[i] + A[j][i] * y_2[j];
  });
}

int main() {
  int n = 4000;
  double(*A)[4000][4000];
  double(*x1)[4000];
  double(*x2)[4000];
  double(*y_1)[4000];
  double(*y_2)[4000];

  posix_memalign((void**)&A, 64, (4000) * (4000) * sizeof(double));
  posix_memalign((void**)&x1, 64, 4000 * sizeof(double));
  posix_memalign((void**)&x2, 64, 4000 * sizeof(double));
  posix_memalign((void**)&y_1, 64, 4000 * sizeof(double));
  posix_memalign((void**)&y_2, 64, 4000 * sizeof(double));

  init_array(n, *x1, *x2, *y_1, *y_2, *A);

  DefaultTimer t;
  t.start();
  kernel_mvt(n, *x1, *x2, *y_1, *y_2, *A);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("x1", &(*x1)[0], n);
  dump("x2", &(*x2)[0], n);

  free((void *)A);
  free((void *)x1);
  free((void *)x2);
  free((void *)y_1);
  free((void *)y_2);
  return 0;
}
