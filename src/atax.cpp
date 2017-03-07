#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double A[1800][2200], double x[2200]) {
  int i, j;
  double fn;
  fn = (double)n;
  for (i = 0; i < n; i++)
    x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (double)((i + j) % n) / (5 * m);
}

static __attribute__ ((noinline)) void kernel_atax(int m, int n, double A[1800][2200], double x[2200], double y[2200], double tmp[1800]) {
  RAJA::forall<Pol_Id_0_Size_1_Parent_null>(RAJA::RangeSegment{0, n}, [=] (int i) {
    y[i] = 0;
  });
  RAJA::forall<Pol_Id_1_Size_1_Parent_null>(RAJA::RangeSegment{0, m}, [=] (int i) {
    RAJA::ReduceSum<typename Reduce<Pol_Id_2_Size_1_Parent_1>::type, double> t(0);
    RAJA::forall<Pol_Id_2_Size_1_Parent_1>(RAJA::RangeSegment{0, n}, [=] (int j) {
	t += A[i][j] * x[j];
    });
    tmp[i] = t;
    RAJA::forall<Pol_Id_3_Size_1_Parent_1>(RAJA::RangeSegment{0, n}, [=] (int j) {
      y[j] = y[j] + A[i][j] * tmp[i];
    });
  });
}

int main() {
  int m = 1800;
  int n = 2200;
  double(*A)[1800][2200];
  double(*x)[2200];
  double(*y)[2200];
  double(*tmp)[1800];

  posix_memalign((void**)&A, 64, (1800) * (2200) * sizeof(double));
  posix_memalign((void**)&x, 64, 2200 * sizeof(double));
  posix_memalign((void**)&y, 64, 2200 * sizeof(double));
  posix_memalign((void**)&tmp, 64, 1800 * sizeof(double));

  init_array(m, n, *A, *x);

  DefaultTimer t;
  t.start();
  kernel_atax(m, n, *A, *x, *y, *tmp);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("y", &(*y)[0], n);

  free((void *)A);
  free((void *)x);
  free((void *)y);
  free((void *)tmp);
  return 0;
}
