#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double r[4000]) {
  int i;
  for (i = 0; i < n; i++) {
    r[i] = (n + 1 - i);
  }
}

static __attribute__ ((noinline)) void kernel_durbin(int n, double r[4000], double y[4000]) {
  double* z;
  posix_memalign((void**)&z, 64, (4000) * sizeof(double));
  double alpha;
  double beta;
  y[0] = -r[0];
  beta = 1.0;
  alpha = -r[0];

  RAJA::forall<Pol_Id_0_Size_1_Parent_null>(RAJA::RangeSegment{1, n}, [=] (int k) {
    beta = (1 - alpha * alpha) * beta;
    RAJA::ReduceSum<Pol_Id_1_Size_1_Parent_0, double> sum(0);
    RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{0, k}, [=] (int i) {
      sum += r[k - i - 1] * y[i];
    });
    alpha = -(r[k] + sum) / beta;
    RAJA::forall<Pol_Id_2_Size_1_Parent_0>(RAJA::RangeSegment{0, k}, [=] (int i) {
      z[i] = y[i] + alpha * y[k - i - 1];
    });
    RAJA::forall<Pol_Id_3_Size_1_Parent_0>(RAJA::RangeSegment{0, k}, [=] (int i) {
      y[i] = z[i];
    });
    y[k] = alpha;
  });
  free (z);
}

int main() {
  int n = 4000;
  double(*r)[4000];
  double(*y)[4000];

  posix_memalign((void**)&r, 64, 4000 * sizeof(double));
  posix_memalign((void**)&y, 64, 4000 * sizeof(double));

  init_array(n, *r);

  DefaultTimer t;
  t.start();
  kernel_durbin(n, *r, *y);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("y", &(*y)[0], n);

  free((void *)r);
  free((void *)y);
  return 0;
}
