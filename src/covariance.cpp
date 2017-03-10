#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double *float_n, double data[3000][2600]) {
  int i, j;
  *float_n = (double)n;

  if (load_init ("data", data[0], m * n))
    return;

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      data[i][j] = ((double)i * j) / m;

  dump_init ("data", data[0], m * n);
}

static __attribute__ ((noinline)) void kernel_covariance(int m, int n, double float_n, double data[3000][2600], double cov[2600][2600], double mean[2600]) {
  RAJA::forall<Pol_Id_0_Size_1_Parent_null>(RAJA::RangeSegment{0, m}, [=] (int j) {
    RAJA::ReduceSum<typename Reduce<Pol_Id_1_Size_1_Parent_0>::type, double> mn (0);
    RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{0, n}, [=] (int i) {
      mn += data[i][j];
    });
    mean[j] = mn / float_n;
  });
  RAJA::forallN<Pol_Id_2_Size_2_Parent_null>(RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, m}, [=] (int i, int j) {
    data[i][j] -= mean[j];
  });
  RAJA::forallN<Pol_Id_3_Size_2_Parent_null>(RAJA::RangeSegment{0, m}, RAJA::RangeSegment{0, m}, [=] (int i, int j) {
    if (j >= i) {
      RAJA::ReduceSum<typename Reduce<Pol_Id_4_Size_1_Parent_3>::type, double> local_cov(0);
      RAJA::forall<Pol_Id_4_Size_1_Parent_3>(RAJA::RangeSegment{0, n}, [=] (int k) {
        local_cov += data[k][i] * data[k][j];
      });
      cov[j][i] = cov[i][j] = local_cov / (float_n - 1.0);
    }
  });
}

int main() {
  int n = 3000;
  int m = 2600;
  double float_n;
  double(*data)[3000][2600];
  double(*cov)[2600][2600];
  double(*mean)[2600];

  posix_memalign((void**)&data, 64, (3000) * (2600) * sizeof(double));
  posix_memalign((void**)&cov, 64, (2600) * (2600) * sizeof(double));
  posix_memalign((void**)&mean, 64, 2600 * sizeof(double));

  init_array(m, n, &float_n, *data);

  DefaultTimer t;
  t.start();
  kernel_covariance(m, n, float_n, *data, *cov, *mean);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("cov", &(*cov)[0][0], m * m);

  free((void *)data);
  free((void *)cov);
  free((void *)mean);
  return 0;
}
