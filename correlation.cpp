#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double *float_n, double data[3000][2600]) {
  int i, j;
  *float_n = (double)m;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      data[i][j] = (double)(i * j) / n + i;
}

static __attribute__ ((noinline)) void kernel_correlation(int m, int n, double float_n, double data[3000][2600], double corr[2600][2600], double mean[2600], double stddev[2600]) {
  double eps = 0.1;
  RAJA::forall<Pol_Id_0_Size_1_Parent_Nil>(RAJA::RangeSegment{0, m}, [=] (int j) {
    RAJA::ReduceSum<Pol_Id_1_Size_1_Parent_0, double> mn(0);
    RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{0, n}, [=] (int i) {
      mn += data[i][j];
    });
    mean[j] = mn / float_n;
  });
  RAJA::forall<Pol_Id_2_Size_1_Parent_Nil>(RAJA::RangeSegment{0, m}, [=] (int j) {
    RAJA::ReduceSum<Pol_Id_3_Size_1_Parent_2, double> stdv(0);
    RAJA::forall<Pol_Id_3_Size_1_Parent_2>(RAJA::RangeSegment{0, n}, [=] (int i) {
      stdv += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    });
    stddev[j] = stdv / float_n;
    stddev[j] = sqrt(stddev[j]);
    stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
  });
  RAJA::forallN<Pol_Id_4_Size_2_Parent_Nil>(RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, m}, [=] (int i, int j) {
    data[i][j] -= mean[j];
    data[i][j] /= sqrt(float_n) * stddev[j];
  });
  RAJA::forall<Pol_Id_5_Size_1_Parent_Nil>(RAJA::RangeSegment{0, m - 1}, [=] (int i) {
    corr[i][i] = 1.0;
    RAJA::forall<Pol_Id_6_Size_1_Parent_5>(RAJA::RangeSegment{i + 1, m}, [=] (int j) {
      RAJA::ReduceSum<Pol_Id_7_Size_1_Parent_6, double> cr(0);
      RAJA::forall<Pol_Id_7_Size_1_Parent_6>(RAJA::RangeSegment{0, n}, [=] (int k) {
        cr += (data[k][i] * data[k][j]);
      });
      corr[j][i] = corr[i][j] = cr;
    });
  });
  corr[m - 1][m - 1] = 1.0;
}

int main() {
  int n = 3000;
  int m = 2600;
  double float_n;
  double(*data)[3000][2600];
  double(*corr)[2600][2600];
  double(*mean)[2600];
  double(*stddev)[2600];

  posix_memalign((void**)&data, 64, (3000) * (2600) * sizeof(double));
  posix_memalign((void**)&corr, 64, (2600) * (2600) * sizeof(double));
  posix_memalign((void**)&mean, 64, 2600 * sizeof(double));
  posix_memalign((void**)&stddev, 64, 2600 * sizeof(double));

  init_array(m, n, &float_n, *data);

  DefaultTimer t;
  t.start();
  kernel_correlation(m, n, float_n, *data, *corr, *mean, *stddev);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("corr", &(*corr)[0][0], m * m);

  free((void *)data);
  free((void *)corr);
  free((void *)mean);
  free((void *)stddev);
  return 0;
}
