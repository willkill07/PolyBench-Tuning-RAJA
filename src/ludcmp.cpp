#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double A[4000][4000], double b[4000], double x[4000], double y[4000]) {
  int i, j;
  double fn = (double)n;
  for (i = 0; i < n; i++) {
    x[i] = 0;
    y[i] = 0;
    b[i] = (i + 1) / fn / 2.0 + 4;
  }
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++)
      A[i][j] = (double)(-j % n) / n + 1;
    for (j = i + 1; j < n; j++) {
      A[i][j] = 0;
    }
    A[i][i] = 1;
  }
  int r, s, t;
  double(*B)[4000][4000];
  posix_memalign((void**)&B, 64, (4000) * (4000) * sizeof(double));
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (*B)[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
        (*B)[r][s] += A[r][t] * A[s][t];
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      A[r][s] = (*B)[r][s];
  free((void *)B);
}

static __attribute__ ((noinline)) void kernel_ludcmp(int n, double A[4000][4000], double b[4000], double x[4000], double y[4000]) {
  RAJA::forall<Pol_Id_0_Size_1_Parent_null>(RAJA::RangeSegment{0, n}, [=] (int i) {
    RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{0, i}, [=] (int j) {
      RAJA::ReduceSum<Pol_Id_2_Size_1_Parent_1, double> w(0);
      RAJA::forall<Pol_Id_2_Size_1_Parent_1>(RAJA::RangeSegment{0, i}, [=] (int k) {
        w += A[i][k] * A[k][j];
      });
      A[i][j] = (A[i][j] - w) / A[j][j];
    });
    RAJA::forall<Pol_Id_3_Size_1_Parent_0>(RAJA::RangeSegment{i, n}, [=] (int j) {
      RAJA::ReduceSum<Pol_Id_4_Size_1_Parent_3, double> w(0);
      RAJA::forall<Pol_Id_4_Size_1_Parent_3>(RAJA::RangeSegment{0, i}, [=] (int k) {
        w += A[i][k] * A[k][j];
      });
      A[i][j] -= w;
    });
  });
  RAJA::forall<Pol_Id_5_Size_1_Parent_null>(RAJA::RangeSegment{0, n}, [=] (int i) {
    RAJA::ReduceSum<Pol_Id_6_Size_1_Parent_5, double> w(0);
    RAJA::forall<Pol_Id_6_Size_1_Parent_5>(RAJA::RangeSegment{0, i}, [=] (int j) {
      w += A[i][j] * y[j];
    });
    y[i] = b[i] - w;
  });
  RAJA::forall<Pol_Id_7_Size_1_Parent_null>(RAJA::RangeSegment{0, n}, [=] (int ii) {
    int i = (n - 1) - ii;
    RAJA::ReduceSum<Pol_Id_8_Size_1_Parent_7, double> w(0);
    RAJA::forall<Pol_Id_8_Size_1_Parent_7> (RAJA::RangeSegment{i + 1, n}, [=] (int j) {
	w += A[i][j] * x[j];
    });
    x[i] = (y[i] - w) / A[i][i];
  });
}

int main() {
  int n = 4000;
  double(*A)[4000][4000];
  double(*b)[4000];
  double(*x)[4000];
  double(*y)[4000];

  posix_memalign((void**)&A, 64, (4000) * (4000) * sizeof(double));
  posix_memalign((void**)&b, 64, 4000 * sizeof(double));
  posix_memalign((void**)&x, 64, 4000 * sizeof(double));
  posix_memalign((void**)&y, 64, 4000 * sizeof(double));

  init_array(n, *A, *b, *x, *y);

  DefaultTimer t;
  t.start();
  kernel_ludcmp(n, *A, *b, *x, *y);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("x", &(*x)[0], n);

  free((void *)A);
  free((void *)b);
  free((void *)x);
  free((void *)y);
  return 0;
}
