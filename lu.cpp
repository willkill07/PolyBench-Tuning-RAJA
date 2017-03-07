#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double A[4000][4000]) {
  int i, j;
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

static __attribute__ ((noinline)) void kernel_lu(int n, double A[4000][4000]) {
  RAJA::forall<Pol_Id_0_Size_1_Parent_Nil>(RAJA::RangeSegment{0, n}, [=] (int i) {
    RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{0, i}, [=] (int j) {
      RAJA::ReduceSum<Pol_Id_2_Size_1_Parent_1, double> a(0);
      RAJA::forall<Pol_Id_2_Size_1_Parent_1>(RAJA::RangeSegment{0, j}, [=] (int k) {
	a += A[i][k] * A[k][j];
      });
      A[i][j] = a / A[j][j];
    });
    RAJA::forall<Pol_Id_3_Size_1_Parent_0>(RAJA::RangeSegment{i, n}, [=] (int j) {
      RAJA::ReduceSum<Pol_Id_4_Size_1_Parent_3, double> a(0);
      RAJA::forall<Pol_Id_4_Size_1_Parent_3>(RAJA::RangeSegment{0, k}, [=] (int k) {
         a += A[i][k] * A[k][j];
      })
      A[i][j] -= a;
    });
  });
}

int main() {
  int n = 4000;
  double(*A)[4000][4000];

  posix_memalign((void**)&A, 64, (4000) * (4000) * sizeof(double));

  init_array(n, *A);

  DefaultTimer t;
  t.start();
  kernel_lu(n, *A);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("A", &(*A)[0][0], n * n);

  free((void *)A);
  return 0;
}
