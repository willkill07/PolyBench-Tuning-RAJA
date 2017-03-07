#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double *alpha, double A[2000][2000], double B[2000][2600]) {
  int i, j;
  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (double)((i + j) % m) / m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (double)((n + (i - j)) % n) / n;
    }
  }
}

static __attribute__ ((noinline)) void kernel_trmm(int m, int n, double alpha, double A[2000][2000], double B[2000][2600]) {
  RAJA::forallN<Pol_Id_0_Size_2_Parent_null>(RAJA::RangeSegment{0, m}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
    RAJA::ReduceSum<typename Reduce<Pol_Id_1_Size_1_Parent_0>::type, double> b(B[i][j]);
    RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{i + 1, m}, [=] (int k) {
      b += A[k][i] * B[k][j];
    });
    B[i][j] = alpha * b;
  });
}

int main() {
  int m = 2000;
  int n = 2600;
  double alpha;
  double(*A)[2000][2000];
  double(*B)[2000][2600];

  posix_memalign((void**)&A, 64, (2000) * (2000) * sizeof(double));
  posix_memalign((void**)&B, 64, (2000) * (2600) * sizeof(double));

  init_array(m, n, &alpha, *A, *B);

  DefaultTimer t;
  t.start();
  kernel_trmm(m, n, alpha, *A, *B);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("B", &(*B)[0][0], m * n);

  free((void *)A);
  free((void *)B);
  return 0;
}
