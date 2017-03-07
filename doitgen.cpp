#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int nr, int nq, int np, double A[250][220][270], double C4[270][270]) {
  int i, j, k;
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
        A[i][j][k] = (double)((i * j + k) % np) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (double)(i * j % np) / np;
}


void kernel_doitgen(int nr, int nq, int np, double A[250][220][270], double C4[270][270], double sum[270]) {
  int r, q, p, s;
  for (r = 0; r < nr; r++) {
    for (q = 0; q < nq; q++) {
      for (p = 0; p < np; p++) {
        sum[p] = 0.0;
        for (s = 0; s < np; s++) {
          sum[p] += A[r][q][s] * C4[s][p];
        }
      }
      for (p = 0; p < np; p++) {
        A[r][q][p] = sum[p];
      }
    }
  }
}

int main() {
  int nr = 250;
  int nq = 220;
  int np = 270;
  double(*A)[250][220][270];
  double(*sum)[270];
  double(*C4)[270][270];

  posix_memalign((void**)&A, 64, (250) * (220) * (270) * sizeof(double));
  posix_memalign((void**)&sum, 64, 270 * sizeof(double));
  posix_memalign((void**)&C4, 64, (270) * (270) * sizeof(double));

  init_array(nr, nq, np, *A, *C4);

  DefaultTimer t;
  t.start();
  kernel_doitgen(nr, nq, np, *A, *C4, *sum);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("A", &(*A)[0][0][0], nr * nq * np);

  free((void *)A);
  free((void *)sum);
  free((void *)C4);
  return 0;
}
