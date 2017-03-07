#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int ni, int nj, int nk, int nl, int nm, double A[1600][2000], double B[2000][1800], double C[1800][2400], double D[2400][2200]) {
  int i, j;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (double)((i * j + 1) % ni) / (5 * ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (double)((i * (j + 1) + 2) % nj) / (5 * nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (double)(i * (j + 3) % nl) / (5 * nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (double)((i * (j + 2) + 2) % nk) / (5 * nk);
}

static __attribute__ ((noinline)) void kernel_3mm(int ni, int nj, int nk, int nl, int nm, double E[1600][1800], double A[1600][2000], double B[2000][1800], double F[1800][2200], double C[1800][2400], double D[2400][2200], double G[1600][2200]) {
  int i, j, k;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      E[i][j] = 0.0;
      for (k = 0; k < nk; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++) {
      F[i][j] = 0.0;
      for (k = 0; k < nm; ++k)
        F[i][j] += C[i][k] * D[k][j];
    }
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      G[i][j] = 0.0;
      for (k = 0; k < nj; ++k)
        G[i][j] += E[i][k] * F[k][j];
    }
}

int main() {
  int ni = 1600;
  int nj = 1800;
  int nk = 2000;
  int nl = 2200;
  int nm = 2400;
  double(*E)[1600][1800];
  double(*A)[1600][2000];
  double(*B)[2000][1800];
  double(*F)[1800][2200];
  double(*C)[1800][2400];
  double(*D)[2400][2200];
  double(*G)[1600][2200];

  posix_memalign((void**)&E, 64, (1600) * (1800) * sizeof(double));
  posix_memalign((void**)&A, 64, (1600) * (2000) * sizeof(double));
  posix_memalign((void**)&B, 64, (2000) * (1800) * sizeof(double));
  posix_memalign((void**)&F, 64, (1800) * (2200) * sizeof(double));
  posix_memalign((void**)&C, 64, (1800) * (2400) * sizeof(double));
  posix_memalign((void**)&D, 64, (2400) * (2200) * sizeof(double));
  posix_memalign((void**)&G, 64, (1600) * (2200) * sizeof(double));

  init_array(ni, nj, nk, nl, nm, *A, *B, *C, *D);
  DefaultTimer t;
  t.start();
  kernel_3mm(ni, nj, nk, nl, nm, *E, *A, *B, *F, *C, *D, *G);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("G", &(*G)[0][0], ni * nl);

  free((void *)E);
  free((void *)A);
  free((void *)B);
  free((void *)F);
  free((void *)C);
  free((void *)D);
  free((void *)G);
  return 0;
}
