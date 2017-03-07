#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int tmax, int nx, int ny, double ex[2000][2600], double ey[2000][2600], double hz[2000][2600], double _fict_[1000]) {
  int i, j;
  for (i = 0; i < tmax; i++)
    _fict_[i] = (double)i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      ex[i][j] = ((double)i * (j + 1)) / nx;
      ey[i][j] = ((double)i * (j + 2)) / ny;
      hz[i][j] = ((double)i * (j + 3)) / nx;
    }
}

static __attribute__ ((noinline)) void kernel_fdtd_2d(int tmax, int nx, int ny, double ex[2000][2600], double ey[2000][2600], double hz[2000][2600], double _fict_[1000]) {
  int t, i, j;
  for (t = 0; t < tmax; t++) {
    for (j = 0; j < ny; j++)
      ey[0][j] = _fict_[t];
    for (i = 1; i < nx; i++)
      for (j = 0; j < ny; j++)
        ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i - 1][j]);
    for (i = 0; i < nx; i++)
      for (j = 1; j < ny; j++)
        ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j - 1]);
    for (i = 0; i < nx - 1; i++)
      for (j = 0; j < ny - 1; j++)
        hz[i][j] = hz[i][j] - 0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
  }
}

int main() {
  int tmax = 1000;
  int nx = 2000;
  int ny = 2600;
  double(*ex)[2000][2600];
  double(*ey)[2000][2600];
  double(*hz)[2000][2600];
  double(*_fict_)[1000];

  posix_memalign((void**)&ex, 64, (2000) * (2600) * sizeof(double));
  posix_memalign((void**)&ey, 64, (2000) * (2600) * sizeof(double));
  posix_memalign((void**)&hz, 64, (2000) * (2600) * sizeof(double));
  posix_memalign((void**)&_fict_, 64, 1000 * sizeof(double));

  init_array(tmax, nx, ny, *ex, *ey, *hz, *_fict_);

  DefaultTimer t;
  t.start();
  kernel_fdtd_2d(tmax, nx, ny, *ex, *ey, *hz, *_fict_);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("ex", &(*ex)[0][0], nx * ny);
  dump("ey", &(*ey)[0][0], nx * ny);
  dump("hz", &(*hz)[0][0], nx * ny);

  free((void *)ex);
  free((void *)ey);
  free((void *)hz);
  free((void *)_fict_);
  return 0;
}
