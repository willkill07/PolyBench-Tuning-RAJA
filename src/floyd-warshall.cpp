#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, int path[5600][5600]) {
  int i, j;

  if (load_init("path", path[0], n * n))
    return;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i * j % 7 + 1;
      if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
        path[i][j] = 999;
    }

  dump_init("path", path[0], n * n);
}

static __attribute__ ((noinline)) void kernel_floyd_warshall(int n, int path[5600][5600]) {
  RAJA::forallN<Pol_Id_0_Size_3_Parent_null>(RAJA::RangeSegment{0, n},RAJA::RangeSegment{0, n},RAJA::RangeSegment{0, n}, [=] (int k, int i, int j) {
    path[i][j] = path[i][j] < path[i][k] + path[k][j] ? path[i][j] : path[i][k] + path[k][j];
  });
}

int main() {
  int n = 5600;
  int(*path)[5600][5600];

  posix_memalign((void**)&path, 64, (5600) * (5600) * sizeof(int));

  init_array(n, *path);

  DefaultTimer t;
  t.start();
  kernel_floyd_warshall(n, *path);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("path", &(*path)[0][0], n * n);

  free((void *)path);
  return 0;
}
