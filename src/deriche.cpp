#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int w, int h, float *alpha, float imgIn[7680][4320]) {
  int i, j;
  *alpha = 0.25;

  if (load_init("imgIn", imgIn[0], w * h))
    return;

  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++)
      imgIn[i][j] = (float)((313 * i + 991 * j) % 65536) / 65535.0f;

  dump_init("imgIn", imgIn[0], w * h);
}

static __attribute__ ((noinline)) void kernel_deriche(int w, int h, float alpha, float imgIn[7680][4320], float imgOut[7680][4320], float y1[7680][4320], float y2[7680][4320]) {
  int i, j;
  float k = (1.0f - expf(-alpha)) * (1.0f - expf(-alpha)) / (1.0f + 2.0f * alpha * expf(-alpha) - expf(2.0f * alpha));
  float a1,a2,a3,a4,a5,a6,a7,a8,b1,b2,c1,c2;
  a1 = a5 = k;
  a2 = a6 = k * expf(-alpha) * (alpha - 1.0f);
  a3 = a7 = k * expf(-alpha) * (alpha + 1.0f);
  a4 = a8 = -k * expf(-2.0f * alpha);
  b1 = powf(2.0f, -alpha);
  b2 = -expf(-2.0f * alpha);
  c1 = c2 = 1;
  RAJA::forall<Pol_Id_0_Size_1_Parent_null>(RAJA::RangeSegment{0, w}, [=] (int i) {
    float xm1_ = 0.0f, ym1_ = 0.0f, ym2_ = 0.0f, *xm1 = &xm1_, *ym1 = &ym1_, *ym2 = &ym2_;
    RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{0, h}, [=] (int j) {
      y1[i][j] = a1 * imgIn[i][j] + a2 * *xm1 + b1 * *ym1 + b2 * *ym2;
      *xm1 = imgIn[i][j];
      *ym2 = *ym1;
      *ym1 = y1[i][j];
    });
  });
  RAJA::forall<Pol_Id_2_Size_1_Parent_null>(RAJA::RangeSegment{0, w}, [=] (int i) {
    float yp1_ = 0.0f, yp2_ = 0.0f, xp1_ = 0.0f, xp2_ = 0.0f, *yp1 = &yp1_, *yp2 = &yp2_, *xp1 = &xp1_, *xp2 = &xp2_;
    RAJA::forall<Pol_Id_3_Size_1_Parent_2>(RAJA::RangeSegment{0, h}, [=] (int jj) {
      int j = h - (jj + 1);
      y2[i][j] = a3 * *xp1 + a4 * *xp2 + b1 * *yp1 + b2 * *yp2;
      *xp2 = *xp1;
      *xp1 = imgIn[i][j];
      *yp2 = *yp1;
      *yp1 = y2[i][j];
    });
  });
  RAJA::forallN<Pol_Id_4_Size_2_Parent_null>(RAJA::RangeSegment{0, w}, RAJA::RangeSegment{0, h}, [=] (int i, int j) {
    imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
  });
  RAJA::forall<Pol_Id_5_Size_1_Parent_null>(RAJA::RangeSegment{0, h}, [=] (int j) {
    float tm1_ = 0.0f, ym1_ = 0.0f, ym2_ = 0.0f, *tm1 = &tm1_, *ym1 = &ym1_, *ym2 = &ym2_;
    RAJA::forall<Pol_Id_6_Size_1_Parent_5>(RAJA::RangeSegment{0, w}, [=] (int i) {
      y1[i][j] = a5 * imgOut[i][j] + a6 * *tm1 + b1 * *ym1 + b2 * *ym2;
      *tm1 = imgOut[i][j];
      *ym2 = *ym1;
      *ym1 = y1[i][j];
    });
  });
  RAJA::forall<Pol_Id_7_Size_1_Parent_null>(RAJA::RangeSegment{0, h}, [=] (int j) {
    float tp1_= 0.0f, tp2_ = 0.0f, yp1_ = 0.0f, yp2_ = 0.0f, *tp1 = &tp1_, *tp2 = &tp2_, *yp1 = &yp1_, *yp2 = &yp2_;
    RAJA::forall<Pol_Id_8_Size_1_Parent_7>(RAJA::RangeSegment{0, w}, [=] (int ii) {
      int i = w - (ii + 1);
      y2[i][j] = a7 * *tp1 + a8 * *tp2 + b1 * *yp1 + b2 * *yp2;
      *tp2 = *tp1;
      *tp1 = imgOut[i][j];
      *yp2 = *yp1;
      *yp1 = y2[i][j];
    });
  });
  RAJA::forallN<Pol_Id_9_Size_2_Parent_null>(RAJA::RangeSegment{0, w}, RAJA::RangeSegment{0, h}, [=] (int i, int j) {
    imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
  });
}

int main() {
  int w = 7680;
  int h = 4320;
  float alpha;
  float(*imgIn)[7680][4320];
  float(*imgOut)[7680][4320];
  float(*y1)[7680][4320];
  float(*y2)[7680][4320];

  posix_memalign((void**)&imgIn, 64, (7680) * (4320) * sizeof(float));
  posix_memalign((void**)&imgOut, 64, (7680) * (4320) * sizeof(float));
  posix_memalign((void**)&y1, 64, (7680) * (4320) * sizeof(float));
  posix_memalign((void**)&y2, 64, (7680) * (4320) * sizeof(float));

  init_array(w, h, &alpha, *imgIn);

  DefaultTimer t;
  t.start();
  kernel_deriche(w, h, alpha, *imgIn, *imgOut, *y1, *y2);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("imgOut", &(*imgOut)[0][0], w * h);

  free((void *)imgIn);
  free((void *)imgOut);
  free((void *)y1);
  free((void *)y2);
  return 0;
}
