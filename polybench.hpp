#ifndef POLYBENCH_H_
#define POLYBENCH_H_

#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <cmath>

#include <string>

#include <cxxabi.h>
#include <memory>
#include <chrono>

#include <RAJA/RAJA.hxx>

template <typename Clock = std::chrono::steady_clock>
class Timer {
  std::chrono::time_point<Clock> begin, end;
public:
  void start() {
    begin = Clock::now();
  }
  void stop() {
    end = Clock::now();
  }
  double timeInSeconds() const {
    return std::chrono::duration<double, std::ratio<1>>(end - begin).count();
  }
};

using DefaultTimer = Timer<>;

void dumpTime(double time) {
  char filename[256];
  snprintf(filename, 256, "%s.txt", __BASE_FILE__);
  FILE* fp = fopen(filename, "w");
  fprintf(fp, "%03.12lf\n", time);
  fclose(fp);
  fp = NULL;
}

template <typename T>
static __attribute__((noinline)) void dump (const char* tag, T* data, size_t elems) {
  char filename[256];
  char* demangledType = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
  snprintf(filename, 256, "%s-%s-%s-%zu.bin", __BASE_FILE__, demangledType, tag, elems);
  free(demangledType);
  demangledType = NULL;
  FILE* fp = fopen(filename, "wb");
  fwrite(data, sizeof(T), elems, fp);
  fclose(fp);
  fp = NULL;
}

#endif
