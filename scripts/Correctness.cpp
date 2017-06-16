#include <cmath>
#include <iomanip>
#include <iostream>
#include <regex>

#include <cstdio>

static const std::regex FILENAME_PARSE{R"((double|int|float|long)-[^-]+-(\d+).bin)", std::regex::optimize};

std::pair<std::string, size_t>
getInformation(const std::string& filename) {
  std::smatch match;
  if (std::regex_search(filename, match, FILENAME_PARSE))
    return {match.str(1), std::stoul(match.str(2))};
  return {"Invalid", 0};
}

template <typename T>
std::unique_ptr<T[]> read(const char* filename, size_t count) {
  FILE* fp = fopen(filename, "rb");
  if (!fp)
    return nullptr;
  auto data = std::make_unique<T[]>(count);
  fread(data.get(), sizeof(T) * count, 1, fp);
  fclose(fp);
  return data;
};

struct Error {
  double absolute{0.0};
  double absolute_max{0.0};
  double relative{1.0};
  double relative_max{0.0};
  size_t count{0};

  Error()             = default;
  Error(const Error&) = default;
  Error(Error&&)      = default;
  Error& operator=(const Error&) = default;

  template <typename T>
  Error(std::unique_ptr<T[]> base, std::unique_ptr<T[]> obs, size_t size) :
    absolute{0.0},
    absolute_max{0.0},
    relative{1.0},
    relative_max{0.0},
    count{size}
  {
    for (size_t i = 0; i < size; ++i) {
      double absdiff = std::abs(obs[i] - base[i]);
      absolute += (absdiff * absdiff);
      absolute_max = std::max(absdiff, absolute_max);
      double ratio = ((obs[i] == 0) ? 1 : obs[i]) / ((base[i] == 0) ? 1 : base[i]);
      relative *= ratio;
      relative_max = std::max(ratio, relative_max);
    }
  }

  Error& operator+=(const Error& other) {
    absolute += other.absolute;
    absolute_max = std::max(absolute_max, other.absolute_max);
    relative *= other.relative;
    relative_max = std::max(relative_max, other.relative_max);
    count += other.count;
    return *this;
  }

  void finalize() {
    absolute = std::sqrt(absolute);
    relative = std::pow(relative, 1.0 / count);
  }

  void print() const {
    printf("\"%s\":%g, \"%s\":%g, \"%s\":%g, \"%s\":%g, \"%s\":%lu",
           "absolute", absolute, "relative", relative, "max_absolute", absolute_max,
           "max_ratio", relative_max, "elements", count);
  }
};

int main(int argc, char* argv[]) {
  if (argc == 1 || (argc & 1) == 0) {
    std::cerr << "Usage: " << argv[0] << " [ <baseline> <observed> ]+\n";
    return EXIT_FAILURE;
  }
  std::vector<Error> errors;
  Error result;
  puts("{\"individual\": [");
  for (int i = 1; i < argc; i += 2) {
    const char* file1     = argv[i];
    const char* file2     = argv[i + 1];
    auto        file1info = getInformation(file1);
    auto        file2info = getInformation(file2);

    if (file1info != file2info) {
      std::cerr << "Error: types do not match: (" << file1 << " and " << file2 << ')' << std::endl;
      return EXIT_FAILURE;
    }

    std::string        dataType = std::get<0>(file1info);
    size_t             dataSize = std::get<1>(file1info);
    switch (dataType[0]) {
      case 'i': // int
        errors.emplace_back(read<int>(file1, dataSize), read<int>(file2, dataSize), dataSize);
        break;
      case 'f': // float
        errors.emplace_back(read<float>(file1, dataSize), read<float>(file2, dataSize), dataSize);
        break;
      case 'd': // double
        errors.emplace_back(read<double>(file1, dataSize), read<double>(file2, dataSize), dataSize);
        break;
    }
    result += errors.back();
    errors.back().finalize();
    printf("{\"expected\":\"%s\", \"actual\":\"%s\", \"data_type\":\"%s\", ", file1, file2, dataType.c_str());
    errors.back().print();
    putchar('}');
    if (i + 2 < argc)
      puts(",");
  }
  puts("], \"aggregated\": {");
  result.finalize();
  result.print();
  puts("}}");
  return EXIT_SUCCESS;
}
