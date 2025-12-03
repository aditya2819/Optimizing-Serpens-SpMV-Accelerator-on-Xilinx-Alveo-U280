#ifndef SERPENS_H
#define SERPENS_H

#include <ap_int.h>
#include <cstdint>

constexpr int NUM_CH_SPARSE = 24;

constexpr int WINDOW_SIZE = 8192;
constexpr int DEP_DIST_LOAD_STORE = 10;
constexpr int X_PARTITION_FACTOR = 8;
constexpr int URAM_DEPTH =
    ((NUM_CH_SPARSE == 16) ? 3 : 2) * 4096;

template <typename T, int N>
struct Vec {
  T data[N];

  T& operator[](int idx) { return data[idx]; }
  const T& operator[](int idx) const { return data[idx]; }
};

using float_v16 = Vec<float, 16>;
using float_v8 = Vec<float, 8>;
using float_v2 = Vec<float, 2>;

inline float to_float(ap_uint<32> u) {
  union {
    uint32_t u32;
    float f32;
  } v{};
  v.u32 = static_cast<uint32_t>(u);
  return v.f32;
}

inline ap_uint<32> to_uint(float f) {
  union {
    uint32_t u32;
    float f32;
  } v{};
  v.f32 = f;
  return v.u32;
}

inline float_v16 vec_add(const float_v16& a, const float_v16& b) {
  float_v16 r{};
  for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
    r.data[i] = a.data[i] + b.data[i];
  }
  return r;
}

inline float_v16 vec_mul(const float_v16& a, float s) {
  float_v16 r{};
  for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
    r.data[i] = a.data[i] * s;
  }
  return r;
}

void Serpens(const int* edge_list_ptr,
             const ap_uint<512>* edge_list_ch0,
             const ap_uint<512>* edge_list_ch1,
             const ap_uint<512>* edge_list_ch2,
             const ap_uint<512>* edge_list_ch3,
             const ap_uint<512>* edge_list_ch4,
             const ap_uint<512>* edge_list_ch5,
             const ap_uint<512>* edge_list_ch6,
             const ap_uint<512>* edge_list_ch7,
             const ap_uint<512>* edge_list_ch8,
             const ap_uint<512>* edge_list_ch9,
             const ap_uint<512>* edge_list_ch10,
             const ap_uint<512>* edge_list_ch11,
             const ap_uint<512>* edge_list_ch12,
             const ap_uint<512>* edge_list_ch13,
             const ap_uint<512>* edge_list_ch14,
             const ap_uint<512>* edge_list_ch15,
             const ap_uint<512>* edge_list_ch16,
             const ap_uint<512>* edge_list_ch17,
             const ap_uint<512>* edge_list_ch18,
             const ap_uint<512>* edge_list_ch19,
             const ap_uint<512>* edge_list_ch20,
             const ap_uint<512>* edge_list_ch21,
             const ap_uint<512>* edge_list_ch22,
             const ap_uint<512>* edge_list_ch23,
             const ap_uint<512>* vec_X,
             const ap_uint<512>* vec_Y,
             ap_uint<512>* vec_Y_out,
             const int NUM_ITE, const int NUM_A_LEN, const int M, const int K,
             const int P_N, const int alpha_u, const int beta_u);

#endif
