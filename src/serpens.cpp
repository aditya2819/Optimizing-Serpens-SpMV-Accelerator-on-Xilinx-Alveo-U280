#include <ap_int.h>
#include <cassert>
#include <hls_stream.h>

#include "serpens.h"

constexpr int FIFO_DEPTH = 2;

const int NUM_CH_SPARSE_div_8 = NUM_CH_SPARSE / 8;
const int NUM_CH_SPARSE_mult_16 = NUM_CH_SPARSE * 16;
const int NUM_CH_SPARSE_mult_2 = NUM_CH_SPARSE * 2;
const int WINDOW_SIZE_div_16 = WINDOW_SIZE >> 4;
static_assert(NUM_CH_SPARSE_div_8 == 3,
              "Arbiter expects 3 channels per group (NUM_CH_SPARSE=24).");

struct MultXVec {
  Vec<ap_uint<18>, 8> row;
  float_v8 axv;
};

static float_v16 unpack_float_v16(const ap_uint<512>& word) {
  float_v16 out{};
  for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
    ap_uint<32> part = word(32 * i + 31, 32 * i);
    out.data[i] = to_float(part);
  }
  return out;
}

static ap_uint<512> pack_float_v16(const float_v16& v) {
  ap_uint<512> word = 0;
  for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
    word(32 * i + 31, 32 * i) = to_uint(v.data[i]);
  }
  return word;
}

void read_edge_list_ptr(const int num_ite, const int M, const int P_N,
                        const int K, const int* edge_list_ptr,
                        hls::stream<int>& PE_inst) {
  const int rp_time = (P_N == 0) ? 1 : P_N;

  PE_inst.write(num_ite);
  PE_inst.write(M);
  PE_inst.write(rp_time);
  PE_inst.write(K);

  const int num_ite_plus1 = num_ite + 1;
  for (int rp = 0; rp < rp_time; ++rp) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    for (int i = 0; i < num_ite_plus1; ++i) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
      PE_inst.write(edge_list_ptr[i]);
    }
  }
}

void read_X(const int P_N, const int K, const ap_uint<512>* vec_X,
            hls::stream<float_v16>& fifo_X) {
  const int rp_time = (P_N == 0) ? 1 : P_N;
  const int num_ite_X = (K + 15) >> 4;

  for (int rp = 0; rp < rp_time; ++rp) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    for (int i = 0; i < num_ite_X; ++i) {
#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
      float_v16 x = unpack_float_v16(vec_X[i]);
      fifo_X.write(x);
    }
  }
}

void read_A(const int P_N, const int A_len, const ap_uint<512>* A,
            hls::stream<ap_uint<512>>& fifo_A) {
  const int rp_time = (P_N == 0) ? 1 : P_N;
  for (int rp = 0; rp < rp_time; ++rp) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    for (int i = 0; i < A_len; ++i) {
#pragma HLS loop_tripcount min=1 max=10000
#pragma HLS pipeline II=1
      fifo_A.write(A[i]);
    }
  }
}

void PEG_Xvec(hls::stream<int>& fifo_inst_in,
              hls::stream<ap_uint<512>>& fifo_A,
              hls::stream<float_v16>& fifo_X_in,
              hls::stream<int>& fifo_inst_out,
              hls::stream<float_v16>& fifo_X_out,
              hls::stream<int>& fifo_inst_out_to_Yvec,
              hls::stream<MultXVec>& fifo_aXvec) {
  const int NUM_ITE = fifo_inst_in.read();
  const int M = fifo_inst_in.read();
  const int rp_time = fifo_inst_in.read();
  const int K = fifo_inst_in.read();

  fifo_inst_out.write(NUM_ITE);
  fifo_inst_out.write(M);
  fifo_inst_out.write(rp_time);
  fifo_inst_out.write(K);

  fifo_inst_out_to_Yvec.write(NUM_ITE);
  fifo_inst_out_to_Yvec.write(M);
  fifo_inst_out_to_Yvec.write(rp_time);

  for (int rp = 0; rp < rp_time; ++rp) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    float local_X[4][WINDOW_SIZE];
#pragma HLS bind_storage variable = local_X type = RAM_2P impl = URAM latency = 2
#pragma HLS array_partition variable = local_X complete dim = 1
#pragma HLS array_partition variable = local_X cyclic factor = X_PARTITION_FACTOR dim = 2

    int start_32 = fifo_inst_in.read();
    fifo_inst_out.write(start_32);
    fifo_inst_out_to_Yvec.write(start_32);

    for (int i = 0; i < NUM_ITE; ++i) {
#pragma HLS loop_tripcount min=1 max=49
      const int num_x_words = ((K + 15) >> 4) - i * WINDOW_SIZE_div_16;
      const int read_limit =
          (num_x_words < WINDOW_SIZE_div_16) ? num_x_words : WINDOW_SIZE_div_16;

      for (int j = 0; j < read_limit; ++j) {
#pragma HLS loop_tripcount min=1 max=512
#pragma HLS pipeline II=1
        float_v16 x = fifo_X_in.read();
        fifo_X_out.write(x);
        for (int kk = 0; kk < 16; ++kk) {
#pragma HLS unroll
          for (int l = 0; l < 4; ++l) {
#pragma HLS unroll
            local_X[l][(j << 4) + kk] = x.data[kk];
          }
        }
      }

      const int end_32 = fifo_inst_in.read();
      fifo_inst_out.write(end_32);
      fifo_inst_out_to_Yvec.write(end_32);

      for (int j = start_32; j < end_32; ++j) {
#pragma HLS loop_tripcount min=1 max=200
#pragma HLS pipeline II=1
        ap_uint<512> a_pes = fifo_A.read();
        MultXVec raxv{};
        for (int p = 0; p < 8; ++p) {
#pragma HLS unroll
          ap_uint<64> a = a_pes(63 + p * 64, p * 64);
          ap_uint<14> a_col = a(63, 50);
          ap_uint<18> a_row = a(49, 32);
          ap_uint<32> a_val = a(31, 0);

          raxv.row.data[p] = a_row;
          raxv.axv.data[p] = 0.0f;
          if (a_row[17] == 0) {
            float a_val_f = to_float(a_val);
            raxv.axv.data[p] = a_val_f * local_X[p / 2][a_col];
          }
        }
        fifo_aXvec.write(raxv);
      }
      start_32 = end_32;
    }
  }
}

inline void PUcore_Ymtx(ap_uint<18> addr_c, float val_d0_f,
                        ap_uint<64> local_C_pe0[URAM_DEPTH]) {
#pragma HLS inline
  ap_uint<64> c_val_u64 = local_C_pe0[addr_c(17, 1)];

  ap_uint<32> c_val_d0_u = c_val_u64(31, 0);
  ap_uint<32> c_val_d1_u = c_val_u64(63, 32);

  ap_uint<32> c_val_u = (addr_c[0]) ? c_val_d1_u : c_val_d0_u;

  float c_val_plus_d0_f = to_float(c_val_u) + val_d0_f;

  c_val_u = to_uint(c_val_plus_d0_f);

  if (addr_c[0]) {
    c_val_d1_u = c_val_u;
  } else {
    c_val_d0_u = c_val_u;
  }

  c_val_u64(63, 32) = c_val_d1_u;
  c_val_u64(31, 0) = c_val_d0_u;
  local_C_pe0[addr_c(17, 1)] = c_val_u64;
}

void PEG_Yvec(hls::stream<int>& fifo_inst_in,
              hls::stream<MultXVec>& fifo_aXvec,
              hls::stream<float_v2>& fifo_Y_out) {
  const int NUM_ITE = fifo_inst_in.read();
  const int M = fifo_inst_in.read();
  const int rp_time = fifo_inst_in.read();

  const int num_v_init = (M + NUM_CH_SPARSE_mult_16 - 1) / NUM_CH_SPARSE_mult_16;
  const int num_v_out = (M + NUM_CH_SPARSE_mult_2 - 1) / NUM_CH_SPARSE_mult_2;

  ap_uint<64> local_C[8][URAM_DEPTH];
#pragma HLS bind_storage variable = local_C type = RAM_2P impl = URAM latency = 1
#pragma HLS array_partition complete variable = local_C dim = 1

  for (int rp = 0; rp < rp_time; ++rp) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    for (int i = 0; i < num_v_init; ++i) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
      for (int p = 0; p < 8; ++p) {
#pragma HLS unroll
        local_C[p][i] = 0;
      }
    }

    int start_32 = fifo_inst_in.read();

    for (int i = 0; i < NUM_ITE; ++i) {
#pragma HLS loop_tripcount min=1 max=49
      const int end_32 = fifo_inst_in.read();

      for (int j = start_32; j < end_32; ++j) {
#pragma HLS loop_tripcount min=1 max=200
#pragma HLS pipeline II=1
#pragma HLS dependence true variable = local_C distance = DEP_DIST_LOAD_STORE
        MultXVec raxv = fifo_aXvec.read();
        for (int p = 0; p < 8; ++p) {
#pragma HLS unroll
          auto a_row = raxv.row.data[p];
          if (a_row[17] == 0) {
            PUcore_Ymtx(a_row, raxv.axv.data[p], local_C[p]);
          }
        }
      }
      start_32 = end_32;
    }

    for (int i = 0, c_idx = 0; i < num_v_out; ++i) {
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1
      float_v2 out_v{};
      ap_uint<64> u_64 = local_C[c_idx][i >> 3];
      for (int d = 0; d < 2; ++d) {
#pragma HLS unroll
        ap_uint<32> u_32_d = u_64(31 + 32 * d, 32 * d);
        out_v.data[d] = to_float(u_32_d);
      }
      fifo_Y_out.write(out_v);
      ++c_idx;
      if (c_idx == 8) {
        c_idx = 0;
      }
    }
  }
}

void Arbiter_Y(const int P_N, const int M,
               hls::stream<float_v2>& in0,
               hls::stream<float_v2>& in1,
               hls::stream<float_v2>& in2,
               hls::stream<float_v2>& fifo_out) {
  const int rp_time = (P_N == 0) ? 1 : P_N;
  const int num_pe_output =
      ((M + NUM_CH_SPARSE_mult_2 - 1) / NUM_CH_SPARSE_mult_2) *
      NUM_CH_SPARSE_div_8;
  const int num_out = (M + 15) >> 4;
  const int num_ite_Y = num_pe_output * rp_time;
  for (int i = 0, c_idx = 0, o_idx = 0; i < num_ite_Y; ++i) {
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1
    float_v2 tmp{};
    if (c_idx == 0) {
      tmp = in0.read();
    } else if (c_idx == 1) {
      tmp = in1.read();
    } else {
      tmp = in2.read();
    }
    if (o_idx < num_out) {
      fifo_out.write(tmp);
    }
    ++c_idx;
    ++o_idx;
    if (c_idx == NUM_CH_SPARSE_div_8) {
      c_idx = 0;
    }
    if (o_idx == num_pe_output) {
      o_idx = 0;
    }
  }
}

void Merger_Y(const int P_N, const int M, hls::stream<float_v2> fifo_in[8],
              hls::stream<float_v16>& fifo_out) {
  const int rp_time = (P_N == 0) ? 1 : P_N;
  const int total_out = ((M + 15) >> 4) * rp_time;
  for (int o = 0; o < total_out; ++o) {
#pragma HLS pipeline II=1
    float_v16 tmpv16{};
    for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
      float_v2 tmp = fifo_in[i].read();
      tmpv16.data[i * 2 + 0] = tmp.data[0];
      tmpv16.data[i * 2 + 1] = tmp.data[1];
    }
    fifo_out.write(tmpv16);
  }
}

void FloatvMultConst(const int P_N, const int M, const int alpha_u,
                     hls::stream<float_v16>& fifo_in,
                     hls::stream<float_v16>& fifo_out) {
  const float alpha_f = to_float(static_cast<ap_uint<32>>(alpha_u));
  const int rp_time = (P_N == 0) ? 1 : P_N;
  const int num_ite_Y = ((M + 15) >> 4) * rp_time;
  for (int i = 0; i < num_ite_Y; ++i) {
#pragma HLS pipeline II=1
    float_v16 tmp = fifo_in.read();
    fifo_out.write(vec_mul(tmp, alpha_f));
  }
}

void read_Y(const int P_N, const int M, const ap_uint<512>* Y,
            hls::stream<float_v16>& fifo_Y) {
  const int rp_time = (P_N == 0) ? 1 : P_N;
  const int num_ite_Y = (M + 15) >> 4;

  for (int rp = 0; rp < rp_time; ++rp) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    for (int i = 0; i < num_ite_Y; ++i) {
#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
      float_v16 tmp = unpack_float_v16(Y[i]);
      fifo_Y.write(tmp);
    }
  }
}

void FloatvAddFloatv(const int P_N, const int M,
                     hls::stream<float_v16>& fifo_in0,
                     hls::stream<float_v16>& fifo_in1,
                     hls::stream<float_v16>& fifo_out) {
  const int rp_time = (P_N == 0) ? 1 : P_N;
  const int total = ((M + 15) >> 4) * rp_time;
  for (int i = 0; i < total; ++i) {
#pragma HLS pipeline II=1
    float_v16 a = fifo_in0.read();
    float_v16 b = fifo_in1.read();
    fifo_out.write(vec_add(a, b));
  }
}

void write_Y(const int P_N, const int M, hls::stream<float_v16>& fifo_Y,
             ap_uint<512>* Y_out) {
  const int rp_time = (P_N == 0) ? 1 : P_N;
  const int num_ite_Y = (M + 15) >> 4;

  for (int rp = 0; rp < rp_time; ++rp) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    for (int i = 0; i < num_ite_Y; ++i) {
#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
      float_v16 tmpv16 = fifo_Y.read();
      Y_out[i] = pack_float_v16(tmpv16);
    }
  }
}

void drain_int_stream(hls::stream<int>& fifo_in, int count) {
  for (int i = 0; i < count; ++i) {
#pragma HLS pipeline II=1
    (void)fifo_in.read();
  }
}

void drain_float_stream(hls::stream<float_v16>& fifo_in, int count) {
  for (int i = 0; i < count; ++i) {
#pragma HLS pipeline II=1
    (void)fifo_in.read();
  }
}

void Serpens(const int* edge_list_ptr, const ap_uint<512>* edge_list_ch0,
             const ap_uint<512>* edge_list_ch1, const ap_uint<512>* edge_list_ch2,
             const ap_uint<512>* edge_list_ch3, const ap_uint<512>* edge_list_ch4,
             const ap_uint<512>* edge_list_ch5, const ap_uint<512>* edge_list_ch6,
             const ap_uint<512>* edge_list_ch7, const ap_uint<512>* edge_list_ch8,
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
             const ap_uint<512>* edge_list_ch23, const ap_uint<512>* vec_X,
             const ap_uint<512>* vec_Y, ap_uint<512>* vec_Y_out,
             const int NUM_ITE, const int NUM_A_LEN, const int M, const int K,
             const int P_N, const int alpha_u, const int beta_u) {
#pragma HLS interface m_axi port = edge_list_ptr offset = slave bundle = gmem_ptr depth = 4096
#pragma HLS interface m_axi port = edge_list_ch0 offset = slave bundle = gmem0 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch1 offset = slave bundle = gmem1 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch2 offset = slave bundle = gmem2 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch3 offset = slave bundle = gmem3 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch4 offset = slave bundle = gmem4 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch5 offset = slave bundle = gmem5 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch6 offset = slave bundle = gmem6 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch7 offset = slave bundle = gmem7 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch8 offset = slave bundle = gmem8 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch9 offset = slave bundle = gmem9 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch10 offset = slave bundle = gmem10 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch11 offset = slave bundle = gmem11 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch12 offset = slave bundle = gmem12 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch13 offset = slave bundle = gmem13 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch14 offset = slave bundle = gmem14 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch15 offset = slave bundle = gmem15 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch16 offset = slave bundle = gmem16 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch17 offset = slave bundle = gmem17 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch18 offset = slave bundle = gmem18 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch19 offset = slave bundle = gmem19 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch20 offset = slave bundle = gmem20 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch21 offset = slave bundle = gmem21 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch22 offset = slave bundle = gmem22 depth = 65536
#pragma HLS interface m_axi port = edge_list_ch23 offset = slave bundle = gmem23 depth = 65536
#pragma HLS interface m_axi port = vec_X offset = slave bundle = gmemX depth = 65536
#pragma HLS interface m_axi port = vec_Y offset = slave bundle = gmemY depth = 65536
#pragma HLS interface m_axi port = vec_Y_out offset = slave bundle = gmemOut depth = 65536
#pragma HLS interface s_axilite port = edge_list_ptr bundle = control
#pragma HLS interface s_axilite port = edge_list_ch0 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch1 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch2 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch3 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch4 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch5 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch6 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch7 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch8 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch9 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch10 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch11 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch12 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch13 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch14 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch15 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch16 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch17 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch18 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch19 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch20 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch21 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch22 bundle = control
#pragma HLS interface s_axilite port = edge_list_ch23 bundle = control
#pragma HLS interface s_axilite port = vec_X bundle = control
#pragma HLS interface s_axilite port = vec_Y bundle = control
#pragma HLS interface s_axilite port = vec_Y_out bundle = control
#pragma HLS interface s_axilite port = NUM_ITE bundle = control
#pragma HLS interface s_axilite port = NUM_A_LEN bundle = control
#pragma HLS interface s_axilite port = M bundle = control
#pragma HLS interface s_axilite port = K bundle = control
#pragma HLS interface s_axilite port = P_N bundle = control
#pragma HLS interface s_axilite port = alpha_u bundle = control
#pragma HLS interface s_axilite port = beta_u bundle = control
#pragma HLS interface s_axilite port = return bundle = control
#pragma HLS dataflow

  const ap_uint<512>* edge_list_ch[NUM_CH_SPARSE] = {
      edge_list_ch0,  edge_list_ch1,  edge_list_ch2,  edge_list_ch3,
      edge_list_ch4,  edge_list_ch5,  edge_list_ch6,  edge_list_ch7,
      edge_list_ch8,  edge_list_ch9,  edge_list_ch10, edge_list_ch11,
      edge_list_ch12, edge_list_ch13, edge_list_ch14, edge_list_ch15,
      edge_list_ch16, edge_list_ch17, edge_list_ch18, edge_list_ch19,
      edge_list_ch20, edge_list_ch21, edge_list_ch22, edge_list_ch23};

  hls::stream<int> PE_inst[NUM_CH_SPARSE + 1];
#pragma HLS stream variable = PE_inst depth = FIFO_DEPTH
  hls::stream<float_v16> fifo_X_pe[NUM_CH_SPARSE + 1];
#pragma HLS stream variable = fifo_X_pe depth = FIFO_DEPTH
  hls::stream<ap_uint<512>> fifo_A[NUM_CH_SPARSE];
#pragma HLS stream variable = fifo_A depth = FIFO_DEPTH
  hls::stream<int> Yvec_inst[NUM_CH_SPARSE];
#pragma HLS stream variable = Yvec_inst depth = FIFO_DEPTH
  hls::stream<MultXVec> fifo_aXvec[NUM_CH_SPARSE];
#pragma HLS stream variable = fifo_aXvec depth = FIFO_DEPTH
  hls::stream<float_v2> fifo_Y_pe[NUM_CH_SPARSE];
#pragma HLS stream variable = fifo_Y_pe depth = FIFO_DEPTH
  hls::stream<float_v2> fifo_Y_pe_abd[8];
#pragma HLS stream variable = fifo_Y_pe_abd depth = FIFO_DEPTH
  hls::stream<float_v16> fifo_Y_AX;
#pragma HLS stream variable = fifo_Y_AX depth = FIFO_DEPTH
  hls::stream<float_v16> fifo_Y_alpha_AX;
#pragma HLS stream variable = fifo_Y_alpha_AX depth = FIFO_DEPTH
  hls::stream<float_v16> fifo_Y_in;
#pragma HLS stream variable = fifo_Y_in depth = FIFO_DEPTH
  hls::stream<float_v16> fifo_Y_in_beta;
#pragma HLS stream variable = fifo_Y_in_beta depth = FIFO_DEPTH
  hls::stream<float_v16> fifo_Y_out_stream;
#pragma HLS stream variable = fifo_Y_out_stream depth = FIFO_DEPTH

  const int rp_time = (P_N == 0) ? 1 : P_N;
  const int inst_drain_count = 4 + rp_time * (NUM_ITE + 1);
  const int x_drain_count = rp_time * ((K + 15) >> 4);

  read_edge_list_ptr(NUM_ITE, M, P_N, K, edge_list_ptr, PE_inst[0]);
  read_X(P_N, K, vec_X, fifo_X_pe[0]);

  for (int ch = 0; ch < NUM_CH_SPARSE; ++ch) {
#pragma HLS unroll
    read_A(P_N, NUM_A_LEN, edge_list_ch[ch], fifo_A[ch]);
  }

  for (int ch = 0; ch < NUM_CH_SPARSE; ++ch) {
#pragma HLS unroll
    PEG_Xvec(PE_inst[ch], fifo_A[ch], fifo_X_pe[ch], PE_inst[ch + 1],
             fifo_X_pe[ch + 1], Yvec_inst[ch], fifo_aXvec[ch]);
  }

  for (int ch = 0; ch < NUM_CH_SPARSE; ++ch) {
#pragma HLS unroll
    PEG_Yvec(Yvec_inst[ch], fifo_aXvec[ch], fifo_Y_pe[ch]);
  }

  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
    Arbiter_Y(P_N, M,
              fifo_Y_pe[i * NUM_CH_SPARSE_div_8 + 0],
              fifo_Y_pe[i * NUM_CH_SPARSE_div_8 + 1],
              fifo_Y_pe[i * NUM_CH_SPARSE_div_8 + 2],
              fifo_Y_pe_abd[i]);
  }

  Merger_Y(P_N, M, fifo_Y_pe_abd, fifo_Y_AX);

  FloatvMultConst(P_N, M, alpha_u, fifo_Y_AX, fifo_Y_alpha_AX);
  read_Y(P_N, M, vec_Y, fifo_Y_in);
  FloatvMultConst(P_N, M, beta_u, fifo_Y_in, fifo_Y_in_beta);
  FloatvAddFloatv(P_N, M, fifo_Y_alpha_AX, fifo_Y_in_beta,
                  fifo_Y_out_stream);
  write_Y(P_N, M, fifo_Y_out_stream, vec_Y_out);

  drain_int_stream(PE_inst[NUM_CH_SPARSE], inst_drain_count);
  drain_float_stream(fifo_X_pe[NUM_CH_SPARSE], x_drain_count);
}
