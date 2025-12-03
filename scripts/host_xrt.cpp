#include <ap_int.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "mmio.h"
#include "serpens.h"
#include "sparse_helper.h"

using std::cout;
using std::endl;
using std::max;
using std::min;
using std::string;
using std::vector;

namespace {

ap_uint<512> pack_float_block(const vector<float>& src, size_t base) {
  ap_uint<512> word = 0;
  for (int i = 0; i < 16; ++i) {
    size_t idx = base + i;
    float v = (idx < src.size()) ? src[idx] : 0.0f;
    word(32 * i + 31, 32 * i) = to_uint(v);
  }
  return word;
}

void unpack_float_block(const ap_uint<512>& word, vector<float>& dst,
                        size_t base) {
  for (int i = 0; i < 16; ++i) {
    size_t idx = base + i;
    if (idx < dst.size()) {
      dst[idx] = to_float(word(32 * i + 31, 32 * i));
    }
  }
}

ap_uint<512> pack_edge_block(const vector<uint64_t>& src, size_t base) {
  ap_uint<512> word = 0;
  for (int i = 0; i < 8; ++i) {
    size_t idx = base + i;
    uint64_t v = (idx < src.size()) ? src[idx] : 0UL;
    word(64 * i + 63, 64 * i) = static_cast<ap_uint<64>>(v);
  }
  return word;
}

void sync_bo_to_device(xrt::bo& bo, size_t bytes) {
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
}

void sync_bo_from_device(xrt::bo& bo, size_t bytes) {
  bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
}

}

int main(int argc, char** argv) {
  static_assert(NUM_CH_SPARSE == 24, "Host assumes 24 sparse channels.");

  if (argc < 3 || argc > 6) {
    cout << "Usage: " << argv[0]
         << " <Serpens.xclbin> <matrix A file> [rp_time] [alpha] [beta]\n";
    return EXIT_FAILURE;
  }

  char* xclbin_path = argv[1];
  char* filename_A = argv[2];

  float ALPHA = 0.85f;
  float BETA = -2.06f;
  int rp_time = 1;

  if (argc >= 4) {
    rp_time = atoi(argv[3]);
  }
  if (argc >= 5) {
    ALPHA = static_cast<float>(atof(argv[4]));
  }
  if (argc == 6) {
    BETA = static_cast<float>(atof(argv[5]));
  }

  cout << "rp_time = " << rp_time << "\n";
  cout << "alpha = " << ALPHA << "\n";
  cout << "beta = " << BETA << "\n";

  int M, K, nnz;
  vector<int> CSCColPtr;
  vector<int> CSCRowIndex;
  vector<float> CSCVal;
  vector<int> CSRRowPtr;
  vector<int> CSRColIndex;
  vector<float> CSRVal;

  cout << "Reading sparse A matrix ...";
  read_suitsparse_matrix(filename_A, CSCColPtr, CSCRowIndex, CSCVal, M, K, nnz,
                         CSC);
  CSC_2_CSR(M, K, nnz, CSCColPtr, CSCRowIndex, CSCVal, CSRRowPtr, CSRColIndex,
            CSRVal);
  cout << "done\n";

  cout << "Matrix size: \n";
  cout << "A: sparse matrix, " << M << " x " << K << ". NNZ = " << nnz << "\n";

  vector<float> vec_X_cpu(K, 0.0f);
  vector<float> vec_Y_cpu(M, 0.0f);

  cout << "Generating vector X ...";
  for (int kk = 0; kk < K; ++kk) {
    vec_X_cpu[kk] = 1.0f * (kk + 1);
  }

  cout << "Generating vector Y ...";
  for (int mm = 0; mm < M; ++mm) {
    vec_Y_cpu[mm] = -2.0f * (mm + 1);
  }
  cout << "done\n";

  cout << "Preparing sparse A for FPGA with " << NUM_CH_SPARSE
       << " HBM channels ...";
  vector<vector<edge>> edge_list_pes;
  vector<int> edge_list_ptr;

  generate_edge_list_for_all_PEs(
      CSCColPtr, CSCRowIndex, CSCVal, NUM_CH_SPARSE * 8, M, K, WINDOW_SIZE,
      edge_list_pes, edge_list_ptr, DEP_DIST_LOAD_STORE);

  vector<int> edge_list_ptr_fpga;
  int edge_list_ptr_fpga_size = ((edge_list_ptr.size() + 15) / 16) * 16;
  int edge_list_ptr_fpga_chunk_size =
      ((edge_list_ptr_fpga_size + 1023) / 1024) * 1024;
  edge_list_ptr_fpga.assign(edge_list_ptr_fpga_chunk_size, 0);
  for (size_t i = 0; i < edge_list_ptr.size(); ++i) {
    edge_list_ptr_fpga[i] = edge_list_ptr[i];
  }

  vector<vector<uint64_t>> sparse_A_fpga_vec(NUM_CH_SPARSE);
  edge_list_64bit(edge_list_pes, edge_list_ptr, sparse_A_fpga_vec,
                  NUM_CH_SPARSE);

  vector<vector<ap_uint<512>>> edge_list_words(NUM_CH_SPARSE);
  for (int ch = 0; ch < NUM_CH_SPARSE; ++ch) {
    const auto& src = sparse_A_fpga_vec[ch];
    size_t words512 = (src.size() + 7) / 8;
    edge_list_words[ch].resize(words512);
    for (size_t i = 0; i < words512; ++i) {
      edge_list_words[ch][i] = pack_edge_block(src, i * 8);
    }
  }
  cout << "done\n";

  cout << "Preparing vector X for FPGA ...";
  int vec_X_fpga_column_size = ((K + 16 - 1) / 16) * 16;
  int vec_X_fpga_chunk_size =
      ((vec_X_fpga_column_size + 1023) / 1024) * 1024;
  vector<float> vec_X_fpga(vec_X_fpga_chunk_size, 0.0f);
  for (int kk = 0; kk < K; ++kk) {
    vec_X_fpga[kk] = vec_X_cpu[kk];
  }

  vector<ap_uint<512>> vec_X_words(
      (vec_X_fpga_chunk_size + 15) / 16, ap_uint<512>(0));
  for (size_t i = 0; i < vec_X_words.size(); ++i) {
    vec_X_words[i] = pack_float_block(vec_X_fpga, i * 16);
  }

  cout << "Preparing vector Y for FPGA ...";
  int vec_Y_fpga_column_size = ((M + 16 - 1) / 16) * 16;
  int vec_Y_fpga_chunk_size =
      ((vec_Y_fpga_column_size + 1023) / 1024) * 1024;
  vector<float> vec_Y_fpga(vec_Y_fpga_chunk_size, 0.0f);
  vector<float> vec_Y_out_fpga(vec_Y_fpga_chunk_size, 0.0f);

  for (int mm = 0; mm < M; ++mm) {
    vec_Y_fpga[mm] = vec_Y_cpu[mm];
  }

  vector<ap_uint<512>> vec_Y_words(
      (vec_Y_fpga_chunk_size + 15) / 16, ap_uint<512>(0));
  vector<ap_uint<512>> vec_Y_out_words(vec_Y_words.size(), ap_uint<512>(0));
  for (size_t i = 0; i < vec_Y_words.size(); ++i) {
    vec_Y_words[i] = pack_float_block(vec_Y_fpga, i * 16);
  }
  cout << "done\n";

  cout << "Run spmv on cpu...";
  auto start_cpu = std::chrono::steady_clock::now();
  cpu_spmv_CSR(M, K, nnz, ALPHA, CSRRowPtr, CSRColIndex, CSRVal, vec_X_cpu,
               BETA, vec_Y_cpu);
  auto end_cpu = std::chrono::steady_clock::now();
  double time_cpu =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu)
          .count();
  time_cpu *= 1e-9;
  cout << "done (" << time_cpu * 1000 << " msec)\n";
  cout << "CPU GFLOPS: " << 2.0 * (nnz + M) / 1e9 / time_cpu << "\n";

  int MAX_SIZE_edge_LIST_PTR = static_cast<int>(edge_list_ptr.size()) - 1;
  int MAX_LEN_edge_PTR = edge_list_ptr[MAX_SIZE_edge_LIST_PTR];

  int* tmpPointer_v;
  tmpPointer_v = reinterpret_cast<int*>(&ALPHA);
  int alpha_int = *tmpPointer_v;
  tmpPointer_v = reinterpret_cast<int*>(&BETA);
  int beta_int = *tmpPointer_v;

  cout << "Loading xclbin: " << xclbin_path << "\n";
  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(xclbin_path);
  auto kernel = xrt::kernel(device, uuid, "Serpens");

  cout << "Allocating buffers ...";
  size_t edge_ptr_bytes = edge_list_ptr_fpga.size() * sizeof(int);
  size_t vecX_bytes = vec_X_words.size() * sizeof(ap_uint<512>);
  size_t vecY_bytes = vec_Y_words.size() * sizeof(ap_uint<512>);
  size_t vecYout_bytes = vec_Y_out_words.size() * sizeof(ap_uint<512>);
  size_t edge_ch_bytes = edge_list_words[0].size() * sizeof(ap_uint<512>);

  xrt::bo edge_ptr_bo(device, edge_ptr_bytes, kernel.group_id(0));
  vector<xrt::bo> edge_ch_bos(NUM_CH_SPARSE);
  for (int ch = 0; ch < NUM_CH_SPARSE; ++ch) {
    edge_ch_bos[ch] = xrt::bo(device, edge_ch_bytes, kernel.group_id(1 + ch));
  }
  xrt::bo vecX_bo(device, vecX_bytes, kernel.group_id(25));
  xrt::bo vecY_bo(device, vecY_bytes, kernel.group_id(26));
  xrt::bo vecYout_bo(device, vecYout_bytes, kernel.group_id(27));

  auto edge_ptr_host = edge_ptr_bo.map<int*>();
  std::copy(edge_list_ptr_fpga.begin(), edge_list_ptr_fpga.end(),
            edge_ptr_host);

  for (int ch = 0; ch < NUM_CH_SPARSE; ++ch) {
    auto ptr = edge_ch_bos[ch].map<ap_uint<512>*>();
    std::copy(edge_list_words[ch].begin(), edge_list_words[ch].end(), ptr);
  }

  auto vecX_host = vecX_bo.map<ap_uint<512>*>();
  std::copy(vec_X_words.begin(), vec_X_words.end(), vecX_host);

  auto vecY_host = vecY_bo.map<ap_uint<512>*>();
  std::copy(vec_Y_words.begin(), vec_Y_words.end(), vecY_host);

  auto vecYout_host = vecYout_bo.map<ap_uint<512>*>();
  std::fill(vecYout_host, vecYout_host + vec_Y_out_words.size(),
            ap_uint<512>(0));

  sync_bo_to_device(edge_ptr_bo, edge_ptr_bytes);
  for (int ch = 0; ch < NUM_CH_SPARSE; ++ch) {
    sync_bo_to_device(edge_ch_bos[ch], edge_ch_bytes);
  }
  sync_bo_to_device(vecX_bo, vecX_bytes);
  sync_bo_to_device(vecY_bo, vecY_bytes);

  cout << "launch kernel\n";
  auto start_kernel = std::chrono::steady_clock::now();
  auto run =
      kernel(edge_ptr_bo, edge_ch_bos[0], edge_ch_bos[1], edge_ch_bos[2],
             edge_ch_bos[3], edge_ch_bos[4], edge_ch_bos[5], edge_ch_bos[6],
             edge_ch_bos[7], edge_ch_bos[8], edge_ch_bos[9], edge_ch_bos[10],
             edge_ch_bos[11], edge_ch_bos[12], edge_ch_bos[13],
             edge_ch_bos[14], edge_ch_bos[15], edge_ch_bos[16],
             edge_ch_bos[17], edge_ch_bos[18], edge_ch_bos[19],
             edge_ch_bos[20], edge_ch_bos[21], edge_ch_bos[22],
             edge_ch_bos[23], vecX_bo, vecY_bo, vecYout_bo,
             MAX_SIZE_edge_LIST_PTR, MAX_LEN_edge_PTR, M, K, rp_time,
             alpha_int, beta_int);
  run.wait();
  auto end_kernel = std::chrono::steady_clock::now();

  sync_bo_from_device(vecYout_bo, vecYout_bytes);

  double time_taken =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_kernel -
                                                           start_kernel)
          .count();
  time_taken *= (1e-9 / rp_time);
  printf("Kernel time is %f ms\n", time_taken * 1000);

  float gflops = 2.0f * (nnz + M) / 1e9f / static_cast<float>(time_taken);
  printf("GFLOPS:%f \n", gflops);

  for (size_t i = 0; i < vec_Y_out_words.size(); ++i) {
    vec_Y_out_words[i] = vecYout_host[i];
    unpack_float_block(vec_Y_out_words[i], vec_Y_out_fpga, i * 16);
  }

  int mismatch_cnt = 0;
  for (int mm = 0; mm < M; ++mm) {
    float v_cpu = vec_Y_cpu[mm];
    float v_fpga = vec_Y_out_fpga[mm];
    float dff = fabs(v_cpu - v_fpga);
    float x = min(fabs(v_cpu), fabs(v_fpga)) + 1e-4f;
    mismatch_cnt += (dff / x > 1e-4f);
  }

  float diffpercent = 100.0f * mismatch_cnt / static_cast<float>(M);
  bool pass = diffpercent < 2.0f;

  if (pass) {
    cout << "Success!\n";
  } else {
    cout << "Failed.\n";
  }
  printf("num_mismatch = %d, percent = %.2f%%\n", mismatch_cnt, diffpercent);

  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
