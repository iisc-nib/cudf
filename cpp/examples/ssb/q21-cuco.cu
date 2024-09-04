#include "ssb_utils.h"

#include <cooperative_groups.h>
#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <curand.h>
#include <stdio.h>

#include <iostream>
#include <map>
#include <set>
#include <vector>

/**
 * SSB Q21
 * select sum(lo_revenue), d_year, p_brand1
 * from lineorder, date, part, supplier
 * where lo_orderdate = d_datekey
 * and lo_partkey = p_partkey
 * and lo_suppkey = s_suppkey
 * and p_category = 'MFGR#12'
 * and s_region = 'AMERICA'
 * group by d_year, p_brand1
 * order by d_year, p_brand1;
 */

using namespace cooperative_groups;
// Alternatively use an alias to avoid polluting the namespace with collective algorithms
namespace cg = cooperative_groups;
#pragma GCC diagnostic ignored "-Wattributes"

#define TILE_SIZE 1

template <typename Map>
__global__ void probe(Map part_map,
                      Map date_map,
                      Map supp_map,
                      int* lo_partkey,
                      int* lo_orderdate,
                      int* lo_suppkey,
                      int* lo_revenue,
                      int* d_year,
                      int* p_brand,
                      int* res,
                      int lo_size, bool* suppkey_bloom, bool* partkey_bloom)
{
  int tid = ((threadIdx.x + blockIdx.x * blockDim.x) / TILE_SIZE);
  if (tid >= lo_size) return;
  // auto this_thread = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  if (suppkey_bloom[(lo_suppkey[tid]-1)%S_LEN] == false || partkey_bloom[(lo_partkey[tid]-1)%P_LEN] == false) return;
  auto supp_idx = supp_map.find(lo_suppkey[tid]);
  if (supp_idx == supp_map.end()) return;
  auto part_idx = part_map.find(lo_partkey[tid]);
  if (part_idx == part_map.end()) return;
  auto date_idx = date_map.find(lo_orderdate[tid]);
  // if (part_idx != part_map.end()) {
    int hash = (p_brand[part_idx->second] * 7 +  (d_year[date_idx->second] - 1992)) % ((1998-1992+1) * (5*5*40));
    res[hash * 4] = d_year[date_idx->second];
    res[hash * 4 + 1] = p_brand[part_idx->second];
    atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(lo_revenue[tid]));
    // res[tid] = 1;
  // }
}

template <typename Map>
__global__ void build_hash_filtered(Map map_ref, int* column, int* filter, int n, int predicate)
{
  int tid          = (threadIdx.x + blockIdx.x * blockDim.x) / TILE_SIZE;
  auto this_thread = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  // filtering based on category or region
  if (tid < n && filter[tid] != predicate) return;
  if (tid < n) { map_ref.insert(this_thread, cuco::pair{column[tid], tid}); }
}

template <typename Map>
__global__ void build_hash(Map map_ref, int* column, int n)
{
  int tid          = (threadIdx.x + blockIdx.x * blockDim.x) / TILE_SIZE;
  auto this_thread = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  // filtering based on category or region
  if (tid < n) { map_ref.insert(this_thread, cuco::pair{column[tid], tid}); }
}

int main(int argc, char** argv)
{
  int* h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int* h_lo_partkey   = loadColumn<int>("lo_partkey", LO_LEN);
  int* h_lo_suppkey   = loadColumn<int>("lo_suppkey", LO_LEN);
  int* h_lo_revenue   = loadColumn<int>("lo_revenue", LO_LEN);

  int* h_p_partkey  = loadColumn<int>("p_partkey", P_LEN);
  int* h_p_brand1   = loadColumn<int>("p_brand1", P_LEN);
  int* h_p_category = loadColumn<int>("p_category", P_LEN);

  int* h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int* h_d_year    = loadColumn<int>("d_year", D_LEN);

  int* h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int* h_s_region  = loadColumn<int>("s_region", S_LEN);
  bool* bloom_suppkey = (bool*)malloc((S_LEN)); // assign S_LEN bits 
  bool* bloom_partkey = (bool*)malloc((P_LEN)); // assing P_LEN bits to this bloomfilter
  memset(bloom_suppkey, 0, (S_LEN)*sizeof(bool));
  memset(bloom_partkey, 0, (P_LEN)*sizeof(bool));
  // for (int i=0; i<S_LEN; i++) {
  //   if (h_s_region == 1) {
  //     int bloom_idx = hash_fn(h_s_suppkey[i])%S_LEN;
  //     int i1 = bloom_idx/sizeof(int);
  //     int val = 0x1 << ((sizeof(int)*8) - 1 - bloom_idx%(sizeof(int)*8));
  //   }
  // }
  int ass=0, aps = 0;
  for (int i=0; i<S_LEN; i++) {
    if (h_s_region[i] == 1) {
      int bloom_idx = (h_s_suppkey[i]-1)%S_LEN;
      bloom_suppkey[bloom_idx] = true;
      ass++;
    }
  }
  for (int i=0; i<P_LEN; i++) {
    if (h_p_category[i] == 1) {
      int bloom_idx = (h_p_partkey[i]-1)%P_LEN;
      bloom_partkey[bloom_idx] = true;
      aps++;
    }
  }

  // int ss=0, ps=0;
  // for (int i=0; i<S_LEN; i++) ss+=bloom_suppkey[i];
  // for (int i=0; i<P_LEN; i++) ps+=bloom_partkey[i];
  // std::cout << ass << " " << ss << std::endl;
  // std::cout << aps << " " << ps << std::endl;


  // std::map<int, int> hmap1;
  // for (int i=0; i<S_LEN; i++) {
  //   if (h_s_region[i] == 1) hmap1[h_s_suppkey[i]] = i;
  // }
  // int res = 0;
  // for (int i=0; i<LO_LEN; i++) {
  //   if (hmap1.find(h_lo_suppkey[i]) != hmap1.end()) res++;
  // }
  // std::cout << "Selectivity of suppkey in lineorder: " << res << std::endl;

  int mfgr12_count = 0;
  for (int i = 0; i < P_LEN; i++) {
    if (h_p_category[i] == 1) mfgr12_count++;
  }
  std::cout << "MFGR12/TOTAL: " << mfgr12_count << "/" << P_LEN
            << " part category selectivity: " << (float)mfgr12_count * 100 / (float)P_LEN << "%" << std::endl;
  int res_size = ((1998-1992+1) * (5 * 5 * 40));
  int res_array_size = res_size * 4;
  int* h_res = (int*)malloc(sizeof(int) * res_array_size);
  memset(h_res, 0, sizeof(int) * res_array_size);

  int *d_p_partkey, *d_p_category, *d_res, *d_lo_partkey, *d_lo_orderdate, *d_lo_suppkey,
    *d_s_region, *d_s_suppkey, *d_d_datekey, *d_p_brand, *d_lo_revenue, *d_d_year;
  bool *d_suppkey_bloom, *d_partkey_bloom;
  cudaMalloc(&d_p_partkey, P_LEN * sizeof(int));
  cudaMalloc(&d_p_category, P_LEN * sizeof(int));
  cudaMalloc(&d_p_brand, P_LEN * sizeof(int));
  cudaMalloc(&d_res, res_array_size * sizeof(int));
  cudaMalloc(&d_lo_partkey, LO_LEN * sizeof(int));
  cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(int));
  cudaMalloc(&d_lo_suppkey, LO_LEN * sizeof(int));
  cudaMalloc(&d_lo_revenue, LO_LEN * sizeof(int));
  cudaMalloc(&d_s_suppkey, S_LEN * sizeof(int));
  cudaMalloc(&d_s_region, S_LEN * sizeof(int));
  cudaMalloc(&d_d_datekey, D_LEN * sizeof(int));
  cudaMalloc(&d_d_year, D_LEN * sizeof(int));
  cudaMalloc(&d_suppkey_bloom, S_LEN * sizeof(bool));
  cudaMalloc(&d_partkey_bloom, P_LEN * sizeof(bool));

  cudaMemcpy(d_p_partkey, h_p_partkey, P_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_category, h_p_category, P_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_brand, h_p_brand1, P_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_res, res_array_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lo_partkey, h_lo_partkey, LO_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lo_orderdate, h_lo_orderdate, LO_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lo_suppkey, h_lo_suppkey, LO_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lo_revenue, h_lo_revenue, LO_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s_suppkey, h_s_suppkey, S_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s_region, h_s_region, S_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d_datekey, h_d_datekey, D_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d_year, h_d_year, D_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_suppkey_bloom, bloom_suppkey, S_LEN * sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_partkey_bloom, bloom_partkey, P_LEN * sizeof(bool), cudaMemcpyHostToDevice);

  // first stage is to make a predicated join using cuco static multimap
  // build hash table for part, supplier and date, and one kernel for probe phase
  auto constexpr load_factor          = 0.5;
  std::size_t const part_capacity     = std::ceil(P_LEN / load_factor);
  std::size_t const supplier_capacity = std::ceil(S_LEN / load_factor);
  std::size_t const date_capacity     = std::ceil(D_LEN / load_factor);
  
  int threadBlock   = 1024 / TILE_SIZE;
  int part_grid     = std::ceil((float)P_LEN / ((float)threadBlock));
  int supplier_grid = std::ceil((float)S_LEN / ((float)threadBlock));
  int date_grid     = std::ceil((float)D_LEN / ((float)threadBlock));
  //   build_hash<<<grid, threadBlock>>>(insert_ref, h_s_suppkey, P_LEN);

  auto part_map = cuco::static_map{part_capacity, cuco::empty_key{-1}, cuco::empty_value{-1}, thrust::equal_to<int>{},
                              cuco::double_hashing<TILE_SIZE, cuco::default_hash_function<int>>{}};
  auto supplier_map = cuco::static_map{supplier_capacity, cuco::empty_key{-1}, cuco::empty_value{-1}, thrust::equal_to<int>{},
                              cuco::double_hashing<TILE_SIZE, cuco::default_hash_function<int>>{}};
  auto date_map = cuco::static_map{date_capacity, cuco::empty_key{-1}, cuco::empty_value{-1}, thrust::equal_to<int>{},
                              cuco::double_hashing<TILE_SIZE, cuco::default_hash_function<int>>{}};

  // std::cout << "Launching kernel with grid: " << part_grid << " tb size: " << threadBlock <<
  // std::endl;
  build_hash_filtered<<<part_grid, TILE_SIZE * threadBlock>>>(
    part_map.ref(cuco::insert),
    d_p_partkey,
    d_p_category,
    P_LEN,
    1 /*filter by part_category=mfgr#12*/);
  cudaDeviceSynchronize();
  build_hash_filtered<<<supplier_grid, TILE_SIZE * threadBlock>>>(
    supplier_map.ref(cuco::insert),
    d_s_suppkey,
    d_s_region,
    P_LEN,
    1 /*s_region = AMERICA*/);
  cudaDeviceSynchronize();
  build_hash<<<date_grid, TILE_SIZE * threadBlock>>>(
    date_map.ref(cuco::insert), d_d_datekey, D_LEN);
  // cudaDeviceSynchronize();

  probe<<<std::ceil((float)LO_LEN / (float)threadBlock), (TILE_SIZE * threadBlock)>>>(
    part_map.ref(cuco::find),
    date_map.ref(cuco::find),
    supplier_map.ref(cuco::find),
    d_lo_partkey,
    d_lo_orderdate,
    d_lo_suppkey,
    d_lo_revenue,
    d_d_year,
    d_p_brand,
    d_res,
    LO_LEN,
    d_suppkey_bloom, d_partkey_bloom);
  cudaDeviceSynchronize();

  // std::cout << "Part map size: " << part_map.get_size()
  //           << "Part map capacity: " << part_map.get_capacity() << std::endl;

  cudaMemcpy(h_res, d_res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < res_size; i++) {
    if (h_res[4*i] != 0) {
      cout << h_res[4*i] << " " << h_res[4*i + 1] << " " << reinterpret_cast<unsigned long long*>(&h_res[4*i + 2])[0]  << endl;
    }
  }
  
  return 0;
}
