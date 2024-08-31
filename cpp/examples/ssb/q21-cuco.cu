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
                      int lo_size)
{
  int tid = (threadIdx.x + blockIdx.x * blockDim.x) / TILE_SIZE;
  if (tid >= lo_size) return;
  auto this_thread = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

  if (!part_map.contains(this_thread, lo_partkey[tid]) ||
      !supp_map.contains(this_thread, lo_suppkey[tid]))
    return;

  res[tid] = 1;
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

  int mfgr12_count = 0;
  for (int i = 0; i < P_LEN; i++) {
    if (h_p_category[i] == 1) mfgr12_count++;
  }
  std::cout << "MFGR12/TOTAL: " << mfgr12_count << "/" << P_LEN
            << " part category selectivity: " << (float)mfgr12_count * 100 / (float)P_LEN << "%" << std::endl;
  int* h_res = (int*)malloc(sizeof(int) * LO_LEN);
  memset(h_res, 0, sizeof(int) * LO_LEN);

  int *d_p_partkey, *d_p_category, *d_res, *d_lo_partkey, *d_lo_orderdate, *d_lo_suppkey,
    *d_s_region, *d_s_suppkey, *d_d_datekey, *d_p_brand, *d_lo_revenue, *d_d_year;
  cudaMalloc(&d_p_partkey, P_LEN * sizeof(int));
  cudaMalloc(&d_p_category, P_LEN * sizeof(int));
  cudaMalloc(&d_p_brand, P_LEN * sizeof(int));
  cudaMalloc(&d_res, LO_LEN * sizeof(int));
  cudaMalloc(&d_lo_partkey, LO_LEN * sizeof(int));
  cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(int));
  cudaMalloc(&d_lo_suppkey, LO_LEN * sizeof(int));
  cudaMalloc(&d_lo_revenue, LO_LEN * sizeof(int));
  cudaMalloc(&d_s_suppkey, S_LEN * sizeof(int));
  cudaMalloc(&d_s_region, S_LEN * sizeof(int));
  cudaMalloc(&d_d_datekey, D_LEN * sizeof(int));
  cudaMalloc(&d_d_year, D_LEN * sizeof(int));

  cudaMemcpy(d_p_partkey, h_p_partkey, P_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_category, h_p_category, P_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p_brand, h_p_brand1, P_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_res, LO_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lo_partkey, h_lo_partkey, LO_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lo_orderdate, h_lo_orderdate, LO_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lo_suppkey, h_lo_suppkey, LO_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lo_revenue, h_lo_revenue, LO_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s_suppkey, h_s_suppkey, S_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s_region, h_s_region, S_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d_datekey, h_d_datekey, D_LEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d_year, h_d_year, D_LEN * sizeof(int), cudaMemcpyHostToDevice);

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

  auto part_map = cuco::static_multimap<
    int,
    int,
    cuda::thread_scope_device,
    cuco::cuda_allocator<char>,
    cuco::legacy::double_hashing<TILE_SIZE, cuco::default_hash_function<int>>>{
    part_capacity, cuco::empty_key{-1}, cuco::empty_value{-1}};
  auto supplier_map = cuco::static_multimap<
    int,
    int,
    cuda::thread_scope_device,
    cuco::cuda_allocator<char>,
    cuco::legacy::double_hashing<TILE_SIZE, cuco::default_hash_function<int>>>{
    supplier_capacity, cuco::empty_key{-1}, cuco::empty_value{-1}};
  auto date_map = cuco::static_multimap<
    int,
    int,
    cuda::thread_scope_device,
    cuco::cuda_allocator<char>,
    cuco::legacy::double_hashing<TILE_SIZE, cuco::default_hash_function<int>>>{
    date_capacity, cuco::empty_key{-1}, cuco::empty_value{-1}};

  // std::cout << "Launching kernel with grid: " << part_grid << " tb size: " << threadBlock <<
  // std::endl;
  build_hash_filtered<<<part_grid, TILE_SIZE * threadBlock>>>(
    part_map.get_device_mutable_view(),
    d_p_partkey,
    d_p_category,
    P_LEN,
    1 /*filter by part_category=mfgr#12*/);
  cudaDeviceSynchronize();
  build_hash_filtered<<<supplier_grid, TILE_SIZE * threadBlock>>>(
    supplier_map.get_device_mutable_view(),
    d_s_suppkey,
    d_s_region,
    P_LEN,
    1 /*s_region = AMERICA*/);
  cudaDeviceSynchronize();
  build_hash<<<date_grid, TILE_SIZE * threadBlock>>>(
    date_map.get_device_mutable_view(), d_d_datekey, D_LEN);
  cudaDeviceSynchronize();

  probe<<<std::ceil((float)LO_LEN / (float)threadBlock), TILE_SIZE * threadBlock>>>(
    part_map.get_device_view(),
    date_map.get_device_view(),
    supplier_map.get_device_view(),
    d_lo_partkey,
    d_lo_orderdate,
    d_lo_suppkey,
    d_lo_revenue,
    d_d_year,
    d_p_brand,
    d_res,
    LO_LEN);
  cudaDeviceSynchronize();

  std::cout << "Part map size: " << part_map.get_size()
            << "Part map capacity: " << part_map.get_capacity() << std::endl;

  cudaMemcpy(h_res, d_res, LO_LEN * sizeof(int), cudaMemcpyDeviceToHost);
  int probed = 0;
  for (int i = 0; i < LO_LEN; i++) {
    probed += h_res[i];
  }
  std::cout << "Total probed: " << probed << std::endl;


  /**
   * Below section is just a temporary workaround for probing.
   * TODO: need to understand how to retrieve probe from the device kernel
   * Also the below code demonstrates how to do a probe of host side if ever needed.
   */
  thrust::device_vector<int> partkey_lookup, datekey_lookup;
  int res = 0;
  for (int i=0; i<LO_LEN; i++) {
    if (h_res[i]) {
      res++;
      partkey_lookup.push_back(h_lo_partkey[i]);
      datekey_lookup.push_back(h_lo_orderdate[i]);
    }
  }
  std::cout << "Total after join: " << res << std::endl;
  thrust::device_vector<cuco::pair<int, int>> part_res(probed), year_res(probed);
  part_map.retrieve(partkey_lookup.begin(), partkey_lookup.end(), part_res.begin());
  date_map.retrieve(datekey_lookup.begin(), datekey_lookup.end(), year_res.begin());
  thrust::host_vector<cuco::pair<int, int>> h_part_res(part_res), h_year_res(year_res);
  // print brands
  std::set<pair<int, int>> groups;
  for (int i=0; i<probed; i++) {
    // std::cout << h_part_res[i].first << ": brand " << h_p_brand1[h_part_res[i].second] << std::endl;
    groups.insert(std::make_pair(h_d_year[h_year_res[i].second], h_p_brand1[h_part_res[i].second]));
  }
  std::cout << "Total groups: " << groups.size() << std::endl;
  
  return 0;
}
