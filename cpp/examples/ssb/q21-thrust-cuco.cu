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
            << " part category selectivity: " << (float)mfgr12_count * 100 / (float)P_LEN << "%"
            << std::endl;
  int* h_res = (int*)malloc(sizeof(int) * LO_LEN);
  memset(h_res, 0, sizeof(int) * LO_LEN);
  thrust::host_vector<cuco::pair<int, int>> h_s_suppkeydict(S_LEN), h_p_partkeydict(P_LEN), h_d_datekeydict(D_LEN);
  for (int i=0; i<P_LEN; i++) {
    h_p_partkeydict[i] = cuco::make_pair(h_p_partkey[i], i);
    // d_p_category[i] = h_p_category[i];
  }
  for (int i=0; i<S_LEN; i++) {
    h_s_suppkeydict[i] = cuco::make_pair(h_s_suppkey[i], i);
    // d_s_region[i] = h_s_region[i];
  }
  for (int i=0; i<D_LEN; i++) {
    h_d_datekeydict[i] = cuco::make_pair(h_d_datekey[i], i);
  }
  thrust::device_vector<cuco::pair<int, int>> d_s_suppkey = h_s_suppkeydict, 
                    d_p_partkey = h_p_partkeydict,
                    d_d_datekey = h_d_datekeydict;

  thrust::device_vector<int> d_s_region(h_s_region, h_s_region+S_LEN), d_p_category(h_p_category, h_p_category+P_LEN);
  cuco::static_multimap<int, int> supp_map{S_LEN*2, cuco::empty_key{-1}, cuco::empty_value{-1}};
  cuco::static_multimap<int, int> part_map{P_LEN*2, cuco::empty_key{-1}, cuco::empty_value{-1}};
  cuco::static_multimap<int, int> date_map{D_LEN*2, cuco::empty_key{-1}, cuco::empty_value{-1}};
  supp_map.insert_if(d_s_suppkey.begin(), d_s_suppkey.end(), d_s_region.begin(), 
    [] __device__(int region) { return region == 1; });
  part_map.insert_if(d_p_partkey.begin(), d_p_partkey.end(), d_p_category.begin(), 
    [] __device__(int category) { return category == 1; });
  date_map.insert(d_d_datekey.begin(), d_d_datekey.end());


  std::cout << "SUPP MAP SIZE: " << supp_map.get_size() << std::endl;
  std::cout << "PART MAP SIZE: " << part_map.get_size() << std::endl;
  std::cout << "DATE MAP SIZE: " << date_map.get_size() << std::endl;

  // probe stage
  // This is not a good way to do it, we need to pipeline the library calls, otherwise there is unnecessary overheads
  thrust::device_vector<int> d_lo_suppkey_probe(h_lo_suppkey, h_lo_suppkey+LO_LEN),
                             d_lo_partkey_probe(h_lo_partkey, h_lo_partkey+LO_LEN),
                             d_lo_orderdate_probe(h_lo_orderdate, h_lo_orderdate+LO_LEN);
  thrust::device_vector<bool>  d_lo_suppkey_res(LO_LEN, false), 
        d_lo_partkey_res(LO_LEN, false), d_lo_orderdate_res(LO_LEN, false);

  supp_map.contains(d_lo_suppkey_probe.begin(), d_lo_suppkey_probe.end(), d_lo_suppkey_res.begin());
  part_map.contains(d_lo_partkey_probe.begin(), d_lo_partkey_probe.end(), d_lo_partkey_res.begin());
  date_map.contains(d_lo_orderdate_probe.begin(), d_lo_orderdate_probe.end(), d_lo_orderdate_res.begin());

  thrust::host_vector<bool> h_lo_suppkey_res = d_lo_suppkey_res, 
        h_lo_partkey_res= d_lo_partkey_res, h_lo_orderdate_res = d_lo_orderdate_res;
  
  std::set<pair<int, int>> groups;
  std::map<pair<int, int>, long long> final_res;
  int res = 0;

  for (int i=0; i<LO_LEN; i++) {
    if (h_lo_suppkey_res[i] && h_lo_partkey_res[i]) {
        auto p = make_pair(h_lo_orderdate[i]/10000, h_p_brand1[h_lo_partkey[i]-1]);
        final_res[p] += h_lo_revenue[i];
        res++;
    }
  }
  std::cout << "Res count: " << res << std::endl;
  std::cout << "YEAR\tBRAND\tREVENUE\n";
  for (auto e: final_res) {
    std::cout << e.first.first << "\t" << e.first.second << "\t" << e.second << std::endl;
  }

//   std::cout << "Total groups: " << final_res.size() << std::endl;

  return 0;
}
