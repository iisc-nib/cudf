/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/dictionary/update_keys.hpp>

#include <limits>

using namespace cudf::test::iterators;

template <typename V>
struct learn_cuda_test : public cudf::test::BaseFixture {};

using K = int32_t;
TYPED_TEST_SUITE(learn_cuda_test, cudf::test::Types<int32_t>);

void print_hello_cpp();

void test_single_agg_learn(cudf::column_view const& keys,
                     cudf::column_view const& values,
                     cudf::column_view const& expect_keys,
                     cudf::column_view const& expect_vals,
                     std::unique_ptr<cudf::groupby_aggregation>&& agg,
                     force_use_sort_impl use_sort                 = force_use_sort_impl::NO,
                     cudf::null_policy include_null_keys          = cudf::null_policy::EXCLUDE,
                     cudf::sorted keys_are_sorted                 = cudf::sorted::NO,
                     std::vector<cudf::order> const& column_order = {},
                     std::vector<cudf::null_order> const& null_precedence = {},
                     cudf::sorted reference_keys_are_sorted               = cudf::sorted::NO)

{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = values;

  requests[0].aggregations.push_back(std::move(agg));

  if (use_sort == force_use_sort_impl::YES) {
    // WAR to force cudf::groupby to use sort implementation
    requests[0].aggregations.push_back(
      cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
  }

  // since the default behavior of cudf::groupby(...) for an empty null_precedence vector is
  // null_order::AFTER whereas for cudf::sorted_order(...) it's null_order::BEFORE
  auto const precedence = null_precedence.empty()
                            ? std::vector<cudf::null_order>(1, cudf::null_order::BEFORE)
                            : null_precedence;

  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), include_null_keys, keys_are_sorted, column_order, precedence);

  auto result = gb_obj.aggregate(requests, cudf::test::get_default_stream());
}

TYPED_TEST(learn_cuda_test, basic)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::MAX>;

  const int num_distinct_keys = 1000;
  const int num_rows          = 10000000;
  const int num_values        = 10;
  std::vector<K> key_vec(num_rows);
  std::vector<V> val_vec(num_rows);
  std::vector<K> expected_keys_vec(num_distinct_keys);
  std::vector<R> expected_vals_vec(num_distinct_keys, 0);

  for (size_t i = 0; i < num_distinct_keys; i++) {
    expected_keys_vec[i] = i;
  }

  for (size_t i = 0; i < key_vec.size(); i++) {
    key_vec[i] = i % num_distinct_keys;
    val_vec[i] = i % num_values;
    expected_vals_vec[key_vec[i]] = expected_vals_vec[key_vec[i]] + val_vec[i];
  }

  cudf::test::fixed_width_column_wrapper<K> dummy({0}); // we'll measure the memory post this dummy init

  cudf::test::fixed_width_column_wrapper<K> keys(key_vec.begin(), key_vec.end());
  cudf::test::fixed_width_column_wrapper<V> vals(val_vec.begin(), val_vec.end());

  cudf::test::fixed_width_column_wrapper<K> expect_keys(expected_keys_vec.begin(),
                                                         expected_keys_vec.end());
  cudf::test::fixed_width_column_wrapper<R> expect_vals(expected_vals_vec.begin(),
                                                         expected_vals_vec.end());

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg_learn(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(learn_cuda_test, basic_join)
{
  // This test reproduced an implementation specific behavior where the combination of these
  // particular values ended up hashing to the empty key sentinel value used by the hash join
  // This test may no longer be relevant if the implementation ever changes.
  auto const left_first_col  = cudf::test::fixed_width_column_wrapper<int32_t>{1197};
  auto const left_second_col = cudf::test::strings_column_wrapper{"201812"};
  auto const left_third_col  = cudf::test::fixed_width_column_wrapper<int64_t>{2550000371};

  auto const right_first_col  = cudf::test::fixed_width_column_wrapper<int32_t>{1197};
  auto const right_second_col = cudf::test::strings_column_wrapper{"201812"};
  auto const right_third_col  = cudf::test::fixed_width_column_wrapper<int64_t>{2550000371};

  cudf::table_view left({left_first_col, left_second_col, left_third_col});
  cudf::table_view right({right_first_col, right_second_col, right_third_col});

  auto result = inner_join(left, right, {0, 1, 2}, {0, 1, 2});

  EXPECT_EQ(result->num_rows(), 1);
}

// TYPED_TEST(learn_cuda_test, basic)
// {
//   using V = TypeParam;
//   using R = cudf::detail::target_type_t<V, cudf::aggregation::MAX>;

//   cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
//   cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

//   cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
//   cudf::test::fixed_width_column_wrapper<R> expect_vals({6, 9, 8});

//   auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
//   test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

//   auto agg2 = cudf::make_max_aggregation<cudf::groupby_aggregation>();
//   test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
// }