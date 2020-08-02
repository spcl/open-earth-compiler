//===- BucketDistribution.h -----------------------------------------------===//
//
// Copyright 2020 Jakub Lichman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Container for any kind of discrete distribution. For now we implemented only
// uniform and exponential one.
//
//===----------------------------------------------------------------------===//

#include "RandGenerator.h"

#include <vector>
#include <utility>

using namespace std;

namespace mlir {
namespace experimental {
  using ulong = unsigned long long int;

  /// Container for any kind of discrete distribution and element type.
  /// DistType has to implement these 2 methods:
  ///   * static void insert(vector<pair<T *, int>> &distribution, T* elem)
  ///   * static void erase(vector<pair<T *, int>> &distribution, T* elem)
  template<typename DistType, typename ElemType>
  class Distribution : RandGenerator {
    using Filter = function<bool(ElemType *)>;

    private:
      vector<pair<ElemType *, int>> distribution;
      map<string, Filter> filters;

      /// Returns true if `el` passes all filters, false otherwise.
      bool passes(ElemType * el) {
        for (auto it : filters)
          if (!it.second(el))
            return false;
        return true;
      }

    public:
      /// Adds element to the container.
      void insert(ElemType* element) {
        DistType::insert(distribution, element);
      }

      /// Size of all items in the distribution no matter
      /// if they pass the filters.
      unsigned int size() {
        unsigned int size = 0;
        for (auto it : distribution)
          if (passes(it.first))
            size++;
        return size;
      }

      bool empty() {
        return distribution.empty();
      }

      bool contains(ElemType* element) {
        for (auto elem : elems())
          if (elem == element)
            return true;
        return false;
      }

      void erase(ElemType* element) {
        DistType::erase(distribution, element);
      }

      /// Sets view i.e. filters out all elements that does not pass `filter`.
      /// Filter id has to be unique otherwise `filter` at `id` is reset.
      void setView(Filter filter, string id) {
        if (filter && id != "")
          this->filters[id] = filter;
      }

      /// Removes the filter identified by the `id`.
      void unsetView(string id) {
        this->filters.erase(id);
      }

      /// Samples from the distribution `n` times.
      vector<ElemType *> sample(unsigned int n) {
        assert(n <= size());
        vector<ElemType *> sampled;
        for (unsigned int i = 0; i < n; i++) {
          ElemType * elem = _sample(rand_range(1, getDistributionSize()));
          auto id = "##__" + to_string(i);
          setView([&](ElemType * el) { return el != elem; }, id);
          sampled.push_back(elem);
        }

        for (unsigned int i = 0; i < n; i++)
          unsetView("##__" + to_string(i));

        return sampled;
      }

      ElemType * sample() {
        int rand = rand_range(1, getDistributionSize());
        return _sample(rand);
      }

      /// Current distribution size, i.e. size of all items
      /// that pass all filters.
      ulong getDistributionSize() {
        ulong distribution_size = 0;
        for (auto it : distribution)
          if (passes(it.first))
            distribution_size += it.second;
        return distribution_size;
      }

      /// Returns all elements passing all filters.
      const vector<ElemType *> elems() {
        vector<ElemType *> arr;
        for (auto it : distribution) {
          if (passes(it.first))
            arr.push_back(it.first);
        }
        return arr;
      }

    private:
      ElemType* _sample(int rand) {
        int sum = 0;
        for (auto it : distribution) {
          if (passes(it.first)) {
            sum += it.second;
            if (sum >= rand)
              return it.first;
          }
        }
        llvm_unreachable("Rand out of bounds");
      }
  };


  /// Uniform distribution.
  class Uniform {
  public:
    template<typename T>
    static void insert(vector<pair<T *, int>> &distribution, T* elem) {
      distribution.push_back({elem, 1});
    }
    
    template<typename T>
    static void erase(vector<pair<T *, int>> &distribution, T* elem) {
      for (auto it = distribution.begin(); it != distribution.end(); ++it) {
        if (it->first == elem) {
          distribution.erase(it);
          return;
        }
      }
    }
  };

  /// Exponential distribution.
  class Exponential {
  public:
    template<typename T>
    static void insert(vector<pair<T *, int>> &distribution, T* elem) {
      if (distribution.size() < 2) {
        distribution.push_back({elem, 1});
      } else {
        if (distribution.size() > 15) {
          distribution.erase(distribution.begin(), distribution.begin()+7);
          distribution[0].second = 1;
          distribution[1].second = 1;
          for (int i = 2; i < (int)distribution.size(); i++)
            distribution[i].second = distribution[i-1].second * 2;
        }
        distribution.push_back({elem, distribution.back().second * 2});
      }
    }
    
    template<typename T>
    static void erase(vector<pair<T *, int>> &distribution, T* elem) {
      if (distribution.empty())
        return;

      auto to_erase = distribution.end();
      for (auto it = distribution.begin(); it != distribution.end(); ++it) {
        if (it->first == elem)
          to_erase = it;
        if (to_erase != distribution.end())
          it->second = max(1, it->second/2);
      }

      if (to_erase == distribution.end())
        return;
      
      distribution.erase(to_erase);
    }
  };
}
}
