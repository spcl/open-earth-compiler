//===- RandGenerator.h ------------------------------------------*- C++ -*-===//
//
// Copyright 2019 Jakub Lichman
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
// Base class for classes that need random number generation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_STENCIL_RANDGENERATOR_H
#define MLIR_TARGET_STENCIL_RANDGENERATOR_H

#include <set>
#include <random>

using namespace std;

namespace mlir {
namespace experimental {

class RandGenerator {
private:
  mt19937 eng;

public:
  RandGenerator() {
    random_device rd;
    eng = mt19937(rd()); // seed the generator
  }

protected:
  int rand_range(int lowerBound, int upperBound) {
    assert(lowerBound <= upperBound);
    if (lowerBound == upperBound)
      return lowerBound;
    
    uniform_int_distribution<> distr(lowerBound, upperBound); // define the range
    return distr(eng);
  }

  template<typename T>
  T* rand_advance(set<T*> &data) {
    if(data.empty())
      return nullptr;

    int rand = rand_range(0, data.size() - 1);
    auto it = data.begin();
    std::advance(it, rand);
    T* result = *it;
    data.erase(it);
    return result;
  }

  pair<Value *, Value *> get2RandValues(set<Operation *> &values) {
    // TODO: analyze if it is correct to remove them from context
    auto rand1 = rand_advance(values);
    auto rand2 = rand_advance(values);
    if(rand2 == nullptr)
      rand2 = rand1;
    
    return make_pair(rand1->getResult(0), rand2->getResult(0));
  }
};

} // namespace experimental
} // namespace mlir

#endif // MLIR_TARGET_STENCIL_RANDGENERATOR_H
