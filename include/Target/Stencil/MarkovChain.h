//===- MarkovChain.h -------------------------------------------*- C++ -*-===//
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
// Markov chain based random program generator.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_STENCIL_MARKOVCHAIN_H
#define MLIR_TARGET_STENCIL_MARKOVCHAIN_H

#include "mlir/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "RandGenerator.h"

#include <vector>
#include <iomanip>
#include <map>

using namespace std;
using Filter = function<bool(string)>;

namespace mlir {
namespace experimental {

string name(Operation * op) {
  return op->getName().getStringRef().str();
}

string to_string_with_precision(const double a_value, const int n)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

class MarkovChain : public RandGenerator{
public:
  MarkovChain(ModuleOp &m) : module(m) {
    createMarkovChain(m);
  }

  void print(llvm::raw_ostream &output) {
    output << "\n\ndigraph G {\n";
    output << "\trankdir=HR;\n\tsize=\"8,12\";\n";
    output << "\tnode [shape = doublecircle]; module module_terminator;\n";
    output << "\tnode [shape = circle];\n";
    for (auto it : markovChain) {
      double sum = 0;
      for (auto it2 : it.second)
        sum += it2.second;
      for (auto it2 : it.second) {
        output << "\t\"" << it.first << "\" -> \"" << it2.first;
        output << "\" [ label = \"" << to_string_with_precision(it2.second/sum, 3) << "\" ];\n";
      }
      output << "\n";
    }
    output << "}\n";
  }

  vector<string> generateRandomSequence(int max_length = 50) {
    vector<string> sequence;
    string curr_op = "module"; 
    int length = 0;
    while(markovChain.find(curr_op) != markovChain.end() 
    && length++ < max_length) {
      sequence.push_back(curr_op);
      auto curr_state = markovChain[curr_op];
      int sum = 0;
      for (auto it : curr_state)
        sum += it.second;
  
      int i = 0, rand = rand_range(0, sum);
      for (auto it : curr_state) {
        i += it.second;
        if (i >= rand) {
          curr_op = it.first;
          break;
        }
      }
      assert(i >= rand);
    }

    if(length >= max_length) {
      sequence.clear();
      sequence.push_back("module");
    }

    sequence.push_back("module_terminator");
    correctIfOps(sequence);
    return sequence;
  }

  string getNextOp(string curr_op, Filter f = nullptr) {
    auto curr_state = markovChain[curr_op];
    int sum = 0;
    for (auto it : curr_state)
      if (f && f(it.first))
        sum += it.second;

    int i = 0, rand = rand_range(0, sum);
    string next;
    for (auto it : curr_state) {
      if (f && f(it.first)) {
        i += it.second;
        if (i >= rand) {
          next = it.first;
          break;
        }
      }
    }
    assert(i >= rand);
    return next;
  }

  vector<string> generateIfOpsTestingSequence(int max_length = 50) {
    vector<string> sequence = {
      "module",
      "func"
    };

    for (int i = 0; i < rand_range(15, max_length); i++) {
      if(rand_range(0, 1) == 0) {
        sequence.push_back("std.constant");
        sequence.push_back("std.constant");
        sequence.push_back("std.cmpf");
        sequence.push_back("scf.if");
      } else
        sequence.push_back("scf.terminator");
    }

    sequence.push_back("func_end");
    sequence.push_back("module_terminator");

    correctIfOps(sequence);
    return sequence;
  }

protected:
  ModuleOp &module;
  map<string, map<string, int>> markovChain;

  static void createChain(vector<Operation *> &chain, Operation * op) {
    chain.push_back(op);
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &opp : block) {
          createChain(chain, &opp);
        }
      }
    }
  }

  void createMarkovChain(ModuleOp &m) {
    vector<Operation *> chain;
    createChain(chain, m.getOperation());

    for (int i = 1; i < (int)chain.size(); i++) {
      markovChain[name(chain[i-1])][name(chain[i])]++;
    }

    // --- STATIC SETTINGS ---
    // ======== uncomment in order to increase chances of generating if ops ========
    // markovChain["std.constant"]["std.cmpf"] *= 10;
    // markovChain["stencil.get_value"]["std.cmpf"] *= 10;
    // markovChain["std.cmpf"]["loop.if"] *= 20;
    // markovChain["stencil.write"]["loop.terminator"] *= 10;
    // markovChain["stencil.write"]["loop.terminator"] *= 20;
    // ========================================================================

    // markovChain["stencil.write"]["stencil._do_method_end"] /= 2;
    // markovChain["stencil.stage"]["stencil.var"] *= 1.5;
    // markovChain["stencil.var"]["stencil.var"] /= 1.5;

    // ======== UNCOMMENT WHEN YOU WANT TO TRAIN ON cosmo_merged.mlir ========
    // markovChain["stencil.var"].erase("stencil.stencil");
    // markovChain["stencil.stencil"].erase("stencil.stencil");

    // we want only one DoMethod
    // markovChain["stencil._do_method_end"] = {{"stencil._stage_end", 1}};
    // we want only one Stage
    // markovChain["stencil._stage_end"] = {{"stencil._multi_stage_end", 1}};
    // we want only one MultiStage
    // markovChain["stencil._multi_stage_end"] = {{"stencil._stencil_end", 1}};
    // we want only one Stencil
    // markovChain["stencil._stencil_end"] = {{"stencil._iir_end", 1}};
    // we want only one IIR
    // ========================================================================
    markovChain["stencil._iir_end"] = {{"module_terminator", 1}};
  }

  void correctIfOps(vector<string> &chain) {
    // every loop.terminator has to be preceeded by loop.if
    // every loop.if must be followed by 1 or 2 loop.terminator 's

    int if_ops_seen = 0;
    int terms_seen = 0;
    int terms_to_see = 0;

    vector<int> to_erase;

    for (auto op : chain)
      if (op == "loop.terminator")
        terms_to_see++;

    for (unsigned int i = 0; i < chain.size(); i++) {
      auto op = chain[i];
      if (op == "scf.if") {
        if_ops_seen++;
        if (if_ops_seen > terms_to_see) {
          to_erase.push_back(i);
          if_ops_seen--;
        }
      } else if (op == "loop.terminator") {
        terms_seen++;
        terms_to_see--;
        if(terms_seen > 2*if_ops_seen) {
          to_erase.push_back(i);
          terms_seen--;
        }
      }
    }

    std::reverse(to_erase.begin(), to_erase.end());
    for (auto index : to_erase) {
      chain.erase(chain.begin() + index);
    }
  }
};

} // namespace experimental
} // namespace mlir

#endif // MLIR_TARGET_STENCIL_MARKOVCHAIN_H
