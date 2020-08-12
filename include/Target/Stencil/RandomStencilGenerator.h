//===- RandomStencilGenerator.h ---------------------------------*- C++ -*-===//
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
// ============================================================================
//
// This file implements random stencil program generation engine.
//
//===---------------------------------------------------------------------===//

#ifndef MLIR_TARGET_STENCIL_RANDOMSTENCILGENERATOR_H
#define MLIR_TARGET_STENCIL_RANDOMSTENCILGENERATOR_H

#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "llvm/Support/raw_ostream.h"

#include "MarkovChain.h"
#include "RandGenerator.h"
#include "BucketDistribution.h"
//#include "mlir/Target/Stencil/DCE.h"

#include <map>
#include <set>
#include <stack>
#include <string>
#include <sstream>
#include <algorithm>

const int DEBUG = 1;

using namespace std;
using namespace mlir::stencil;

namespace mlir {
namespace experimental {

struct Context {
  public:
    int index = 0;

    Distribution<Uniform, Value> fields;
    Distribution<Exponential, Operation> values;
    Distribution<Exponential, Operation> bool_values;

    // set of IfOp indexes that have else region
    set<int> if_else;

    // set of thenRegion terminators of IfOps that do have elseRegions
    set<int> then_terms;

    void print(llvm::raw_ostream &output) {
      output << "fields:" << fields.size() 
             << " values:" << values.size() 
             << " bool values:" << bool_values.size()
             << " if_else: ";
      for (auto i : if_else)
        output << i << " ";
      output << "then_terms: ";
      for (auto i : then_terms)
        output << i << " ";
      output << "\n";
    }
};

class Accumulator {
protected:
  OpBuilder* builder;
  Location loc;
public:
  Accumulator(OpBuilder* b, Location l)
    : builder(b), loc(l) {}
};

class FuncAccumulator : public Accumulator, RandGenerator {
private:
  int min_no_args, max_no_args, fnum;
public:
  FuncAccumulator(set<Operation *> ops, OpBuilder* b, Location l)
    : Accumulator(b, l), min_no_args(1), max_no_args(1), fnum(0) {
    for (auto op : ops) {
      int no_args = op->getRegion(0).front().getNumArguments();
      min_no_args = max(1, min(min_no_args, no_args));
      max_no_args = max(max_no_args, no_args);
    }
  }

  FuncOp get(Context* c) {
    int rand = rand_range(min_no_args, max_no_args);
    auto fieldType =
      FieldType::get(builder->getContext(), builder->getF64Type(), {-1, -1, -1});
    auto typeList = SmallVector<Type, 8>(rand, fieldType);
    auto funcType = FunctionType::get(typeList, {}, builder->getContext());

    // Create func with `rand` number of 3D dynamic shape fields
    // and add body block.
    auto func = builder->create<FuncOp>(loc, "stencil" + fnum++, funcType);
    builder->createBlock(&func.getBody(), {}, typeList);

    // Add parameter fields to the context.
    for (auto arg : func.getArguments())
      c->fields.insert(&arg);

    return func;
  }
};

using IntVec = llvm::ArrayRef<int64_t>;

class FieldAccessAccumulator : public Accumulator, RandGenerator {
private:
  Distribution<Uniform, IntVec> accesses;
  map<Operation *, vector<IntVec*>> used_offsets;
public:
  FieldAccessAccumulator(set<Operation *> ops, OpBuilder* b, Location l)
    : Accumulator(b, l) {
      for (auto op : ops) {
        auto access = dyn_cast<stencil::AccessOp>(op);
        auto offset = access.offset().getValue();
        int64_t ioffset = offset[0].cast<IntegerAttr>().getInt();
        int64_t joffset = offset[1].cast<IntegerAttr>().getInt();
        int64_t koffset = offset[2].cast<IntegerAttr>().getInt();
        accesses.insert(new IntVec({ioffset, joffset, koffset}));
      }
  }

  stencil::AccessOp get(Context* c) {
    auto field = c->fields.sample();
    auto def_op = field->getDefiningOp();

    accesses.setView([&](IntVec* o) {
      for (auto oo : used_offsets[def_op]) {
        if (memcmp(o->data(), oo->data(), 3)) return false;
      }
      return true;
    }, "FieldAccessFilter");

    if (!accesses.size())
      return nullptr;

    auto offset = accesses.sample();
    used_offsets[def_op].push_back(offset);
    accesses.unsetView("FieldAccessFilter");

    auto access = builder->create<stencil::AccessOp>(loc, *field, *offset);
    c->values.insert(access.getOperation());
    return access;
  }
};

class ConstantAccumulator : public Accumulator, RandGenerator{
private:
  map<double, int> constants;
  int size = 0;

public:
  ConstantAccumulator(set<Operation *> ops, OpBuilder* b, Location l) 
    : Accumulator(b, l) {
    for (auto op : ops) {
      if (auto con = dyn_cast<ConstantOp>(op)) {
        auto attr = con.getValue();
        double value;
        if (attr.getKind() == StandardAttributes::Kind::Float) {
          value = attr.cast<FloatAttr>().getValueAsDouble();
        } else if (attr.getKind() == StandardAttributes::Kind::Integer) {
          value = (double)attr.cast<IntegerAttr>().getInt();
        }
        constants[value]++;
        size++;
      }
    }
  }

  ConstantOp get(Context* context) {
    int rand = rand_range(0, size), sum = 0, c_val;
    for (auto it : constants) {
      sum += it.second;
      if (sum >= rand) {
        c_val = it.first;
        break;
      }
    }
    assert(sum >= rand);
    auto c = builder->create<ConstantOp>(
      loc,
      builder->getF64Type(),
      builder->getF64FloatAttr(c_val)
    );
    context->values.insert(c);
    return c;
  }
};

class RandomStencilGenerator : public RandGenerator {
public:
  RandomStencilGenerator(ModuleOp &m)
    : builder(new OpBuilder(m.getContext())),
      loc(m.getLoc()), chain(m) {
    aggregate(m);
    setHandlers();
    learnDistributions(m);
    unOps = {"std.sqrt", "stencil.fabs", "stencil.exp", "stencil.pow"};
    binOps = {"std.addf", "std.cmpf", "std.divf", "std.mulf", "std.subf"};
  }

  bool isValid(string op, Context* c) {
    if (op == "stencil.access" && c->fields.empty())
      return false;
    if (op == "stencil.store" && (c->fields.empty() || c->values.empty()))
      return false;
    if (op == "std.select" && (c->values.size() < 2 || c->bool_values.empty()))
      return false;
    if (op == "scf.if" && (c->bool_values.empty()))
      return false;
    if (unOps.find(op) != unOps.end() && c->values.empty())
      return false;
    if (binOps.find(op) != binOps.end() && c->values.size() < 2)
      return false;

    return true;
  }

  void schedule_if_ops(Context * context, vector<string> &chain) {
    context->index = 0;

    int terms_to_see = 0;
    int ifs_to_see = 0;
    for (auto op : chain) {
      if (op == "scf.yield")
        terms_to_see++;
      else if (op == "scf.if")
        ifs_to_see++;
    }

    auto update_context = [&](vector<int> * top) {
      context->if_else.insert(top->at(0));
      context->then_terms.insert(top->at(1));
    };

    stack<vector<int>> stack;

    int i = 0;
    for (auto op : chain) {
      if (op == "scf.if") {
        stack.push({i});
        ifs_to_see--;
        i++;
      } else if (op == "scf.yield") {
        auto * top = &stack.top();
        if (top->size() == 3) {
          update_context(top);
          stack.pop();
          top = &stack.top();
        }
        if (top->size() == 1) {
          top->push_back(i); // every if has to have a body
        } else {
          assert(top->size() == 2);
          // give else blocks until we can
          if (terms_to_see - ifs_to_see >= (int)stack.size()) {
            update_context(top);
            stack.pop();
          } else {
            stack.pop();
            stack.top().push_back(i);
          }
        }
        terms_to_see--;
        i++;
      }
    }

    while (!stack.empty()) {
      if (stack.top().size() == 3)
        update_context(&stack.top());
      stack.pop();
    }
  }

  ModuleOp generateRandomProgram(llvm::raw_ostream &output) {
    if (DEBUG > 0) {
      chain.print(output);
      output << "\n\n";
    }
    auto program_chain = chain.generateRandomSequence(500);
    Operation * moduleop;
    Context* c = new Context();
    schedule_if_ops(c, program_chain);
    for (auto op : program_chain) {
      if (DEBUG) {
        output << op << "\n";
        c->print(output);
      }

      if (!isValid(op, c)) {
        if (DEBUG)
          output << "skipping - invalid\n";
        continue;
      }
      auto opp = handlers[op](c);
      if (op == "module")
        moduleop = opp;
    }

    return dyn_cast<ModuleOp>(moduleop);
  }

  ModuleOp generateRandomProgramOnTheFly(llvm::raw_ostream &output) {
    if (DEBUG > 0) {
      chain.print(output);
      output << "\n\n";
    }

    int i = 0;
    const int MIN_SIZE = 20, MAX_SIZE = 500;
    string curr_op = "module";
    Context* c = new Context();
    ModuleOp moduleop = dyn_cast<ModuleOp>(handlers[curr_op](c));
    while (curr_op != "module_terminator") {
      auto f = [&](string s) {
        return isValid(s, c) && s != "scf.if" && s != "scf.yield"
          && (i < MIN_SIZE ? s != "std.return" : true);
      };

      curr_op = chain.getNextOp(curr_op, f);
      if (DEBUG) {
        output << curr_op << "\n";
        c->print(output);
      }

      if (i == MAX_SIZE && curr_op != "module_terminator")
        curr_op = "std.return";

      handlers[curr_op](c);
      i++;
    }

    return moduleop;
  }

private:
  map<string, function<Operation*(Context*)>> handlers;
  map<string, set<Operation *>> aggregated;
  set<string> unOps;
  set<string> binOps;
  OpBuilder * builder;
  Location loc;

  // Accumulators
  FuncAccumulator* funcAccumulator;
  FieldAccessAccumulator* fieldAccessAccumulator;
  ConstantAccumulator* constantAccumulator;

  int stencil_name_counter = 0;
  int field_name_counter = 0;
  int var_name_counter = 0;

  MarkovChain chain;
  
  void aggregate(ModuleOp &m) {
    m.walk([&](Operation *op) {
      aggregated[name(op)].insert(op);
    });
  }

  void setHandlers() {
    handlers = {
      // Standard structured ops.
      {"module", [&](Context* c) { return getModule(c); }},
      {"module_terminator", [&](Context* c) { return getModuleTerminator(c); }},
      {"func", [&](Context* c) { return getFunc(c); }},
      {"std.return", [&](Context* c) { return getStdReturn(c); }},

      // Standard arithmetic ops.
      {"std.addf", [&](Context* c) { return getAddf(c); }},
      {"std.subf", [&](Context* c) { return getSubf(c); }},
      {"std.mulf", [&](Context* c) { return getMulf(c); }},
      {"std.divf", [&](Context* c) { return getDivf(c); }},
      {"std.cmpf", [&](Context* c) { return getCmpf(c); }},

      {"std.constant", [&](Context* c) { return getConstant(c); }},
      {"std.negf", [&](Context* c) {return getNegf(c); }},
      {"std.sqrt", [&](Context* c) {return getSqrt(c); }},
      {"std.select", [&](Context* c) { return getSelect(c); }},

      // Control flow ops.
      {"scf.if", [&](Context* c) { return getScfIf(c); }},
      {"scf.yield", [&](Context* c) { return getScfYield(c); }},

      // Stencil ops.
      {"stencil.access", [&](Context* c) { return getStencilAccess(c); }},
      {"stencil.apply", [&](Context* c) { return getStencilApply(c); }},
      {"stencil.assert", [&](Context* c) { return getStencilAssert(c); }},
      {"stencil.load", [&](Context* c) { return getStencilLoad(c); }},
      {"stencil.store", [&](Context* c) { return getStencilStore(c); }},
      {"stencil.index", [&](Context* c) { return getStencilIndex(c); }},
      {"stencil.return", [&](Context* c) { return getStencilReturn(c); }},
    };
  }

  void learnDistributions(ModuleOp &module) {
    funcAccumulator = 
      new FuncAccumulator(aggregated["func"], builder, loc);
    fieldAccessAccumulator = 
      new FieldAccessAccumulator(aggregated["stencil.access"], builder, loc);
    constantAccumulator = 
      new ConstantAccumulator(aggregated["std.constant"], builder, loc);
  }

  //===-------------------------------------------------------------------===//
  // Standard Structured Ops.
  //===-------------------------------------------------------------------===//

  Operation * getModule(experimental::Context* context) {
    auto module = ModuleOp::create(loc);
    builder->setInsertionPointToStart(module.getBody());
    return module;
  }

  Operation * getModuleTerminator(experimental::Context* context) {
    return nullptr;
  }

  Operation * getFunc(experimental::Context* context) {
    auto func = funcAccumulator->get(context);
    builder->setInsertionPointToStart(&func.getBody().front());
    return func;
  }

  Operation * getStdReturn(experimental::Context* context) {
    return nullptr;
  }

  //===-------------------------------------------------------------------===//
  // Standard arithmetic ops.
  //===-------------------------------------------------------------------===//

  Operation * getAddf(experimental::Context* context) {
    return getBinOp<AddFOp>(context, nonZeroConstView);
  }

  Operation * getSubf(experimental::Context* context) {
    return getBinOp<SubFOp>(context, nonZeroConstView);
  }

  Operation * getMulf(experimental::Context* context) {
    return getBinOp<MulFOp>(context, nonZeroOrOneConstView);
  }

  Operation * getDivf(experimental::Context* context) {
    return getBinOp<DivFOp>(context, nonZeroOrOneConstView);
  }

  Operation * getCmpf(experimental::Context* context) {
    // TODO(limo1996): prevent comparison of 2 constants
    auto values = context->values.sample(2);
    auto cmpf = builder->create<CmpFOp>(loc, 
      static_cast<CmpFPredicate>(
        rand_range(
          static_cast<int>(CmpFPredicate::UEQ), 
          static_cast<int>(CmpFPredicate::UNE)
        )
      ),
      values[0]->getResult(0),
      values[1]->getResult(0)
    );
    context->bool_values.insert(cmpf);
    return cmpf;
  }

  Operation * getConstant(experimental::Context* context) {
    return constantAccumulator->get(context);
  }

  Operation * getNegf(experimental::Context* context) {
    return getUnOp<NegFOp>(context, nonZeroConstView);
  }

  //===-------------------------------------------------------------------===//
  // Control flow ops.
  //===-------------------------------------------------------------------===//

  Operation * getScfIf(experimental::Context* context) {
    bool hasElseRegion = 
      context->if_else.find(context->index) != context->if_else.end();
    auto cond = context->bool_values.sample()->getResult(0);
    auto ifOp = builder->create<scf::IfOp>(loc, cond, hasElseRegion);
    builder->setInsertionPointToStart(&ifOp.thenRegion().front());
    context->index++;
    return ifOp;
  }

  Operation * getScfYield(experimental::Context* context) {
    auto block = builder->getInsertionBlock();
    auto ifOp = dyn_cast<scf::IfOp>(block->back().getParentOp());
    if (!ifOp)
      return nullptr;

    auto term = builder->create<scf::YieldOp>(loc);
    auto& terms = context->then_terms;
    if (terms.find(context->index) != terms.end()) {
      // This terminator terminates thenRegion of IfOp that has elseRegion
      builder->setInsertionPointToStart(&ifOp.elseRegion().front());
    } else {
      // This terminator either terminates elseRegion or thenRegion
      // of IfOp that does not have elseRegion
      builder->setInsertionPointAfter(ifOp.getOperation());
    }

    // we cannot use values defined in this block elsewhere
    for (auto& op : *block) {
      context->values.erase(&op);
      context->bool_values.erase(&op);
    }

    context->index++;
    return term;
  }


  //===-------------------------------------------------------------------===//
  // Stencil ops.
  //===-------------------------------------------------------------------===//

  Operation * getStencilAccess(experimental::Context* context) {
    return nullptr;
  }

  Operation * getStencilApply(experimental::Context* context) {
    return nullptr;
  }

  Operation * getStencilAssert(experimental::Context* context) {
    return nullptr;
  }

  Operation * getStencilLoad(experimental::Context* context) {
    return nullptr;
  }

  Operation * getStencilStore(experimental::Context* context) {
    return nullptr;
  }

  Operation * getStencilIndex(experimental::Context* context) {
    return nullptr;
  }

  Operation * getStencilReturn(experimental::Context* context) {
    return nullptr;
  }

  //===-------------------------------------------------------------------===//
  // Arithmetics 2.
  //===-------------------------------------------------------------------===//

  Operation * getSelect(experimental::Context* context) {
    auto values = context->values.sample(2);
    auto cond = context->bool_values.sample()->getResult(0);
    auto select = builder->create<SelectOp>(
      loc, cond, 
      values[0]->getResult(0),
      values[1]->getResult(0)
    );
    context->values.insert(select);
    return select;
  }

  Operation * getSqrt(experimental::Context* context) {
    return getUnOp<SqrtOp>(context,
      [](Operation * op) { return getConstValOrDefault(op, 1.0) >= 0.0; });
  }

  /*Operation * getPow(experimental::Context* context) {
    return getBinOp<stencil::PowOp>(context, nonZeroOrOneConstView);
  }

  Operation * getFabs(experimental::Context* context) {
    return getUnOp<stencil::FabsOp>(context);
  }

  Operation * getExp(experimental::Context* context) {
    return getUnOp<stencil::ExpOp>(context, nonZeroOrOneConstView);
  }*/


  //===-------------------------------------------------------------------===//
  // Utils
  //===-------------------------------------------------------------------===//

  static bool getConstValOrDefault(Operation * op, double def) {
    double value = def;
    if (auto c = dyn_cast<ConstantOp>(op)) {
      auto attr = c.getValue();
      if (attr.getKind() == StandardAttributes::Kind::Float) {
        value = attr.cast<FloatAttr>().getValueAsDouble();
      } else if (attr.getKind() == StandardAttributes::Kind::Integer) {
        value = (double)attr.cast<IntegerAttr>().getInt();
      }
    }
    return value;
  }

  static bool nonZeroConstView(Operation * op) {
    return getConstValOrDefault(op, 1.0) != 0.0;
  }

  static bool nonOneConstView(Operation * op) {
    return getConstValOrDefault(op, 0.0) != 1.0;
  }

  static bool nonZeroOrOneConstView(Operation * op) {
    return nonZeroConstView(op) && nonOneConstView(op);
  }

  template<typename T>
  Block* getBody(T op) {
    return &op.getOperation()->getRegion(0).front();
  }

  template<typename T>
  Operation * getBinOp(experimental::Context* context, 
                       function<bool(Operation*)> view = nullptr) {
    context->values.setView(view, "BinOp#1");
    if (context->values.size() < 2) {
      context->values.unsetView("BinOp#1");
      return nullptr;
    }

    auto values = context->values.sample(2);
    context->values.unsetView("BinOp#1");
    auto binop = builder->create<T>(
      loc,
      values[0]->getResult(0),
      values[1]->getResult(0)
    );
    context->values.insert(binop);
    return binop;
  }

  template<typename T>
  Operation * getUnOp(experimental::Context* context,
                      function<bool(Operation*)> view = nullptr) {
    context->values.setView(view, "UnOp#1");
    if (context->values.size() < 1) {
      context->values.unsetView("UnOp#1");
      return nullptr;
    }
    auto value = context->values.sample();
    context->values.unsetView("UnOp#1");
    auto unop = builder->create<T>(loc, value->getResult(0));
    context->values.insert(unop);
    return unop;
  }
};

} // namespace experimental
} // namespace mlir

#endif MLIR_TARGET_STENCIL_RANDOMSTENCILGENERATOR_H