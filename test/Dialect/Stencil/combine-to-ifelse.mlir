// RUN: oec-opt %s -split-input-file --stencil-combine-to-ifelse -cse | oec-opt | FileCheck %s
// RUN: oec-opt %s -split-input-file --stencil-combine-to-ifelse='internal-only' -cse | oec-opt | FileCheck --check-prefix=CHECKINT %s

// CHECK-LABEL: func @simple
// CHECKINT-LABEL: func @simple
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECKINT-COUNT-2: {{%.*}} = stencil.apply
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0([0, 0, 0] : [65, 64, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<65x64x60xf64>
  // CHECK: {{%.*}} = stencil.apply ([[ARG0:%.*]] = {{%.*}} : !stencil.temp<65x64x60xf64>, [[ARG1:%.*]] = {{%.*}} : !stencil.temp<65x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
  // CHECK-DAG: [[IDX:%.*]] = stencil.index 0 [0, 0, 0] : index
  // CHECK-DAG: [[C32:%.*]] = constant 32 : index
  // CHECK-DAG: [[COND:%.*]] = cmpi "ult", [[IDX]], [[C32]] : index
  // CHECK: [[RES:%.*]] = scf.if [[COND]] -> (!stencil.result<f64>) {
  // CHECK-NEXT: [[ACC1:%.*]] = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<65x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES1:%.*]] = stencil.store_result [[ACC1]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT:  scf.yield [[RES1]] : !stencil.result<f64>
  // CHECK-NEXT: } else {
  // CHECK-NEXT: [[ACC2:%.*]] = stencil.access [[ARG1]] [1, 0, 0] : (!stencil.temp<65x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES2:%.*]] = stencil.store_result [[ACC2]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT:  scf.yield [[RES2]] : !stencil.result<f64>
  // CHECK-NEXT: }
  // CHECK-NEXT: stencil.return [[RES]] : !stencil.result<f64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<65x64x60xf64>) -> !stencil.temp<32x64x60xf64> {
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<65x64x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [32, 64, 60])
  %4 = stencil.apply (%arg2 = %2 : !stencil.temp<65x64x60xf64>) -> !stencil.temp<32x64x60xf64> {
    %6 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<65x64x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([32, 0, 0] : [64, 64, 60])
  // CHECK-NOT: {{%.*}} = stencil.combine
  %5 = stencil.combine 0 at 32 lower = (%3 : !stencil.temp<32x64x60xf64>) upper = (%4 : !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>
  stencil.store %5 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @nested
// CHECKINT-LABEL: func @nested
func @nested(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECKINT-COUNT-3: {{%.*}} = stencil.apply
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0([-1, 0, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x66x60xf64>
  // CHECK: {{%.*}} = stencil.apply ([[ARG0:%.*]] = {{%.*}} : !stencil.temp<66x66x60xf64>, [[ARG1:%.*]] = {{%.*}} : !stencil.temp<66x66x60xf64>, [[ARG2:%.*]] = {{%.*}} : !stencil.temp<66x66x60xf64>) -> !stencil.temp<64x66x60xf64> {
  // CHECK: {{%.*}} = stencil.index 1 [0, 0, 0] : index
  // CHECK: {{%.*}} = scf.if {{%.*}} -> (!stencil.result<f64>) {
  // CHECK: {{%.*}} = stencil.index 0 [0, 0, 0] : index
  // CHECK: {{%.*}} = scf.if {{%.*}} -> (!stencil.result<f64>) {
  // CHECK: {{%.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<66x66x60xf64>) -> f64
  // CHECK: } else {
  // CHECK: {{%.*}} = stencil.access [[ARG1]] [1, 0, 0] : (!stencil.temp<66x66x60xf64>) -> f64
  // CHECK: } else {
  // CHECK: {{%.*}} = stencil.access [[ARG2]] [-1, 0, 0] : (!stencil.temp<66x66x60xf64>) -> f64
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<66x66x60xf64>) -> !stencil.temp<32x64x60xf64> {
    %7 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<66x66x60xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 0, 0] : [32, 64, 60])
  %4 = stencil.apply (%arg2 = %2 : !stencil.temp<66x66x60xf64>) -> !stencil.temp<32x64x60xf64> {
    %7 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<66x66x60xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([32, 0, 0] : [64, 64, 60])
  %5 = stencil.combine 0 at 32 lower = (%3 : !stencil.temp<32x64x60xf64>) upper = (%4 : !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>
  %6 = stencil.apply (%arg2 = %2 : !stencil.temp<66x66x60xf64>) -> !stencil.temp<64x2x60xf64> {
    %7 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<66x66x60xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 64, 0] : [64, 66, 60])
  // CHECK: } to ([0, 0, 0] : [64, 66, 60])
  // CHECK-NOT: {{%.*}} = stencil.combine
  %7 = stencil.combine 1 at 64 lower = (%5 : !stencil.temp<64x64x60xf64>) upper = (%6 : !stencil.temp<64x2x60xf64>) ([0, 0, 0] : [64, 66, 60]) : !stencil.temp<64x66x60xf64>
  stencil.store %7 to %1([0, 0, 0] : [64, 66, 60]) : !stencil.temp<64x66x60xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @internal
// CHECKINT-LABEL: func @internal
func @internal(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECK-COUNT-2: {{%.*}} = stencil.apply
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0([0, 0, 0] : [64, 64, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<64x64x60xf64>
  // CHECKINT: {{%.*}} = stencil.apply ({{%.*}} = {{%.*}} : !stencil.temp<64x64x60xf64>, {{%.*}} = {{%.*}} : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
  // CHECKINT-DAG: [[IDX:%.*]] = stencil.index 0 [0, 0, 0] : index
  // CHECKINT-DAG: [[C32:%.*]] = constant 32 : index
  // CHECKINT-DAG: [[COND:%.*]] = cmpi "ult", [[IDX]], [[C32]] : index
  // CHECKINT: {{%.*}} = scf.if [[COND]] -> (!stencil.result<f64>) {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<32x64x60xf64> {
    %7 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 0, 0] : [32, 64, 60])
  %4 = stencil.apply (%arg2 = %2 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<32x64x60xf64> {
    %7 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([32, 0, 0] : [64, 64, 60])
  // CHECKINT: } to ([0, 0, 0] : [64, 64, 60])
  %5 = stencil.combine 0 at 32 lower = (%3 : !stencil.temp<32x64x60xf64>) upper = (%4 : !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>
  %6 = stencil.apply (%arg2 = %5 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %7 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  // CHECKINT: } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %6 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @multiple_extra
// CHECKINT-LABEL: func @multiple_extra
func @multiple_extra(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECKINT-COUNT-1: {{%.*}} = stencil.apply
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.cast %arg2([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %3 = stencil.cast %arg3([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  // CHECK: [[APPLY_RES:%.*]]:4 = stencil.apply
  // CHECK-DAG: [[CST0:%.*]] = constant 0.000000e+00 : f64
  // CHECK-DAG: [[CST1:%.*]] = constant 1.000000e+00 : f64
  // CHECK-DAG: [[CST2:%.*]] = constant 2.000000e+00 : f64
  // CHECK-DAG: [[CST3:%.*]] = constant 3.000000e+00 : f64
  // CHECK-DAG: [[CST4:%.*]] = constant 4.000000e+00 : f64
  // CHECK: [[IF_RES:%.*]]:4 = scf.if {{%.*}} -> (!stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>) {
  %4:2 = stencil.apply -> (!stencil.temp<48x64x60xf64>, !stencil.temp<48x64x60xf64>) {
    %cst = constant 0.000000e+00 : f64
    %cst_0 = constant 1.000000e+00 : f64
    // CHECK-DAG: [[RES0:%.*]] = stencil.store_result [[CST0]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES1:%.*]] = stencil.store_result [[CST1]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES2:%.*]] = stencil.store_result : () -> !stencil.result<f64>
    %7 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    %8 = stencil.store_result %cst_0 : (f64) -> !stencil.result<f64>
    // CHECK-NEXT: scf.yield [[RES0]], [[RES1]], [[RES2]], [[RES2]] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
    stencil.return %7, %8 : !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0] : [48, 64, 60])
  %5:3 = stencil.apply -> (!stencil.temp<16x64x60xf64>, !stencil.temp<16x64x60xf64>, !stencil.temp<16x64x60xf64>) {
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 3.000000e+00 : f64
    %cst_1 = constant 4.000000e+00 : f64
    // CHECK-DAG: [[RES3:%.*]] = stencil.store_result [[CST2]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES4:%.*]] = stencil.store_result [[CST3]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES5:%.*]] = stencil.store_result [[CST4]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES6:%.*]] = stencil.store_result : () -> !stencil.result<f64>
    %7 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    %8 = stencil.store_result %cst_0 : (f64) -> !stencil.result<f64>
    %9 = stencil.store_result %cst_1 : (f64) -> !stencil.result<f64>
    // CHECK-NEXT: scf.yield [[RES4]], [[RES6]], [[RES3]], [[RES5]] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
    stencil.return %7, %8, %9 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  } to ([48, 0, 0] : [64, 64, 60])
  // CHECK: stencil.return [[IF_RES]]#0, [[IF_RES]]#1, [[IF_RES]]#2, [[IF_RES]]#3 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  %6:4 = stencil.combine 0 at 48 lower = (%4#0 : !stencil.temp<48x64x60xf64>) upper = (%5#1 : !stencil.temp<16x64x60xf64>) lowerext = (%4#1 : !stencil.temp<48x64x60xf64>) upperext = (%5#0, %5#2 : !stencil.temp<16x64x60xf64>, !stencil.temp<16x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>
  // CHECK: stencil.store [[APPLY_RES]]#0 to {{%.*}}([0, 0, 0] : [64, 64, 60])
  // CHECK: stencil.store [[APPLY_RES]]#1 to {{%.*}}([0, 0, 0] : [48, 64, 60])
  // CHECK: stencil.store [[APPLY_RES]]#2 to {{%.*}}([48, 0, 0] : [64, 64, 60])
  // CHECK: stencil.store [[APPLY_RES]]#3 to {{%.*}}([48, 0, 0] : [64, 64, 60])
  stencil.store %6#0 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %6#1 to %1([0, 0, 0] : [48, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %6#2 to %2([48, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %6#3 to %3([48, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @single_extra
// CHECKINT-LABEL: func @single_extra
func @single_extra(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECKINT-COUNT-1: {{%.*}} = stencil.apply
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.cast %arg3([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  // CHECK: [[APPLY_RES:%.*]]:3 = stencil.apply
  // CHECK-DAG: [[CST0:%.*]] = constant 0.000000e+00 : f64
  // CHECK-DAG: [[CST1:%.*]] = constant 1.000000e+00 : f64
  // CHECK-DAG: [[CST2:%.*]] = constant 2.000000e+00 : f64
  // CHECK-DAG: [[CST3:%.*]] = constant 3.000000e+00 : f64
  // CHECK-DAG: [[CST4:%.*]] = constant 4.000000e+00 : f64
  // CHECK: [[IF_RES:%.*]]:3 = scf.if {{%.*}} -> (!stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>) {
  %4:2 = stencil.apply -> (!stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) {
    %cst = constant 0.000000e+00 : f64
    %cst_0 = constant 1.000000e+00 : f64
    // CHECK-DAG: [[RES0:%.*]] = stencil.store_result [[CST0]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES1:%.*]] = stencil.store_result [[CST1]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES2:%.*]] = stencil.store_result : () -> !stencil.result<f64>
    %7 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    %8 = stencil.store_result %cst_0 : (f64) -> !stencil.result<f64>
    // CHECK-NEXT: scf.yield [[RES1]], [[RES0]], [[RES2]] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
    stencil.return %7, %8 : !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0] : [32, 64, 60])
  %5:3 = stencil.apply -> (!stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) {
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 3.000000e+00 : f64
    %cst_1 = constant 4.000000e+00 : f64
    // CHECK-DAG: [[RES3:%.*]] = stencil.store_result [[CST2]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES4:%.*]] = stencil.store_result [[CST3]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES5:%.*]] = stencil.store_result [[CST4]] : (f64) -> !stencil.result<f64>
    %7 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    %8 = stencil.store_result %cst_0 : (f64) -> !stencil.result<f64>
    %9 = stencil.store_result %cst_1 : (f64) -> !stencil.result<f64>
    // CHECK-NEXT: scf.yield [[RES4]], [[RES5]], [[RES3]] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
    stencil.return %7, %8, %9 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  } to ([32, 0, 0] : [64, 64, 60])
  // CHECK: stencil.return [[IF_RES]]#0, [[IF_RES]]#1, [[IF_RES]]#2 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  %6:3 = stencil.combine 0 at 32 lower = (%4#1, %4#0 : !stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) upper = (%5#1, %5#2 : !stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) upperext = (%5#0 : !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>
  // CHECK: stencil.store [[APPLY_RES]]#0 to {{%.*}}([0, 0, 0] : [64, 64, 60])
  // CHECK: stencil.store [[APPLY_RES]]#1 to {{%.*}}([0, 0, 0] : [32, 64, 60])
  // CHECK: stencil.store [[APPLY_RES]]#2 to {{%.*}}([32, 0, 0] : [64, 64, 60])
  stencil.store %6#0 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %6#1 to %1([0, 0, 0] : [32, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %6#2 to %2([32, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @multiple_apply
// CHECKINT-LABEL: func @multiple_apply
func @multiple_apply(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECKINT-COUNT-1: {{%.*}} = stencil.apply
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.cast %arg2([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %3 = stencil.load %2([0, 0, 0] : [32, 64, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<32x64x60xf64>
  %4 = stencil.load %2([0, 0, 0] : [33, 64, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<33x64x60xf64>
  // CHECK: {{%.*}}:2 = stencil.apply 
  // CHECK-DAG: [[CST2:%.*]] = constant 2.000000e+00 : f64
  // CHECK-DAG: [[CST3:%.*]] = constant 3.000000e+00 : f64
  // CHECK: [[IF_RES:%.*]]:2 = scf.if {{%.*}} -> (!stencil.result<f64>, !stencil.result<f64>) {
  %50 = stencil.apply (%arg3 = %3 : !stencil.temp<32x64x60xf64>) -> (!stencil.temp<32x64x60xf64>) {
    // CHECK-DAG: [[VAL0:%.*]] = stencil.access {{%.*}} [0, 0, 0] : (!stencil.temp<32x64x60xf64>) -> f64
    // CHECK-DAG: [[RES0:%.*]] = stencil.store_result [[VAL0]] : (f64) -> !stencil.result<f64>
    %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<32x64x60xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 0, 0] : [32, 64, 60])
  %51 = stencil.apply (%arg4 = %4 : !stencil.temp<33x64x60xf64>) -> (!stencil.temp<32x64x60xf64>) {
    // CHECK-DAG: [[VAL1:%.*]] = stencil.access {{%.*}} [1, 0, 0] : (!stencil.temp<33x64x60xf64>) -> f64
    // CHECK-DAG: [[RES1:%.*]] = stencil.store_result [[VAL1]] : (f64) -> !stencil.result<f64>
    %7 = stencil.access %arg4 [1, 0, 0] : (!stencil.temp<33x64x60xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
    // CHECK-NEXT: scf.yield [[RES0]], [[RES1]] : !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0] : [32, 64, 60])
  // CHECK: } else {
  %6:2 = stencil.apply -> (!stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) {
    // CHECK-DAG: [[RES2:%.*]] = stencil.store_result [[CST2]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES3:%.*]] = stencil.store_result [[CST3]] : (f64) -> !stencil.result<f64>
    %cst = constant 2.000000e+00 : f64
    %cst_0 = constant 3.000000e+00 : f64
    %7 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    %8 = stencil.store_result %cst_0 : (f64) -> !stencil.result<f64>
    // CHECK-NEXT: scf.yield [[RES2]], [[RES3]] : !stencil.result<f64>, !stencil.result<f64>
    stencil.return %7, %8 : !stencil.result<f64>, !stencil.result<f64>
  } to ([32, 0, 0] : [64, 64, 60])  
  // CHECK: stencil.return [[IF_RES]]#0, [[IF_RES]]#1 : !stencil.result<f64>, !stencil.result<f64>
  %7:2 = stencil.combine 0 at 32 lower = (%50, %51 : !stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) upper = (%6#0, %6#1 : !stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>
  stencil.store %7#0 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %7#1 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  return
}