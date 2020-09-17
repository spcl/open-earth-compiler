// RUN: oec-opt %s -split-input-file --stencil-combine-lowering -cse | oec-opt | FileCheck %s
// RUN: oec-opt %s -split-input-file --stencil-combine-lowering='internal-only' -cse | oec-opt | FileCheck --check-prefix=CHECKINT %s

// CHECK-LABEL: func @simple
// CHECKINT-LABEL: func @simple
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECKINT-COUNT-2: {{%.*}} = stencil.apply
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0([0, 0, 0] : [65, 64, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<65x64x60xf64>
  // CHECK-DAG: [[IDX:%.*]] = stencil.index 0 [0, 0, 0] : index
  // CHECK-DAG: [[C32:%.*]] = constant 32 : index
  // CHECK-DAG: [[COND:%.*]] = cmpi "ult", [[IDX]], [[C32]] : index
  // CHECK: [[RES:%.*]] = scf.if [[COND]] -> (!stencil.result<f64>) {
  // CHECK-NEXT: [[ACC1:%.*]] = stencil.access {{%.*}}[0, 0, 0] : (!stencil.temp<65x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES1:%.*]] = stencil.store_result [[ACC1]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT:  scf.yield [[RES1]] : !stencil.result<f64>
  // CHECK-NEXT: } else {
  // CHECK-NEXT: [[ACC2:%.*]] = stencil.access {{%.*}}[1, 0, 0] : (!stencil.temp<65x64x60xf64>) -> f64
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
  // CHECK: {{%.*}} = stencil.apply ({{%.*}} = {{%.*}} : !stencil.temp<66x66x60xf64>, {{%.*}} = {{%.*}} : !stencil.temp<66x66x60xf64>, {{%.*}} = {{%.*}} : !stencil.temp<66x66x60xf64>) -> !stencil.temp<64x66x60xf64> {
  // CHECK: {{%.*}} = stencil.index 1 [0, 0, 0] : index
  // CHECK: {{%.*}} = scf.if {{%.*}} -> (!stencil.result<f64>) {
  // CHECK: {{%.*}} = stencil.index 0 [0, 0, 0] : index
  // CHECK: {{%.*}} = scf.if {{%.*}} -> (!stencil.result<f64>) {
  // CHECK: {{%.*}} = stencil.access {{%.*}} [0, 0, 0] : (!stencil.temp<66x66x60xf64>) -> f64
  // CHECK: } else {
  // CHECK: {{%.*}} = stencil.access {{%.*}} [1, 0, 0] : (!stencil.temp<66x66x60xf64>) -> f64
  // CHECK: } else {
  // CHECK: {{%.*}} = stencil.access {{%.*}} [-1, 0, 0] : (!stencil.temp<66x66x60xf64>) -> f64
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
