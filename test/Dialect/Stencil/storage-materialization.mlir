// RUN: oec-opt %s -split-input-file --stencil-storage-materialization -cse | oec-opt | FileCheck %s

// CHECK-LABEL: func @apply
func @apply(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  // CHECK: [[VAL1:%.*]] = stencil.apply -> !stencil.temp<64x64x60xf64> {
  %2 = stencil.apply -> !stencil.temp<64x64x60xf64> {
    %cst = constant 0.000000e+00 : f64
    %8 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  // CHECK: [[VAL2:%.*]] = stencil.buffer [[VAL1]] : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
  // CHECK: [[VAL3:%.*]] = stencil.apply ({{%.*}} = [[VAL2:%.*]]: !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  // CHECK: [[VAL4:%.*]] = stencil.apply ({{%.*}} = [[VAL3:%.*]] : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %4 = stencil.apply (%arg2 = %3 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  // CHECK-DAG: stencil.store [[VAL3]] to {{%.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  // CHECK-DAG: stencil.store [[VAL4]] to {{%.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %4 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @combine
func @combine(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.apply -> !stencil.temp<32x64x60xf64> {
    %cst = constant 0.000000e+00 : f64
    %8 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 0, 0] : [32, 64, 60])
  %2 = stencil.apply -> !stencil.temp<32x64x60xf64> {
    %cst = constant 1.000000e+00 : f64
    %8 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([32, 0, 0] : [64, 64, 60])
  // CHECK: [[VAL1:%.*]] = stencil.combine
  %3 = stencil.combine 0 at 32 lower = (%1 : !stencil.temp<32x64x60xf64>) upper = (%2 : !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>
  // CHECK: [[VAL2:%.*]] = stencil.buffer [[VAL1]] : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
  // CHECK: [[VAL3:%.*]] = stencil.apply ({{%.*}} = [[VAL2:%.*]]: !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %4 = stencil.apply (%arg2 = %3 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])  
  // CHECK-DAG: stencil.store [[VAL3]] to {{%.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %4 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  return
}
