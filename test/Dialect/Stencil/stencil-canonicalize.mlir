// RUN: oec-opt %s -split-input-file --canonicalize | oec-opt | FileCheck %s

// CHECK-LABEL: func @apply_arg
// CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<?x?x?xf64>) ->
// CHECK: stencil.return %{{.*}} : !stencil.result<f64>
func @apply_arg(%arg0: !stencil.temp<?x?x?xf64>, %arg1: !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> attributes {stencil.program} {
  %4 = stencil.apply (%arg2 = %arg0 : !stencil.temp<?x?x?xf64>, %arg3 = %arg0 : !stencil.temp<?x?x?xf64>, %arg4 = %arg1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %6 = stencil.access %arg3[-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.access %arg2[0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %8 = addf %6, %7 : f64
    %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    stencil.return %9 : !stencil.result<f64>
  }
  return %4 : !stencil.temp<?x?x?xf64>
}

// -----

// CHECK-LABEL: func @apply_res
// CHECK: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK-DAG: [[VAL0:%.*]] = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
// CHECK-DAG: [[VAL1:%.*]] = stencil.access [[ARG0]] [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
// CHECK-DAG: [[RES0:%.*]] = stencil.store_result [[VAL0]] : (f64) -> !stencil.result<f64>
// CHECK-DAG: [[RES1:%.*]] = stencil.store_result [[VAL1]] : (f64) -> !stencil.result<f64>
// CHECK:  stencil.return unroll [1, 2, 1] [[RES0]], [[RES1]] : !stencil.result<f64>, !stencil.result<f64>
func @apply_res(%arg0: !stencil.temp<?x?x?xf64>, %arg1: !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> attributes {stencil.program} {
  %1, %2 = stencil.apply (%arg2 = %arg0 : !stencil.temp<?x?x?xf64>, %arg3 = %arg1 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
    %3 = stencil.access %arg2[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2[0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.access %arg3[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = stencil.access %arg3[0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.store_result %3 : (f64) -> !stencil.result<f64>
    %8 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
    %9 = stencil.store_result %5 : (f64) -> !stencil.result<f64>
    %10 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return unroll [1, 2, 1] %7, %8, %9, %10 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  }
  return %1 : !stencil.temp<?x?x?xf64>
}

// -----

// CHECK-LABEL: func @apply_load
func @apply_load(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) -> !stencil.temp<64x64x64xf64> attributes {stencil.program} {
  %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
  %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
  // CHECK: [[VAL0:%.*]] = stencil.load %{{.*}}([0, 0, 0] : [64, 64, 64]) : (!stencil.field<72x72x72xf64>) -> !stencil.temp<64x64x64xf64>
  // CHECK: [[VAL1:%.*]] = stencil.load %{{.*}}([-1, -1, 0] : [65, 65, 64]) : (!stencil.field<72x72x72xf64>) -> !stencil.temp<66x66x64xf64>
  %2 = stencil.load %0([-1, 0, 0] : [65, 64, 64]) : (!stencil.field<72x72x72xf64>) -> !stencil.temp<66x64x64xf64>
  %3 = stencil.load %0([0, -1, 0] : [64, 65, 64]) : (!stencil.field<72x72x72xf64>) -> !stencil.temp<64x66x64xf64>
  %4 = stencil.load %1([0, 0, 0] : [64, 64, 64]) : (!stencil.field<72x72x72xf64>) -> !stencil.temp<64x64x64xf64>
  // CHECK: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<66x66x64xf64>, [[ARG1:%.*]] = %{{.*}} : !stencil.temp<64x64x64xf64>) -> !stencil.temp<64x64x64xf64> {
  %5 = stencil.apply (%arg2 = %2 : !stencil.temp<66x64x64xf64>, %arg3 = %4 : !stencil.temp<64x64x64xf64>, %arg4 = %3 : !stencil.temp<64x66x64xf64>) -> !stencil.temp<64x64x64xf64> {
    // CHECK: %{{.*}} = stencil.access [[ARG0]] [-1, 0, 0] : (!stencil.temp<66x66x64xf64>) -> f64
    // CHECK: %{{.*}} = stencil.access [[ARG1]] [0, 0, 0] : (!stencil.temp<64x64x64xf64>) -> f64
    // CHECK: %{{.*}} = stencil.access [[ARG0]] [0, 1, 0] : (!stencil.temp<66x66x64xf64>) -> f64
    %6 = stencil.access %arg2[-1, 0, 0] : (!stencil.temp<66x64x64xf64>) -> f64
    %7 = stencil.access %arg3[0, 0, 0] : (!stencil.temp<64x64x64xf64>) -> f64
    %8 = stencil.access %arg4[0, 1, 0] : (!stencil.temp<64x66x64xf64>) -> f64
    %9 = addf %6, %7 : f64
    %10 = addf %9, %8 : f64
    %11 = stencil.store_result %10 : (f64) -> !stencil.result<f64>
    stencil.return %11 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 64])
  return %5 : !stencil.temp<64x64x64xf64>
}

// -----

// CHECK-LABEL: func @hoist
func @hoist(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECK: stencil.cast
  // CHECK: stencil.cast
  // CHECK: stencil.cast
  // CHECK: stencil.load
  // CHECK: stencil.apply
  // CHECK: stencil.apply
  // CHECK: stencil.store
  %1 = stencil.apply -> !stencil.temp<?x?x?xf64> {
    %2 = stencil.store_result : () -> !stencil.result<f64>
    stencil.return %2 : !stencil.result<f64>
  }
  %3 = stencil.cast %arg0 ([-3, -3, 0]:[67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %4 = stencil.cast %arg1 ([-3, -3, 0]:[67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  stencil.store %1 to %4([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  %5 = stencil.load %3 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  %6 = stencil.apply (%arg3 = %5 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  }
  %9 = stencil.cast %arg2 ([-3, -3, 0]:[67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  stencil.store %6 to %9([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @combine_res
func @combine_res(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
  // CHECK: [[VAL0:%.*]] = stencil.apply 
  // CHECK: [[VAL1:%.*]] = stencil.apply 
  %1, %2 = stencil.apply -> (!stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) {
    %cst = constant 0.000000e+00 : f64
    %7 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return %7, %7 : !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0] : [32, 64, 60])
  %3, %4 = stencil.apply -> (!stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) {
    %cst = constant 1.000000e+00 : f64
    %7 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return %7, %7 : !stencil.result<f64>, !stencil.result<f64>
  } to ([32, 0, 0] : [64, 64, 60])
  // CHECK: %{{.*}} = stencil.combine 0 at 32 lower = ([[VAL0]] : !stencil.temp<32x64x60xf64>) upper = ([[VAL1]] : !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>
  %5, %6 = stencil.combine 0 at 32 lower = (%1, %2 : !stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) upper = (%3, %4 : !stencil.temp<32x64x60xf64>, !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>
  stencil.store %6 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>
  return
}

// -----

// CHECK-LABEL: func @combine_empty
func @combine_empty(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
  %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
  %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
  // CHECK: [[VAL0:%.*]]:2 = stencil.apply -> (!stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>) {
  %3, %4 = stencil.apply -> (!stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>) {
    // CHECK: %{{.*}} = constant 0.000000e+00 : f64
    %cst = constant 0.000000e+00 : f64
    %10 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return %10, %10 : !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  %5, %6 = stencil.apply -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
    %cst = constant 1.0 : f64
    %10 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return %10, %10 : !stencil.result<f64>, !stencil.result<f64>
  } 
  // CHECK: [[VAL1:%.*]] = stencil.apply -> !stencil.temp<64x64x60xf64> {
  // CHECK-NEXT: %{{.*}} = stencil.store_result : () -> !stencil.result<f64>
  %7, %8, %9 = stencil.combine 0 at 77 lower = (%3 : !stencil.temp<64x64x60xf64>) upper = (%5 : !stencil.temp<?x?x?xf64>) lowerext = (%4 : !stencil.temp<64x64x60xf64>) upperext = (%6 : !stencil.temp<?x?x?xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>
  // CHECK: stencil.store [[VAL0]]#0 to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>
  stencil.store %7 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>
  // CHECK: stencil.store [[VAL0]]#1 to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>
  stencil.store %8 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>
  // CHECK: stencil.store [[VAL1]] to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>
  stencil.store %9 to %2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>
  return
}

// -----

// CHECK-LABEL: func @combine_symm
func @combine_symm(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
  %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
  %2 = stencil.apply -> !stencil.temp<32x64x60xf64> {
    %7 = stencil.store_result : () -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [32, 64, 60])
  %3 = stencil.apply -> !stencil.temp<32x64x60xf64> {
    %7 = stencil.store_result : () -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([32, 0, 0] : [64, 64, 60])
  // CHECK: [[VAL0:%.*]] = stencil.apply -> !stencil.temp<64x64x60xf64> {
  // CHECK-NEXT: %{{.*}} = constant 0.000000e+00 : f64
  %4 = stencil.apply -> !stencil.temp<64x64x60xf64> {
    %cst = constant 0.000000e+00 : f64
    %10 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return %10 : !stencil.result<f64>
  // CHECK: } to ([0, 0, 0] : [64, 64, 60])
  } to ([0, 0, 0] : [64, 64, 60])
  // CHECK: [[VAL1:%.*]] = stencil.apply -> !stencil.temp<64x64x60xf64> {
  // CHECK-NEXT: %{{.*}} = stencil.store_result : () -> !stencil.result<f64>
  // CHECK: } to ([0, 0, 0] : [64, 64, 60])
  %5, %6 = stencil.combine 0 at 32 lower = (%4, %2 : !stencil.temp<64x64x60xf64>, !stencil.temp<32x64x60xf64>) upper = (%4, %3 : !stencil.temp<64x64x60xf64>, !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>
  // CHECK: stencil.store [[VAL0]] to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>
  // CHECK: stencil.store [[VAL1]] to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>
  stencil.store %5 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>
  stencil.store %6 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<72x72x72xf64>  
  return
}