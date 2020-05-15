// RUN: oec-opt %s -split-input-file --stencil-loop-mapping='block-sizes=64,2,1' -cse | FileCheck %s

// CHECK-LABEL: @simple_tiling
func @simple_tiling(%arg0: memref<64x64x64xf64>, %arg1: memref<64x64x64xf64>) {
  // CHECK-DAG: [[C0:%.*]] = constant 0 : index
  // CHECK-DAG: [[C1:%.*]] = constant 1 : index
  // CHECK-DAG: [[C2:%.*]] = constant 2 : index
  // CHECK-DAG: [[C64:%.*]] = constant 64 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c64 = constant 64 : index
  // CHECK: scf.parallel ({{%.*}}, [[ARG1_1:%.*]], [[ARG3:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C64]], [[C64]], [[C64]]) step ([[C64]], [[C2]], [[C1]]) {
  // CHECK-NEXT: scf.parallel ([[ARG0:%.*]], [[ARG1_2:%.*]], {{%.*}}) = ([[C0]], [[C0]], [[C0]]) to ([[C64]], [[C2]], [[C1]]) step ([[C1]], [[C1]], [[C1]]) {
  scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c64, %c64, %c64) step (%c1, %c1, %c1) {
    // CHECK-NEXT: [[IDX1:%.*]] = affine.apply #map{{[0-9]+}}([[ARG1_1]], [[ARG1_2]]) 
    // CHECK-NEXT: {{%.*}} = load {{%.*}}{{\[}}[[ARG0]], [[IDX1]], [[ARG3]]] 
    %0 = load %arg0[%arg2, %arg3, %arg4] : memref<64x64x64xf64>
    store %0, %arg1[%arg2, %arg3, %arg4] : memref<64x64x64xf64>
  }
  return
}

// -----

// CHECK-LABEL: @simple_mapping
func @simple_mapping(%arg0: memref<64x64x64xf64>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c64 = constant 64 : index
  // CHECK: scf.parallel ({{%.*}}, {{%.*}}, {{%.*}}) = ({{%.*}}, {{%.*}}, {{%.*}}) to ({{%.*}}, {{%.*}}, {{%.*}}) step ({{%.*}}, {{%.*}}, {{%.*}}) {
  // CHECK-NEXT: scf.parallel ({{%.*}}, {{%.*}}, {{%.*}}) = ({{%.*}}, {{%.*}}, {{%.*}}) to ({{%.*}}, {{%.*}}, {{%.*}}) step ({{%.*}}, {{%.*}}, {{%.*}}) {
  scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c64, %c64, %c64) step (%c1, %c1, %c1) {
    %0 = constant 0.0 : f64
    store %0, %arg0[%arg2, %arg3, %arg4] : memref<64x64x64xf64>
    scf.yield
  }
  // CHECK-COUNT-1: {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 3 : i64}
  // CHECK-COUNT-1: {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 4 : i64}
  // CHECK-COUNT-1: {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 5 : i64}
  // CHECK: scf.yield
  // CHECK-COUNT-1: {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 0 : i64}
  // CHECK-COUNT-1: {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 1 : i64}
  // CHECK-COUNT-1: {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 2 : i64}
  return
}

// -----

// CHECK-LABEL: @predicate_execution
func @predicate_execution(%arg0: memref<65x65x65xf64>, %arg1: memref<65x65x65xf64>) {
  // CHECK-DAG: [[C0:%.*]] = constant 0 : index
  // CHECK-DAG: [[C1:%.*]] = constant 1 : index
  // CHECK-DAG: [[C2:%.*]] = constant 2 : index
  // CHECK-DAG: [[C64:%.*]] = constant 64 : index
  // CHECK-DAG: [[C65:%.*]] = constant 65 : index
  // CHECK-DAG: [[C66:%.*]] = constant 66 : index
  // CHECK-DAG: [[C128:%.*]] = constant 128 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c65 = constant 65 : index
  // CHECK: scf.parallel ([[ARG0_1:%.*]], [[ARG1_1:%.*]], [[ARG3:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C128]], [[C66]], [[C65]]) step ([[C64]], [[C2]], [[C1]]) {
  // CHECK-NEXT: scf.parallel ([[ARG0_2:%.*]], [[ARG1_2:%.*]], {{%.*}}) = ([[C0]], [[C0]], [[C0]]) to ([[C64]], [[C2]], [[C1]]) step ([[C1]], [[C1]], [[C1]]) {
  scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c65, %c65, %c65) step (%c1, %c1, %c1) {
    // CHECK-NEXT: [[IDX0:%.*]] = affine.apply #map{{[0-9]+}}([[ARG0_1]], [[ARG0_2]]) 
    // CHECK-NEXT: [[IDX1:%.*]] = affine.apply #map{{[0-9]+}}([[ARG1_1]], [[ARG1_2]]) 
    // CHECK-NEXT: [[CMP0:%.*]] = cmpi "slt", [[IDX0]], [[C65]] : index
    // CHECK-NEXT: [[CMP1:%.*]] = cmpi "slt", [[IDX1]], [[C65]] : index
    // CHECK-NEXT: [[PRED:%.*]] = and [[CMP0]], [[CMP1]] : i1
    // CHECK-NEXT: scf.if [[PRED:%.*]] {
    // CHECK-NEXT: {{%.*}} = load {{%.*}}{{\[}}[[IDX0]], [[IDX1]], [[ARG3]]] 
    %0 = load %arg0[%arg2, %arg3, %arg4] : memref<65x65x65xf64>
    store %0, %arg1[%arg2, %arg3, %arg4] : memref<65x65x65xf64>
  }
  return
}

// -----

// CHECK-LABEL: @predicate_execution_with_step
func @predicate_execution_with_step(%arg0: memref<65x65x65xf64>, %arg1: memref<65x65x65xf64>) {
  // CHECK-DAG: [[C0:%.*]] = constant 0 : index
  // CHECK-DAG: [[C1:%.*]] = constant 1 : index
  // CHECK-DAG: [[C3:%.*]] = constant 3 : index
  // CHECK-DAG: [[C6:%.*]] = constant 6 : index
  // CHECK-DAG: [[C64:%.*]] = constant 64 : index
  // CHECK-DAG: [[C65:%.*]] = constant 65 : index
  // CHECK-DAG: [[C66:%.*]] = constant 66 : index
  // CHECK-DAG: [[C128:%.*]] = constant 128 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c65 = constant 65 : index
  // CHECK: scf.parallel ([[ARG0_1:%.*]], [[ARG1_1:%.*]], [[ARG3:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C128]], [[C66]], [[C65]]) step ([[C64]], [[C6]], [[C1]]) {
  // CHECK-NEXT: scf.parallel ([[ARG0_2:%.*]], [[ARG1_2:%.*]], {{%.*}}) = ([[C0]], [[C0]], [[C0]]) to ([[C64]], [[C6]], [[C1]]) step ([[C1]], [[C3]], [[C1]]) {
  scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c65, %c65, %c65) step (%c1, %c3, %c1) {
    // CHECK-NEXT: [[IDX0:%.*]] = affine.apply #map{{[0-9]+}}([[ARG0_1]], [[ARG0_2]]) 
    // CHECK-NEXT: [[IDX1:%.*]] = affine.apply #map{{[0-9]+}}([[ARG1_1]], [[ARG1_2]]) 
    // CHECK-NEXT: [[CMP0:%.*]] = cmpi "slt", [[IDX0]], [[C65]] : index
    // CHECK-NEXT: [[CMP1:%.*]] = cmpi "slt", [[IDX1]], [[C65]] : index
    // CHECK-NEXT: [[PRED:%.*]] = and [[CMP0]], [[CMP1]] : i1
    // CHECK-NEXT: scf.if [[PRED:%.*]] {
    // CHECK-NEXT: {{%.*}} = load {{%.*}}{{\[}}[[IDX0]], [[IDX1]], [[ARG3]]] 
    %0 = load %arg0[%arg2, %arg3, %arg4] : memref<65x65x65xf64>
    store %0, %arg1[%arg2, %arg3, %arg4] : memref<65x65x65xf64>
  }
  return
}