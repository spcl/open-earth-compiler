// RUN: oec-opt %s -split-input-file --convert-stencil-to-std | FileCheck %s

// CHECK-LABEL: @func_lowering
// CHECK: (%{{.*}}: memref<?x?x?xf64>) {
func @func_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECK: %{{.*}} = memref_cast %{{.*}} : memref<?x?x?xf64> to memref<777x77x7xf64>
  stencil.assert %arg0 ([0, 0, 0]:[7, 77, 777]) : !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @parallel_loop
func @parallel_loop(%arg0 : f64) attributes {stencil.program} {
  // CHECK-DAG: [[C0_1:%.*]] = constant 0 : index
  // CHECK-DAG: [[C0_2:%.*]] = constant 0 : index
  // CHECK-DAG: [[C0_3:%.*]] = constant 0 : index
  // CHECK-DAG: [[C1_1:%.*]] = constant 1 : index
  // CHECK-DAG: [[C1_2:%.*]] = constant 1 : index
  // CHECK-DAG: [[C1_3:%.*]] = constant 1 : index
  // CHECK-DAG: [[C7:%.*]] = constant 7 : index
  // CHECK-DAG: [[C77:%.*]] = constant 77 : index
  // CHECK-DAG: [[C777:%.*]] = constant 777 : index
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) = ([[C0_1]], [[C0_2]], [[C0_3]]) to ([[C7]], [[C77]], [[C777]]) step ([[C1_1]], [[C1_2]], [[C1_3]]) {  
  %0 = stencil.apply (%arg1 = %arg0 : f64) -> !stencil.temp<7x77x777xf64> {
    // CHECK-DAG:  [[IDX2:%.*]] = affine.apply [[MAP0]]([[ARG0]], %{{.*}})
    // CHECK-DAG:  [[IDX1:%.*]] = affine.apply [[MAP0]]([[ARG1]], %{{.*}})
    // CHECK-DAG:  [[IDX0:%.*]] = affine.apply [[MAP0]]([[ARG2]], %{{.*}})
    // CHECK: store %{{.*}}, %{{.*}}{{\[}}[[IDX0]], [[IDX1]], [[IDX2]]] 
    stencil.return %arg1 : f64
  } to ([0, 0, 0]:[7, 77, 777])
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @parallel_loop_unroll
func @parallel_loop_unroll(%arg0 : f64) attributes {stencil.program} {
  // CHECK-DAG: [[C0_1:%.*]] = constant 0 : index
  // CHECK-DAG: [[C0_2:%.*]] = constant 0 : index
  // CHECK-DAG: [[CM1:%.*]] = constant -1 : index
  // CHECK-DAG: [[C1_1:%.*]] = constant 1 : index
  // CHECK-DAG: [[C1_2:%.*]] = constant 1 : index
  // CHECK-DAG: [[C2:%.*]] = constant 2 : index
  // CHECK-DAG: [[C7:%.*]] = constant 7 : index
  // CHECK-DAG: [[C77:%.*]] = constant 77 : index
  // CHECK-DAG: [[C777:%.*]] = constant 777 : index
  // CHECK-NEXT: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) = ([[C0_1]], [[CM1]], [[C0_2]]) to ([[C7]], [[C77]], [[C777]]) step ([[C1_1]], [[C2]], [[C1_2]]) {  
  %0 = stencil.apply (%arg1 = %arg0 : f64) -> !stencil.temp<777x78x7xf64> {
    // CHECK-DAG:  [[U0O1:%.*]] = constant 1 : index
    // CHECK-DAG:  [[U0IDX1:%.*]] = affine.apply [[MAP0]]([[ARG1]], [[U0O1]])
    // CHECK: store %{{.*}}, %{{.*}}{{\[}}%{{.*}}, [[U0IDX1]], %{{.*}}]
    // CHECK-DAG:  [[U1O1:%.*]] = constant 2 : index
    // CHECK-DAG:  [[U1IDX1:%.*]] = affine.apply [[MAP0]]([[ARG1]], [[U1O1]])
    // CHECK: store %{{.*}}, %{{.*}}{{\[}}%{{.*}}, [[U1IDX1]], %{{.*}}]
    stencil.return unroll [1, 2, 1] %arg1, %arg1 : f64, f64
  } to ([0, -1, 0]:[7, 77, 777])
  return
}