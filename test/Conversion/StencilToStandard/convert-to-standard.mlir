// RUN: oec-opt %s -split-input-file --convert-stencil-to-standard | FileCheck %s

// CHECK-LABEL: @func_lowering
// CHECK: (%{{.*}}: memref<7x77x777xf64, #map{{[0-9]+}}>) {
func @func_lowering(%arg0: !stencil.field<ijk,f64>) attributes {stencil.program} {
  stencil.assert %arg0 ([0, 0, 0]:[7, 77, 777]) : !stencil.field<ijk,f64>
  return
}

// -----

// CHECK-LABEL: @parallel_loop
func @parallel_loop(%arg0 : f64) attributes {stencil.program} {
  // CHECK: [[C0:%.*]] = constant 0 : index
  // CHECK-DAG: [[C1:%.*]] = constant 1 : index
  // CHECK-DAG: [[C7:%.*]] = constant 7 : index
  // CHECK-DAG: [[C77:%.*]] = constant 77 : index
  // CHECK-DAG: [[C777:%.*]] = constant 777 : index
  // CHECK-NEXT: loop.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C7]], [[C77]], [[C777]]) step ([[C1]], [[C1]], [[C1]]) {  
  %1 = stencil.apply %arg1 = %arg0  : f64 {
    // CHECK: store %{{.*}}, %{{.*}}{{\[}}[[ARG0]], [[ARG1]], [[ARG2]]] 
    stencil.return %arg1 : f64
  } to ([0, 0, 0]:[7, 77, 777]) : !stencil.view<ijk,f64>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @parallel_loop_unroll
func @parallel_loop_unroll(%arg0 : f64) attributes {stencil.program} {
  // CHECK: [[C0:%.*]] = constant 0 : index
  // CHECK-DAG: [[C1:%.*]] = constant 1 : index
  // CHECK-DAG: [[C2:%.*]] = constant 2 : index
  // CHECK-DAG: [[C7:%.*]] = constant 7 : index
  // CHECK-DAG: [[C77:%.*]] = constant 77 : index
  // CHECK-DAG: [[C777:%.*]] = constant 777 : index
  // CHECK-NEXT: loop.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C7]], [[C77]], [[C777]]) step ([[C1]], [[C2]], [[C1]]) {  
  %1 = stencil.apply %arg1 = %arg0  : f64 {
    // CHECK-NEXT: store %{{.*}}, %{{.*}}{{\[}}[[ARG0]], [[ARG1]], [[ARG2]]] 
    // CHECK-NEXT: [[OFF:%.*]] = constant 1 : index
    // CHECK-NEXT: [[IDX:%.*]] = affine.apply [[MAP0]]([[ARG1]], [[OFF]]) 
    // CHECK-NEXT: store %{{.*}}, %{{.*}}{{\[}}[[ARG0]], [[IDX]], [[ARG2]]] 
    stencil.return unroll [1, 2, 1] %arg1, %arg1 : f64, f64
  } to ([0, 0, 0]:[7, 77, 777]) : !stencil.view<ijk,f64>
  return
}

// -----

// CHECK-LABEL: @alloc_temp
func @alloc_temp(%arg0 : f64) attributes {stencil.program} {
  // CHECK: [[TEMP:%.*]] = alloc() : memref<7x7x7xf64, #map{{[0-9]+}}>
  %1 = stencil.apply %arg1 = %arg0 : f64 {
    // CHECK: store %{{.*}}, [[TEMP]]  
    stencil.return %arg1 : f64
  } to ([0, 0, 0]:[7, 7, 7]) : !stencil.view<ijk,f64>
  %3 = stencil.apply %arg2 = %1 : !stencil.view<ijk,f64> {
    // CHECK: load [[TEMP]]  
    %4 = stencil.access %arg2[0,0,0] : (!stencil.view<ijk,f64>) -> f64
    stencil.return %4 : f64
  } to ([0, 0, 0]:[7, 7, 7]) : !stencil.view<ijk,f64>
  // CHECK: dealloc [[TEMP]] : memref<7x7x7xf64, #map{{[0-9]+}}>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @access_lowering
func @access_lowering(%arg0: !stencil.field<ijk,f64>) attributes {stencil.program} {
  // CHECK: [[VIEW:%.*]] = std.subview 
  stencil.assert %arg0 ([0, 0, 0]:[10, 10, 10]) : !stencil.field<ijk,f64>
  %0 = stencil.load %arg0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  // CHECK: loop.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %1 = stencil.apply %arg1 = %0  : !stencil.view<ijk,f64> {
    // CHECK-DAG: [[C0:%.*]] = constant 0 : index
    // CHECK-DAG: [[O0:%.*]] = affine.apply [[MAP0]]([[ARG0]], [[C0]])
    // CHECK-DAG: [[C1:%.*]] = constant 1 : index
    // CHECK-DAG: [[O1:%.*]] = affine.apply [[MAP0]]([[ARG1]], [[C1]])
    // CHECK-DAG: [[C2:%.*]] = constant 2 : index
    // CHECK-DAG: [[O2:%.*]] = affine.apply [[MAP0]]([[ARG2]], [[C2]])
    // CHECK-NEXT: %{{.*}} = load [[VIEW:%.*]]{{\[}}[[O0]], [[O1]], [[O2]]{{[]]}}
    %2 = stencil.access %arg1[0, 1, 2] : (!stencil.view<ijk,f64>) -> f64
    stencil.return %2 : f64
  } to ([0, 0, 0]:[7, 7, 7]) : !stencil.view<ijk,f64>
  return
}

// -----

// CHECK-LABEL: @return_lowering
func @return_lowering(%arg0: f64) attributes {stencil.program} {
  // CHECK: loop.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %1,%2 = stencil.apply %arg1 = %arg0 : f64 {
    // CHECK-COUNT-2: store %{{.*}}, %{{.*}}{{\[}}[[ARG0]], [[ARG1]], [[ARG2]]] : memref<7x7x7xf64, #map{{[0-9]+}}> 
    stencil.return %arg1, %arg1 : f64, f64
  } to ([0, 0, 0]:[7, 7, 7]) : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>
  return
}

// -----

// CHECK-LABEL: @load_lowering
func @load_lowering(%arg0: !stencil.field<ijk,f64>) attributes {stencil.program} {
  // CHECK: %{{.*}} = std.subview %{{.*}}[][][] : memref<11x12x13xf64, #map{{[0-9]+}}> to memref<9x9x9xf64, #map{{[0-9]+}}>
  stencil.assert %arg0 ([0, 0, 0]:[11, 12, 13]) : !stencil.field<ijk,f64>
  %0 = stencil.load %arg0 ([1, 2, 3]:[10, 11, 12]) : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  return
}

// -----

// CHECK-LABEL: @store_lowering
func @store_lowering(%arg0: !stencil.field<ijk,f64>) attributes {stencil.program} {
  // CHECK: [[VIEW:%.*]] = std.subview %{{.*}}[][][] : memref<10x10x10xf64, #map{{[0-9]+}}> to memref<7x7x7xf64, #map{{[0-9]+}}>
  stencil.assert %arg0 ([0, 0, 0]:[10, 10, 10]) : !stencil.field<ijk,f64>
  %c1 = constant 1.0 : f64
  %1 = stencil.apply %arg1 = %c1 : f64 {
    // CHECK: store %{{.*}} [[VIEW]]
    stencil.return %arg1 : f64
  } to ([0, 0, 0]:[7, 7, 7]) : !stencil.view<ijk,f64>
  stencil.store %1 to %arg0 ([1, 2, 3]:[8, 9, 10]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
