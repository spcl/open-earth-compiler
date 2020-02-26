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
  // CHECK-NEXT: [[C1:%.*]] = constant 1 : index
  // CHECK-NEXT: [[C7:%.*]] = constant 7 : index
  // CHECK-NEXT: [[C77:%.*]] = constant 77 : index
  // CHECK-NEXT: [[C777:%.*]] = constant 777 : index
  // CHECK-NEXT: loop.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C7]], [[C77]], [[C777]]) step ([[C1]], [[C1]], [[C1]]) {  
  %1 = stencil.apply %arg1 = %arg0  : f64 {
    // CHECK: store %{{.*}}, %{{.*}}{{[[]}}[[ARG0]], [[ARG1]], [[ARG2]]{{[]]}} 
    stencil.return %arg1 : f64
    // CHECK-COUNT-1: {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 3 : i64}
    // CHECK-COUNT-1: {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 0 : i64}
    // CHECK-COUNT-1: {bound = affine_map<(d0) -> (d0)>, map = affine_map<(d0) -> (d0)>, processor = 1 : i64}
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

// CHECK-LABEL: @access_lowering
func @access_lowering(%arg0: !stencil.field<ijk,f64>) attributes {stencil.program} {
  // CHECK: [[VIEW:%.*]] = std.subview 
  stencil.assert %arg0 ([0, 0, 0]:[10, 10, 10]) : !stencil.field<ijk,f64>
  %0 = stencil.load %arg0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  // CHECK: loop.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %1 = stencil.apply %arg1 = %0  : !stencil.view<ijk,f64> {
    // CHECK-NEXT: [[C0:%.*]] = constant 0 : index
    // CHECK-NEXT: [[O0:%.*]] = affine.apply #map{{[0-9]+}}([[ARG0]], [[C0]])
    // CHECK-NEXT: [[C1:%.*]] = constant 1 : index
    // CHECK-NEXT: [[O1:%.*]] = affine.apply #map{{[0-9]+}}([[ARG1]], [[C1]])
    // CHECK-NEXT: [[C2:%.*]] = constant 2 : index
    // CHECK-NEXT: [[O2:%.*]] = affine.apply #map{{[0-9]+}}([[ARG2]], [[C2]])
    // CHECK-NEXT: %{{.*}} = load [[VIEW:%.*]]{{[[]}}[[O0]], [[O1]], [[O2]]{{[]]}}
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
    // CHECK-COUNT-2: store %{{.*}}, %{{.*}}{{[[]}}[[ARG0]], [[ARG1]], [[ARG2]]{{[]]}} : memref<7x7x7xf64, #map{{[0-9]+}}> 
    stencil.return %arg1, %arg1 : f64, f64
  } to ([0, 0, 0]:[7, 7, 7]) : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>
  return
}

// -----

// CHECK-LABEL: @subview_lowering
func @subview_lowering(%arg0: !stencil.field<ijk,f64>) attributes {stencil.program} {
  // CHECK: %{{.*}} = std.subview %{{.*}}[][][] : memref<11x12x13xf64, #map{{[0-9]+}}> to memref<9x9x9xf64, #map{{[0-9]+}}>
  stencil.assert %arg0 ([0, 0, 0]:[11, 12, 13]) : !stencil.field<ijk,f64>
  %0 = stencil.load %arg0 ([1, 2, 3]:[10, 11, 12]) : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  return
}