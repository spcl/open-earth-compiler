// RUN: oec-opt %s -split-input-file --canonicalize | oec-opt | FileCheck %s

// CHECK-LABEL: func @apply(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
// CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<?x?x?xf64>) ->
// CHECK: stencil.return %{{.*}} : f64
func @apply(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg2 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<?x?x?xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %1,%2 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %0 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
    %3 = stencil.access %arg3[-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg4[0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5, %5  : f64, f64
  }
  stencil.store %1 to %arg1 ([0, 0, 0]:[64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  stencil.store %2 to %arg2 ([0, 0, 0]:[64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @hoist(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
func @hoist(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECK: stencil.assert
  // CHECK: stencil.assert
  // CHECK: stencil.assert
  // CHECK: stencil.load
  // CHECK: stencil.apply
  // CHECK: stencil.apply
  // CHECK: stencil.store
  %1 = stencil.apply -> !stencil.temp<?x?x?xf64> {
    %2 = constant 1.0 : f64
    stencil.return %2 : f64
  }
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  %3 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %4 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %5 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    stencil.return %5 : f64
  }
  stencil.assert %arg2([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.store %4 to %arg2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}


