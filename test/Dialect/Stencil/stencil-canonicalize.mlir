// RUN: oec-opt %s -split-input-file --canonicalize | oec-opt | FileCheck %s

// CHECK-LABEL: func @apply(%{{.*}}: !stencil.temp<?x?x?xf64>, %{{.*}}: !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) attributes {stencil.program}
// CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<?x?x?xf64>) ->
// CHECK: stencil.return %{{.*}} : f64
func @apply(%arg0: !stencil.temp<?x?x?xf64>, %arg1: !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) attributes {stencil.program} {
  %4,%5 = stencil.apply (%arg2 = %arg0 : !stencil.temp<?x?x?xf64>, %arg3 = %arg0 : !stencil.temp<?x?x?xf64>, %arg4 = %arg1 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
    %6 = stencil.access %arg3[-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.access %arg2[0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %8 = addf %6, %7 : f64
    stencil.return %8, %8  : f64, f64
  }
  return %4, %5 : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
}

// -----

func @apply(%arg0: !stencil.temp<?x?x?xf64>, %arg1: !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) attributes {stencil.program} {
  %4,%5 = stencil.apply (%arg2 = %arg0 : !stencil.temp<?x?x?xf64>, %arg3 = %arg0 : !stencil.temp<?x?x?xf64>, %arg4 = %arg1 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
    %6 = constant 1.0 : f64
    stencil.return %6, %6  : f64, f64
  }
  return %4, %5 : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
}

// -----

// CHECK-LABEL: func @hoist(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
func @hoist(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECK: stencil.cast
  // CHECK: stencil.cast
  // CHECK: stencil.cast
  // CHECK: stencil.load
  // CHECK: stencil.apply
  // CHECK: stencil.apply
  // CHECK: stencil.store
  %1 = stencil.apply -> !stencil.temp<?x?x?xf64> {
    %2 = constant 1.0 : f64
    stencil.return %2 : f64
  }
  %3 = stencil.cast %arg0 ([-3, -3, 0]:[67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %4 = stencil.cast %arg1 ([-3, -3, 0]:[67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  stencil.store %1 to %4([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  %5 = stencil.load %3 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  %6 = stencil.apply (%arg3 = %5 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    stencil.return %7 : f64
  }
  %8 = stencil.cast %arg2 ([-3, -3, 0]:[67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  stencil.store %6 to %8([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}


