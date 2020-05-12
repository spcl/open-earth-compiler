// RUN: oec-opt %s -split-input-file --stencil-shape-shift | oec-opt | FileCheck %s

// CHECK-LABEL: func @simple(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECK: stencil.assert %{{.*}}([0, 0, 0] : [70, 70, 60]) : !stencil.field<?x?x?xf64>
  // CHECK: stencil.assert %{{.*}}([1, 1, 0] : [70, 70, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-2, -2, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  // CHECK: %{{.*}} = stencil.load %{{.*}}([2, 1, 0] : [68, 69, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<66x68x60xf64>
  %0 = stencil.load %arg0([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<66x68x60xf64>
  %1 = stencil.apply (%arg2 = %0 : !stencil.temp<66x68x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %2 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<66x68x60xf64>) -> f64
    %3 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<66x68x60xf64>) -> f64
    %4 = addf %2, %3 : f64
    stencil.return %4 : f64
  // CHECK: } to ([3, 3, 0] : [67, 67, 60])
  } to ([0, 0, 0] : [64, 64, 60])
  // CHECK: stencil.store %{{.*}} to %{{.*}}([3, 3, 0] : [67, 67, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<?x?x?xf64>
  stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<?x?x?xf64>
  return
}