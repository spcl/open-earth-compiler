// RUN: oec-opt %s --stencil-shape-inference | oec-opt | FileCheck %s

// CHECK-LABEL: func @lap_stencil(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}}([-2, -2, 0] : [66, 66, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>

//       CHECK: } to ([-1, -1, 0]:[65, 65, 60])
//       CHECK: } to ([0, 0, 0]:[64, 64, 60])
func @lap_stencil(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, 0, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %1 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.access %arg2 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = stencil.access %arg2 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %8 = addf %3, %4 : f64
    %9 = addf %5, %6 : f64
    %10 = addf %8, %9 : f64
    %cst = constant -4.000000e+00 : f64
    %11 = mulf %7, %cst : f64
    %12 = addf %11, %10 : f64
    stencil.return %12 : f64
  }
  %2 = stencil.apply (%arg2 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.access %arg2 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = stencil.access %arg2 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %8 = addf %3, %4 : f64
    %9 = addf %5, %6 : f64
    %10 = addf %8, %9 : f64
    %cst = constant -4.000000e+00 : f64
    %11 = mulf %7, %cst : f64
    %12 = addf %11, %10 : f64
    stencil.return %12 : f64
  }
  stencil.store %2 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}




