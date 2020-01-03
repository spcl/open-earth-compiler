// RUN: oec-opt %s --convert-stencil-to-standard | oec-opt | FileCheck %s

module {
  func @lap_stencil(%arg0: !stencil.field<ijk,f64>, %arg1: !stencil.field<ijk,f64>) attributes {stencil.program} {
    stencil.assert %arg0 ([0, 0, 0]:[70, 70, 60]) : !stencil.field<ijk,f64>
    stencil.assert %arg1 ([0, 0, 0]:[70, 70, 60]) : !stencil.field<ijk,f64>
    %0 = stencil.load %arg0 ([1, 1, 0]:[69, 69, 60]) : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
    %1 = stencil.apply %arg2 = %0 : !stencil.view<ijk,f64> {
      %3 = stencil.access %arg2[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg2[2, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = stencil.access %arg2[1, 2, 0] : (!stencil.view<ijk,f64>) -> f64
      %6 = stencil.access %arg2[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %7 = stencil.access %arg2[1, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %8 = addf %3, %4 : f64
      %9 = addf %5, %6 : f64
      %10 = addf %8, %9 : f64
      %cst = constant -4.000000e+00 : f64
      %11 = mulf %7, %cst : f64
      %12 = addf %11, %10 : f64
      stencil.return %12 : f64
    } to ([0, 0, 0]:[66, 66, 60]) : !stencil.view<ijk,f64>
    %2 = stencil.apply %arg2 = %1 : !stencil.view<ijk,f64> {
      %3 = stencil.access %arg2[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg2[2, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = stencil.access %arg2[1, 2, 0] : (!stencil.view<ijk,f64>) -> f64
      %6 = stencil.access %arg2[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %7 = stencil.access %arg2[1, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %8 = addf %3, %4 : f64
      %9 = addf %5, %6 : f64
      %10 = addf %8, %9 : f64
      %cst = constant -4.000000e+00 : f64
      %11 = mulf %7, %cst : f64
      %12 = addf %11, %10 : f64
      stencil.return %12 : f64
    } to ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64>
    stencil.store %2 to %arg1 ([3, 3, 0]:[67, 67, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
    return
  }
}

// CHECK-LABEL: func @lap_stencil(%{{.*}}: memref<70x70x60xf64, #{{.*}}>, %{{.*}}: memref<70x70x60xf64, #{{.*}}>) {
//  CHECK-NEXT: %{{.*}} = std.subview %{{.*}}[][][] : memref<70x70x60xf64, #{{.*}}> to memref<68x68x60xf64, #{{.*}}>
//  CHECK-NEXT: %{{.*}} = alloc() : memref<66x66x60xf64, #{{.*}}>
//  CHECK-NEXT: affine.for %{{.*}} = 0 to 60 {
//  CHECK-NEXT: affine.for %{{.*}} = 0 to 66 {
//  CHECK-NEXT: affine.for %{{.*}} = 0 to 66 {


//       CHECK: affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] 
//       CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] 
