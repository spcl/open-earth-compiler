// RUN: oec-opt %s | oec-opt | FileCheck %s

func @lap(%in : !stencil.temp<ijk,f64>) -> f64
  attributes { stencil.function } {
  %0 = "stencil.access"(%in) {offset = [-1, 0, 0]} : (!stencil.temp<ijk,f64>) -> f64
  %1 = "stencil.access"(%in) {offset = [ 1, 0, 0]} : (!stencil.temp<ijk,f64>) -> f64
  %2 = "stencil.access"(%in) {offset = [ 0, 1, 0]} : (!stencil.temp<ijk,f64>) -> f64
  %3 = "stencil.access"(%in) {offset = [ 0,-1, 0]} : (!stencil.temp<ijk,f64>) -> f64
  %4 = "stencil.access"(%in) {offset = [ 0, 0, 0]} : (!stencil.temp<ijk,f64>) -> f64
  %5 = addf %0, %1 : f64
  %6 = addf %2, %3 : f64
  %7 = addf %5, %6 : f64
  %8 = constant -4.0 : f64
  %9 = mulf %4, %8 : f64
  %10 = addf %9, %7 : f64
  return %10 : f64
}

// CHECK-LABEL: func @lap(%{{.*}}: !stencil.temp<ijk,f64>) -> f64 attributes {stencil.function} {
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

// -----

func @unroll(%in : f64, %out : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
  %0 = "stencil.apply"(%in) ({
    ^bb0(%1 : f64):
    //  CHECK: stencil.return unroll [1, 2, 1] %{{.*}}, %{{.*}} : f64, f64
    "stencil.return"(%1, %1) {unroll = [1, 2, 1]} : (f64, f64) -> ()
  }) : (f64) -> !stencil.temp<ijk,f64>
  return
}

// -----

func @unroll_2(%in_0 : f64, %in_1 : f32, %out : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
  "stencil.assert"(%out) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<ijk,f64>) -> ()
  %0, %1 = "stencil.apply"(%in_0, %in_1) ({
    ^bb0(%2 : f64, %3 : f32):
    //  CHECK: stencil.return unroll [1, 2, 1] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f32, f32
    "stencil.return"(%2, %2, %3, %3) {unroll = [1, 2, 1]} : (f64, f64, f32, f32) -> ()
  }) : (f64, f32) -> (!stencil.temp<ijk,f64>, !stencil.temp<ijk,f32>)
  return
}
