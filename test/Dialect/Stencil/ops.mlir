// RUN: oec-opt %s -split-input-file | oec-opt | FileCheck %s

// CHECK-LABEL: func @access(%{{.*}}: !stencil.temp<ijk,f64>, %{{.*}}: !stencil.temp<ij,f32>) {
func @access(%in1 : !stencil.temp<ijk,f64>, %in2 : !stencil.temp<ij,f32>) {
  //  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[-1, 2, -3] : (!stencil.temp<ijk,f64>) -> f64
  %0 = "stencil.access"(%in1) {offset = [-1, 2, -3]} : (!stencil.temp<ijk,f64>) -> f64
  //  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[3, -2, 1] : (!stencil.temp<ij,f32>) -> f32
  %1 = "stencil.access"(%in2) {offset = [3, -2, 1]} : (!stencil.temp<ij,f32>) -> f32
  return
}

// -----

// CHECK-LABEL: func @return(%{{.*}}: f64, %{{.*}}: !stencil.field<ijk,f64>) 
func @return(%in : f64, %out : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
  // CHECK %{{.*}} = stencil.apply (%{{.*}}: f64) -> !stencil.temp<ijk,f64>
  %0 = "stencil.apply"(%in) ({
    ^bb0(%1 : f64):
    //  CHECK: stencil.return %{{.*}} : f64
    "stencil.return"(%1) : (f64) -> ()
  }) : (f64) -> !stencil.temp<ijk,f64>
  return
}

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
  %7 = "stencil.load"(%out) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<ijk,f64>) -> (!stencil.temp<ijk,f64>)
  
  %0, %1 = "stencil.apply"(%in_0, %in_1) ({
    ^bb0(%2 : f64, %3 : f32):
    //  CHECK: stencil.return unroll [1, 2, 1] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f32, f32
    "stencil.return"(%2, %2, %3, %3) {unroll = [1, 2, 1]} : (f64, f64, f32, f32) -> ()
  }) : (f64, f32) -> (!stencil.temp<ijk,f64>, !stencil.temp<ijk,f32>)
  return
}
