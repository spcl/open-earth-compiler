// RUN: oec-opt %s -split-input-file | oec-opt | FileCheck %s

// CHECK-LABEL: func @access(%{{.*}}: !stencil.temp<1x2x3xf64>, %{{.*}}: !stencil.temp<1x2x0xf32>) {
func @access(%in1 : !stencil.temp<1x2x3xf64>, %in2 : !stencil.temp<1x2x0xf32>) {
  //  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[-1, 2, -3] : (!stencil.temp<1x2x3xf64>) -> f64
  %0 = "stencil.access"(%in1) {offset = [-1, 2, -3]} : (!stencil.temp<1x2x3xf64>) -> f64
  //  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[3, -2, 1] : (!stencil.temp<1x2x0xf32>) -> f32
  %1 = "stencil.access"(%in2) {offset = [3, -2, 1]} : (!stencil.temp<1x2x0xf32>) -> f32
  return
}

// -----

// CHECK-LABEL: func @access() {
func @access() {
  //  CHECK-NEXT: %{{.*}} = stencil.index 2 [3, -2, 1] : index
  %0 = "stencil.index"() {offset = [3, -2, 1], dim = 2} : () -> (index)
  return
}

// -----

// CHECK-LABEL: func @assert(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) {
func @assert(%in : !stencil.field<?x?x?xf64>, %out : !stencil.field<?x?x?xf64>) {
  //  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  "stencil.assert"(%in) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> ()
  //  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  "stencil.assert"(%out) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func @load(%{{.*}}: !stencil.field<?x?x?xf64>) {
func @load(%in1 : !stencil.field<?x?x?xf64>, %in2 : !stencil.field<?x?x?xf64>) {
  //  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %0 = "stencil.load"(%in1)  : (!stencil.field<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>)
  //  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}}([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<70x70x60xf64>
  %1 = "stencil.load"(%in2) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> (!stencil.temp<70x70x60xf64>)
  "stencil.assert"(%in1) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> ()
  "stencil.assert"(%in2) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> ()  
  return
}

// -----

// CHECK-LABEL: func @store(%{{.*}}: !stencil.field<?x?x?xf64>) {
func @store(%out : !stencil.field<?x?x?xf64>) {
  // CHECK %{{.*}} = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %0 = "stencil.apply"() ({
    //  CHECK: %{{.*}} = constant 
    %1 = constant 1.0 : f64
    //  CHECK: stencil.return %{{.*}} : f64
    //  CHECK: }
    "stencil.return"(%1) : (f64) -> ()
  }) : () -> !stencil.temp<?x?x?xf64>
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  "stencil.store"(%0, %out)  {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.temp<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> () 
  "stencil.assert"(%out) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> ()  
  return
}

// -----

// CHECK-LABEL: func @return(%{{.*}}: f64, %{{.*}}: !stencil.field<?x?x?xf64>) 
func @return(%in : f64, %out : !stencil.field<?x?x?xf64>)
  attributes { stencil.program } {
  // CHECK %{{.*}} = stencil.apply (%{{.*}}: f64) -> !stencil.temp<1x2x3xf64>
  %0 = "stencil.apply"(%in) ({
    ^bb0(%1 : f64):
    //  CHECK: stencil.return %{{.*}} : f64
    "stencil.return"(%1) : (f64) -> ()
  }) : (f64) -> !stencil.temp<1x2x3xf64>
  return
}

// -----

func @unroll(%in : f64, %out : !stencil.field<?x?x?xf64>)
  attributes { stencil.program } {
  %0 = "stencil.apply"(%in) ({
    ^bb0(%1 : f64):
    //  CHECK: stencil.return unroll [1, 2, 1] %{{.*}}, %{{.*}} : f64, f64
    "stencil.return"(%1, %1) {unroll = [1, 2, 1]} : (f64, f64) -> ()
  }) : (f64) -> !stencil.temp<?x?x?xf64>
  return
}

// -----

func @unroll_2(%in_0 : f64, %in_1 : f32, %out : !stencil.field<?x?x?xf64>)
  attributes { stencil.program } {
  "stencil.assert"(%out) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> ()
  %7 = "stencil.load"(%out) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> (!stencil.temp<1x2x3xf64>)  
  %0, %1 = "stencil.apply"(%in_0, %in_1) ({
    ^bb0(%2 : f64, %3 : f32):
    //  CHECK: stencil.return unroll [1, 2, 1] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f32, f32
    "stencil.return"(%2, %2, %3, %3) {unroll = [1, 2, 1]} : (f64, f64, f32, f32) -> ()
  }) : (f64, f32) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf32>)
  return
}
