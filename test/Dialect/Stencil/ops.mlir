// RUN: oec-opt %s -split-input-file | oec-opt | FileCheck %s

// CHECK-LABEL: func @access(%{{.*}}: !stencil.temp<10x20x30xf64>, %{{.*}}: !stencil.temp<10x20x0xf32>) {
func @access(%in1 : !stencil.temp<10x20x30xf64>, %in2 : !stencil.temp<10x20x0xf32>) {
  //  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[-1, 2, -3] : (!stencil.temp<10x20x30xf64>) -> f64
  %0 = "stencil.access"(%in1) {offset = [-1, 2, -3]} : (!stencil.temp<10x20x30xf64>) -> f64
  //  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[3, -2, 1] : (!stencil.temp<10x20x0xf32>) -> f32
  %1 = "stencil.access"(%in2) {offset = [3, -2, 1]} : (!stencil.temp<10x20x0xf32>) -> f32
  return
}

// -----

// CHECK-LABEL: func @dyn_access(%{{.*}}: !stencil.temp<10x20x30xf64>, %{{.*}}: !stencil.temp<10x20x0xf32>, %{{.*}}: index) {
func @dyn_access(%in1 : !stencil.temp<10x20x30xf64>, %in2 : !stencil.temp<10x20x0xf32>, %idx : index) {
  //  CHECK-NEXT: %{{.*}} = stencil.dyn_access %{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) in [-3, -3, 0] : [3, 3, 0] : (!stencil.temp<10x20x30xf64>) -> f64
  %0 = "stencil.dyn_access"(%in1, %idx, %idx, %idx) {lb=[-3,-3,0], ub=[3,3,0]} : (!stencil.temp<10x20x30xf64>, index, index, index) -> f64
  //  CHECK-NEXT: %{{.*}} = stencil.dyn_access %{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) in [-3, -3, 0] : [3, 3, 0] : (!stencil.temp<10x20x0xf32>) -> f32
  %1 = "stencil.dyn_access"(%in2, %idx, %idx, %idx) {lb=[-3,-3,0], ub=[3,3,0]} : (!stencil.temp<10x20x0xf32>, index, index, index) -> f32
  return
}

// -----

// CHECK-LABEL: func @index() {
func @index() {
  //  CHECK-NEXT: %{{.*}} = stencil.index 2 [3, -2, 1] : index
  %0 = "stencil.index"() {offset = [3, -2, 1], dim = 2} : () -> (index)
  return
}

// -----

// CHECK-LABEL: func @cast(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) {
func @cast(%in : !stencil.field<?x?x?xf64>, %out : !stencil.field<?x?x?xf64>) {
  //  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  "stencil.cast"(%in) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> (!stencil.field<70x70x60xf64>)
  //  CHECK-NEXT: stencil.cast %{{.*}}([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  "stencil.cast"(%out) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> (!stencil.field<70x70x60xf64>)
  return
}

// -----

// CHECK-LABEL: func @load(%{{.*}}: !stencil.field<?x?x?xf64>) {
func @load(%in1 : !stencil.field<?x?x?xf64>, %in2 : !stencil.field<?x?x?xf64>) {
  //  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %0 = "stencil.cast"(%in1) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> (!stencil.field<70x70x60xf64>)
  //  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = "stencil.cast"(%in2) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> (!stencil.field<70x70x60xf64>)  
  //  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  %2 = "stencil.load"(%0)  : (!stencil.field<70x70x60xf64>) -> (!stencil.temp<?x?x?xf64>)
  //  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}}([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<70x70x60xf64>
  %3 = "stencil.load"(%1) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<70x70x60xf64>) -> (!stencil.temp<70x70x60xf64>)
  return
}

// -----

// CHECK-LABEL: func @buffer() {
func @buffer() {
  %0 = "stencil.apply"() ({
    %1 = constant 1.0 : f64
    %2 = "stencil.store_result"(%1) : (f64) -> !stencil.result<f64>
    "stencil.return"(%2) : (!stencil.result<f64>) -> ()
  }) : () -> !stencil.temp<?x?x?xf64>
  // CHECK: %{{.*}} = stencil.buffer %{{.*}} : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %1 = "stencil.buffer"(%0): (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @store(%{{.*}}: !stencil.field<?x?x?xf64>) {
func @store(%out : !stencil.field<?x?x?xf64>) {
  %0 = "stencil.cast"(%out) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> (!stencil.field<70x70x60xf64>) 
  %1 = "stencil.apply"() ({
    %1 = constant 1.0 : f64
    %2 = "stencil.store_result"(%1) : (f64) -> !stencil.result<f64>
    "stencil.return"(%2) : (!stencil.result<f64>) -> ()
  }) : () -> !stencil.temp<?x?x?xf64>
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  "stencil.store"(%1, %0)  {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.temp<?x?x?xf64>, !stencil.field<70x70x60xf64>) -> () 
  return
}

// -----

// CHECK-LABEL: func @return(%{{.*}}: f64) 
func @return(%in : f64)
  attributes { stencil.program } {
  // CHECK %{{.*}} = stencil.apply (%{{.*}}: f64) -> !stencil.temp<?x?x?xf64>
  %0 = "stencil.apply"(%in) ({
    ^bb0(%1 : f64):
    //  CHECK: stencil.store_result %{{.*}} : (f64) -> !stencil.result<f64>
    %2 = "stencil.store_result"(%1) : (f64) -> !stencil.result<f64>
    //  CHECK: stencil.return %{{.*}} : !stencil.result<f64>
    "stencil.return"(%2) : (!stencil.result<f64>) -> ()
  }) : (f64) -> !stencil.temp<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @unroll(%{{.*}}: f64) 
func @unroll(%in : f64)
  attributes { stencil.program } {
  %0 = "stencil.apply"(%in) ({
    ^bb0(%1 : f64):
    %2 = "stencil.store_result"(%1) : (f64) -> !stencil.result<f64>
    %3 = "stencil.store_result"(%1) : (f64) -> !stencil.result<f64>
    //  CHECK: stencil.return unroll [1, 2, 1] %{{.*}}, %{{.*}} : !stencil.result<f64>, !stencil.result<f64>
    "stencil.return"(%2, %3) {unroll = [1, 2, 1]} : (!stencil.result<f64>, !stencil.result<f64>) -> ()
  }) : (f64) -> !stencil.temp<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @unroll_2(%{{.*}}: f64, %{{.*}}: f32, %{{.*}}: !stencil.field<?x?x?xf64>)
func @unroll_2(%in_0 : f64, %in_1 : f32, %out : !stencil.field<?x?x?xf64>)
  attributes { stencil.program } {
  %0 = "stencil.cast"(%out) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<?x?x?xf64>) -> (!stencil.field<70x70x60xf64>)
  %1 = "stencil.load"(%0) {lb=[-3,-3,0], ub=[67,67,60]} : (!stencil.field<70x70x60xf64>) -> (!stencil.temp<70x70x60xf64>)  
  %2, %3 = "stencil.apply"(%in_0, %in_1) ({
    ^bb0(%4 : f64, %5 : f32):
    %6 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
    %7 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
    %8 = "stencil.store_result"(%5) : (f32) -> !stencil.result<f32>
    %9 = "stencil.store_result"(%5) : (f32) -> !stencil.result<f32>
    //  CHECK: stencil.return unroll [1, 2, 1] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f32>, !stencil.result<f32>
    "stencil.return"(%6, %7, %8, %9) {unroll = [1, 2, 1]} : (!stencil.result<f64>, !stencil.result<f64>, !stencil.result<f32>, !stencil.result<f32>) -> ()
  }) : (f64, f32) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @sequential(%{{.*}}: f64) 
func @sequential(%in : f64)
  attributes { stencil.program } {
  // CHECK %{{.*}} = stencil.apply seq(dim = 2, range = 0 to 60, dir = 1) (%{{.*}}: f64) -> !stencil.temp<1x2x3xf64>
  %0 = "stencil.apply"(%in) ({
    ^bb0(%1 : f64):
    %2 = "stencil.store_result"(%1) : (f64) -> !stencil.result<f64>
    "stencil.return"(%2) : (!stencil.result<f64>) -> ()
  }) {seq=[2, 0, 60, 1]} : (f64) -> !stencil.temp<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @store_result(%{{.*}}: f64) 
func @store_result(%in : f64)
  attributes { stencil.program } {
  %0, %1 = "stencil.apply"(%in) ({
    ^bb0(%1 : f64):
    //  CHECK: %{{.*}} = stencil.store_result %{{.*}} : (f64) -> !stencil.result<f64>
    %2= "stencil.store_result"(%1) : (f64) -> !stencil.result<f64>
    //  CHECK: %{{.*}} = stencil.store_result : () -> !stencil.result<f64>
    %3 = "stencil.store_result"() : () -> !stencil.result<f64>
    "stencil.return"(%2, %3) : (!stencil.result<f64>, !stencil.result<f64>) -> ()
  }) : (f64) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>)
  return
}
