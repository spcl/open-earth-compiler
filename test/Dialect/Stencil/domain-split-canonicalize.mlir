// RUN: oec-opt %s --stencil-domain-split --canonicalize | oec-opt | FileCheck %s

// CHECK-LABEL: func @split_apply_multiple_results
func @split_apply_multiple_results(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
// CHECK: [[VAL2:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 1.000000e+00 : f64
// CHECK: [[VAL3:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 2.000000e+00 : f64
%2:2 = stencil.apply -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
  %one = constant 1.000000e+00 : f64
  %two = constant 2.000000e+00 : f64
  %3 = stencil.store_result %one : (f64) -> !stencil.result<f64>
  %4 = stencil.store_result %two : (f64) -> !stencil.result<f64>
  stencil.return %3, %4 : !stencil.result<f64>, !stencil.result<f64>
}
// CHECK: stencil.store [[VAL2]] to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
stencil.store %2#0 to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
// CHECK: stencil.store [[VAL3]] to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %2#1 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}


// -----

// CHECK-LABEL: func @split_rhombus_multiple_results
func @split_rhombus_multiple_results(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg100: !stencil.field<?x?x?xf64>, %arg101: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%100 = stencil.cast %arg100([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%101 = stencil.cast %arg101([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
// CHECK: [[VAL4:%.*]]:3 = stencil.apply -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
// CHECK: %cst = constant 1.000000e+00 : f64
// CHECK: [[VAL5:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 1.000000e+00 : f64
%2:3 = stencil.apply -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
  %cst = constant 1.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6, %6, %6 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
}
// CHECK: [[VAL6:%.*]] = stencil.apply (%arg4 = [[VAL5]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 2.000000e+00 : f64
// CHECK: [[VAL7:%.*]]:3 = stencil.apply (%arg4 = [[VAL4]]#0 : !stencil.temp<?x?x?xf64>, %arg5 = [[VAL4]]#1 : !stencil.temp<?x?x?xf64>, %arg6 = [[VAL4]]#2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) { 
// CHECK: %cst = constant 2.000000e+00 : f64
%3:3 = stencil.apply(%arg2 = %2#0 : !stencil.temp<?x?x?xf64>, %arg3 = %2#1 : !stencil.temp<?x?x?xf64>, %arg4 = %2#2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
  %cst = constant 2.000000e+00 : f64
  %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %8 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %10 = mulf %cst, %6 : f64
  %11 = mulf %cst, %7 : f64
  %12 = mulf %cst, %8 : f64
  %15 = stencil.store_result %10 : (f64) -> !stencil.result<f64>
  %16 = stencil.store_result %11 : (f64) -> !stencil.result<f64>
  %17 = stencil.store_result %12 : (f64) -> !stencil.result<f64>
  stencil.return %15, %16, %17 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
}
// CHECK: [[VAL8:%.*]] = stencil.apply (%arg4 = [[VAL5]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 3.000000e+00 : f64
// CHECK: [[VAL9:%.*]]:3 = stencil.apply (%arg4 = [[VAL4]]#0 : !stencil.temp<?x?x?xf64>, %arg5 = [[VAL4]]#1 : !stencil.temp<?x?x?xf64>, %arg6 = [[VAL4]]#2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) { 
// CHECK: %cst = constant 3.000000e+00 : f64
%4:3 = stencil.apply(%arg2 = %2#0 : !stencil.temp<?x?x?xf64>, %arg3 = %2#1 : !stencil.temp<?x?x?xf64>, %arg4 = %2#2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
  %cst = constant 3.000000e+00 : f64
  %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %8 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %10 = mulf %cst, %6 : f64
  %11 = mulf %cst, %7 : f64
  %12 = mulf %cst, %8 : f64
  %15 = stencil.store_result %10 : (f64) -> !stencil.result<f64>
  %16 = stencil.store_result %11 : (f64) -> !stencil.result<f64>
  %17 = stencil.store_result %12 : (f64) -> !stencil.result<f64>
  stencil.return %15, %16, %17 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
}
// CHECK: [[VAL10:%.*]]:3 = stencil.apply (%arg4 = [[VAL7]]#0 : !stencil.temp<?x?x?xf64>, %arg5 = [[VAL7]]#1 : !stencil.temp<?x?x?xf64>, %arg6 = [[VAL7]]#2 : !stencil.temp<?x?x?xf64>, %arg7 = [[VAL9]]#0 : !stencil.temp<?x?x?xf64>, %arg8 = [[VAL9]]#1 : !stencil.temp<?x?x?xf64>, %arg9 = [[VAL9]]#2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
// CHECK: %cst = constant 4.000000e+00 : f64
// CHECK: [[VAL11:%.*]] = stencil.apply (%arg4 = [[VAL6]] : !stencil.temp<?x?x?xf64>, %arg5 = [[VAL8]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 4.000000e+00 : f64
%5:3 = stencil.apply(%arg2 = %3#0 : !stencil.temp<?x?x?xf64>, %arg3 = %3#1 : !stencil.temp<?x?x?xf64>, %arg4 = %3#2 : !stencil.temp<?x?x?xf64>, %arg5 = %4#0 : !stencil.temp<?x?x?xf64>, %arg6 = %4#1 : !stencil.temp<?x?x?xf64>, %arg7 = %4#2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
  %cst = constant 4.000000e+00 : f64
  %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %8 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %9 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %10 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %11 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
  %20 = mulf %cst, %6 : f64
  %21 = mulf %cst, %7 : f64
  %22 = mulf %cst, %8 : f64
  %23 = mulf %20, %9 : f64
  %24 = mulf %21, %10: f64
  %25 = mulf %22, %11 : f64
  %30 = stencil.store_result %23 : (f64) -> !stencil.result<f64>
  %31 = stencil.store_result %24 : (f64) -> !stencil.result<f64>
  %32 = stencil.store_result %25 : (f64) -> !stencil.result<f64>
  stencil.return %30, %31, %32 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
}
// CHECK: stencil.store [[VAL11]] to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
// CHECK-NEXT: stencil.store [[VAL10]]#0 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
// CHECK-NEXT: stencil.store [[VAL10]]#1 to %2([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
// CHECK-NEXT: stencil.store [[VAL10]]#2 to %3([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>

stencil.store %5#0 to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
stencil.store %5#0 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %5#1 to %100([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %5#2 to %101([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}
