// RUN: oec-opt %s --stencil-domain-split | oec-opt | FileCheck %s

// CHECK-LABEL: func @split_apply
func @split_apply(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%2 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %3 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %3 : !stencil.result<f64>
}
stencil.store %2 to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
stencil.store %2 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_apply_multiple_results
func @split_apply_multiple_results(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%2:2 = stencil.apply -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
  %cst = constant 1.000000e+00 : f64
  %3 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %3, %3 : !stencil.result<f64>, !stencil.result<f64>
}
stencil.store %2#0 to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
stencil.store %2#1 to %1([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_combine_buffer_and_load
func @split_combine_buffer_and_load(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%100 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%101 = stencil.load %100 : (!stencil.field<136x136x136xf64>) -> !stencil.temp<?x?x?xf64>
%1 = stencil.apply(%arg2 = %101 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %5 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %5 : !stencil.result<f64>
}
%2 = stencil.buffer %1 : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
%3 = stencil.apply(%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 2.000000e+00 : f64
  %4 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %4 : !stencil.result<f64>
}
%4 = stencil.combine 2 at 11 lower = (%3 : !stencil.temp<?x?x?xf64>) upper = (%3 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
stencil.store %4 to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_combine
func @split_combine(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%1 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %4 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %4 : !stencil.result<f64>
}
%3 = stencil.combine 2 at 11 lower = (%1 : !stencil.temp<?x?x?xf64>) upper = (%1 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
stencil.store %3 to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_combine_on_op
func @split_combine_on_op(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%2 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %5 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %5 : !stencil.result<f64>
}
%3 = stencil.combine 2 at 11 lower = (%2 : !stencil.temp<?x?x?xf64>) upper = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
%4 = stencil.combine 2 at 11 lower = (%2 : !stencil.temp<?x?x?xf64>) upper = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
stencil.store %3 to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %4 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_multiple_applies
func @split_multiple_applies(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%2 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %4 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %4 : !stencil.result<f64>
}
%3 = stencil.apply(%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 2.000000e+00 : f64
  %4 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %4 : !stencil.result<f64>
}
stencil.store %3 to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
stencil.store %3 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_multiple_combines
func @split_multiple_combines(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%2 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %5 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %5 : !stencil.result<f64>
}
%3 = stencil.combine 2 at 11 lower = (%2 : !stencil.temp<?x?x?xf64>) upper = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
%4 = stencil.combine 2 at 11 lower = (%2 : !stencil.temp<?x?x?xf64>) upper = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
%5 = stencil.combine 1 at 30 lower = (%3 : !stencil.temp<?x?x?xf64>) upper = (%4 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
stencil.store %5 to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_rhombus
func @split_rhombus(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%2 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6 : !stencil.result<f64>
}
%3 = stencil.apply(%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 2.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6 : !stencil.result<f64>
}
%4 = stencil.apply(%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 3.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6 : !stencil.result<f64>
}
%5 = stencil.apply(%arg3 = %3 : !stencil.temp<?x?x?xf64>, %arg4 = %4 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 4.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6 : !stencil.result<f64>
}
stencil.store %5 to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
stencil.store %5 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_rhombus_multiple_results
func @split_rhombus_multiple_results(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg100: !stencil.field<?x?x?xf64>, %arg101: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%100 = stencil.cast %arg100([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%101 = stencil.cast %arg101([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%2:3 = stencil.apply -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
  %cst = constant 1.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6, %6, %6 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
}
%3:3 = stencil.apply(%arg2 = %2#0 : !stencil.temp<?x?x?xf64>, %arg3 = %2#1 : !stencil.temp<?x?x?xf64>, %arg4 = %2#2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
  %cst = constant 2.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6, %6, %6 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
}
%4:3 = stencil.apply(%arg2 = %2#0 : !stencil.temp<?x?x?xf64>, %arg3 = %2#1 : !stencil.temp<?x?x?xf64>, %arg4 = %2#2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
  %cst = constant 3.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6, %6, %6 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
}
%5:3 = stencil.apply(%arg2 = %3#0 : !stencil.temp<?x?x?xf64>, %arg3 = %3#1 : !stencil.temp<?x?x?xf64>, %arg4 = %3#2 : !stencil.temp<?x?x?xf64>, %arg5 = %4#0 : !stencil.temp<?x?x?xf64>, %arg6 = %4#1 : !stencil.temp<?x?x?xf64>, %arg7 = %4#2 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
  %cst = constant 4.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6, %6, %6 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
}
stencil.store %5#0 to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
stencil.store %5#0 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %5#1 to %100([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %5#2 to %101([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}
