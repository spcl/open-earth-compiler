// RUN: oec-opt %s --stencil-domain-split | oec-opt | FileCheck %s

// CHECK-LABEL: func @split_apply
func @split_apply(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
// CHECK: [[VAL2:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: [[VAL1:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
%2 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %3 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %3 : !stencil.result<f64>
}
// CHECK: stencil.store [[VAL1]] to %0([0, 0, 0] : [64, 64, 64])
// CHECK: stencil.store [[VAL2]] to %1([0, 0, 0] : [128, 128, 128])
stencil.store %2 to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
stencil.store %2 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_combine
func @split_combine(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
// CHECK: [[VAL1:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: [[VAL2:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
%1 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %4 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %4 : !stencil.result<f64>
}
// CHECK: [[VAL3:%.*]] = stencil.combine 2 at 11 lower = ([[VAL2]] : !stencil.temp<?x?x?xf64>) upper = ([[VAL1]] : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>
%3 = stencil.combine 2 at 11 lower = (%1 : !stencil.temp<?x?x?xf64>) upper = (%1 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
// CHECK: stencil.store [[VAL3]] to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %3 to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_combine_buffer_and_load
func @split_combine_buffer_and_load(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
// CHECK: [[VAL1:%.*]] = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%100 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
// CHECK-NEXT: [[VAL2:%.*]] = stencil.load [[VAL1]] : (!stencil.field<136x136x136xf64>) -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT: [[VAL3:%.*]] = stencil.load [[VAL1]] : (!stencil.field<136x136x136xf64>) -> !stencil.temp<?x?x?xf64>
%101 = stencil.load %100 : (!stencil.field<136x136x136xf64>) -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT: [[VAL4:%.*]] = stencil.apply (%arg2 = [[VAL3]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: [[VAL5:%.*]] = stencil.apply (%arg2 = [[VAL2]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
%1 = stencil.apply(%arg2 = %101 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %5 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %5 : !stencil.result<f64>
}
// CHECK: [[VAL6:%.*]] = stencil.buffer [[VAL5]] : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT: [[VAL7:%.*]] = stencil.buffer [[VAL4]] : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
%2 = stencil.buffer %1 : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT: [[VAL8:%.*]] = stencil.apply (%arg2 = [[VAL7]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: [[VAL9:%.*]] = stencil.apply (%arg2 = [[VAL6]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
%3 = stencil.apply(%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 2.000000e+00 : f64
  %4 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %4 : !stencil.result<f64>
}
// CHECK: [[VAL10:%.*]] = stencil.combine 2 at 11 lower = ([[VAL9]] : !stencil.temp<?x?x?xf64>) upper = ([[VAL8]] : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>
%4 = stencil.combine 2 at 11 lower = (%3 : !stencil.temp<?x?x?xf64>) upper = (%3 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
// CHECK-NEXT: stencil.store [[VAL10]] to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %4 to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_combine_on_op
func @split_combine_on_op(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
// CHECK: [[VAL2:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: [[VAL3:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: [[VAL4:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: [[VAL5:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
%2 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %5 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %5 : !stencil.result<f64>
}
// CHECK: [[VAL6:%.*]] = stencil.combine 2 at 11 lower = ([[VAL4]] : !stencil.temp<?x?x?xf64>) upper = ([[VAL3]] : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>
%3 = stencil.combine 2 at 11 lower = (%2 : !stencil.temp<?x?x?xf64>) upper = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
// CHECK-NEXT: [[VAL7:%.*]] = stencil.combine 2 at 11 lower = ([[VAL5]] : !stencil.temp<?x?x?xf64>) upper = ([[VAL2]] : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>
%4 = stencil.combine 2 at 11 lower = (%2 : !stencil.temp<?x?x?xf64>) upper = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
// CHECK-NEXT: stencil.store [[VAL6]] to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %3 to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
// CHECK-NEXT: stencil.store [[VAL7]] to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %4 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_multiple_combines
func @split_multiple_combines(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
// CHECK: [[VAL1:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: [[VAL2:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: [[VAL3:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: [[VAL4:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
%2 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %5 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %5 : !stencil.result<f64>
}
// CHECK: [[VAL5:%.*]] = stencil.combine 2 at 11 lower = ([[VAL4]] : !stencil.temp<?x?x?xf64>) upper = ([[VAL3]] : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>
%3 = stencil.combine 2 at 11 lower = (%2 : !stencil.temp<?x?x?xf64>) upper = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
// CHECK-NEXT: [[VAL6:%.*]] = stencil.combine 2 at 11 lower = ([[VAL2]] : !stencil.temp<?x?x?xf64>) upper = ([[VAL1]] : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>
%4 = stencil.combine 2 at 11 lower = (%2 : !stencil.temp<?x?x?xf64>) upper = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
// CHECK-NEXT: [[VAL7:%.*]] = stencil.combine 1 at 30 lower = ([[VAL5]] : !stencil.temp<?x?x?xf64>) upper = ([[VAL6]] : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>
%5 = stencil.combine 1 at 30 lower = (%3 : !stencil.temp<?x?x?xf64>) upper = (%4 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>
// CHECK-NEXT: stencil.store [[VAL7]] to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %5 to %0([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}

// -----

// CHECK-LABEL: func @split_rhombus
func @split_rhombus(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
%0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
%1 = stencil.cast %arg1([-4, -4, -4] : [132, 132, 132]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<136x136x136xf64>
// CHECK: [[VAL2:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 1.000000e+00 : f64
// CHECK: [[VAL3:%.*]] = stencil.apply -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 1.000000e+00 : f64
%2 = stencil.apply -> !stencil.temp<?x?x?xf64> {
  %cst = constant 1.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6 : !stencil.result<f64>
}
// CHECK: [[VAL4:%.*]] = stencil.apply (%arg2 = [[VAL3]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 2.000000e+00 : f64
// CHECK: [[VAL5:%.*]] = stencil.apply (%arg2 = [[VAL2]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 2.000000e+00 : f64
%3 = stencil.apply(%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 2.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6 : !stencil.result<f64>
}
// CHECK: [[VAL6:%.*]] = stencil.apply (%arg2 = [[VAL3]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 3.000000e+00 : f64
// CHECK: [[VAL7:%.*]] = stencil.apply (%arg2 = [[VAL2]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 3.000000e+00 : f64
%4 = stencil.apply(%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 3.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6 : !stencil.result<f64>
}
// CHECK: [[VAL8:%.*]] = stencil.apply (%arg2 = [[VAL5]] : !stencil.temp<?x?x?xf64>, %arg3 = [[VAL7]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 4.000000e+00 : f64
// CHECK: [[VAL9:%.*]] = stencil.apply (%arg2 = [[VAL4]] : !stencil.temp<?x?x?xf64>, %arg3 = [[VAL6]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
// CHECK: %cst = constant 4.000000e+00 : f64
%5 = stencil.apply(%arg3 = %3 : !stencil.temp<?x?x?xf64>, %arg4 = %4 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %cst = constant 4.000000e+00 : f64
  %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
  stencil.return %6 : !stencil.result<f64>
}
// CHECK: stencil.store [[VAL9]] to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
stencil.store %5 to %0([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
// CHECK-NEXT: stencil.store [[VAL8]] to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
stencil.store %5 to %1([0, 0, 0] : [128, 128, 128]) : !stencil.temp<?x?x?xf64> to !stencil.field<136x136x136xf64>
return
}
