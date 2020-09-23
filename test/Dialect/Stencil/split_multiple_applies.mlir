// RUN: oec-opt %s --stencil-domain-split | oec-opt | FileCheck %s

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
