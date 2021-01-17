// RUN: oec-opt %s -split-input-file --stencil-shape-overlap | oec-opt | FileCheck %s

// CHECK-LABEL: func @ioverlap(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @ioverlap(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.cast %arg2([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %3 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  // CHECK: [[TEMP1:%.*]] = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  // CHECK: [[TEMP2:%.*]] = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %4 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  }
  %5 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  }
  // CHECK:      [[TEMP3:%.*]]:2 = stencil.combine 0 at 64 lower = ([[TEMP2]] : !stencil.temp<?x?x?xf64>) upper = ([[TEMP2]] : !stencil.temp<?x?x?xf64>) lowerext = ([[TEMP1]] : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
  // CHECK-NEXT: [[TEMP4:%.*]]:2 = stencil.combine 0 at 0 lower = ([[TEMP1]] : !stencil.temp<?x?x?xf64>) upper = ([[TEMP3]]#1 : !stencil.temp<?x?x?xf64>) upperext = ([[TEMP3]]#0 : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
  // CHECK-NEXT: stencil.store [[TEMP4:%.*]]#0 to %{{.*}}([-1, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  // CHECK-NEXT: stencil.store [[TEMP4:%.*]]#1 to %{{.*}}([0, 0, 0] : [65, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  stencil.store %4 to %1([-1, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  stencil.store %5 to %2([0, 0, 0] : [65, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----
