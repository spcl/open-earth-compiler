// RUN: oec-opt %s -split-input-file --stencil-peel-odd-iterations -cse | oec-opt | FileCheck %s

// CHECK-LABEL: func @peel_loop
func @peel_loop(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0([0, 0, 0] : [64, 61, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<64x61x60xf64>
  // CHECK: [[BODY:%.*]] = stencil.apply ({{%.*}} = {{%.*}} : !stencil.temp<64x61x60xf64>) -> !stencil.temp<64x60x60xf64> {
  // CHECK: } to ([0, 0, 0] : [64, 60, 60])
  // CHECK: [[PEEL:%.*]] = stencil.apply ({{%.*}} = {{%.*}} : !stencil.temp<64x61x60xf64>) -> !stencil.temp<64x1x60xf64> {
  // CHECK-NEXT: [[ACC1:%.*]] = stencil.access {{%.*}}[0, 0, 0] : (!stencil.temp<64x61x60xf64>) -> f64
  // CHECK-NEXT: [[RES1:%.*]] = stencil.store_result [[ACC1]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT: [[RES2:%.*]] = stencil.store_result : () -> !stencil.result<f64>
  // CHECK: stencil.return unroll [1, 4, 1] [[RES1]], [[RES2]], [[RES2]], [[RES2]] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>  
  // CHECK: } to ([0, 60, 0] : [64, 61, 60])
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<64x61x60xf64>) -> !stencil.temp<64x61x60xf64> {
    %4 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x61x60xf64>) -> f64
    %5 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
    %6 = stencil.access %arg2 [0, 1, 0] : (!stencil.temp<64x61x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    %8 = stencil.access %arg2 [0, 2, 0] : (!stencil.temp<64x61x60xf64>) -> f64
    %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    %10 = stencil.access %arg2 [0, 3, 0] : (!stencil.temp<64x61x60xf64>) -> f64
    %11 = stencil.store_result %10 : (f64) -> !stencil.result<f64>
    stencil.return unroll [1, 4, 1] %5, %7, %9, %11 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0] : [64, 61, 60])
  // CHECK: {{%.*}} = stencil.combine 1 at 60 lower = ([[BODY]] : !stencil.temp<64x60x60xf64>) upper = ([[PEEL]] : !stencil.temp<64x1x60xf64>) ([0, 0, 0] : [64, 61, 60]) : !stencil.temp<64x61x60xf64>
  stencil.store %3 to %1([0, 0, 0] : [64, 61, 60]) : !stencil.temp<64x61x60xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @peel_only
func @peel_only(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0([0, 0, 0] : [64, 1, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<64x1x60xf64>
  // CHECK: [[PEEL:%.*]] = stencil.apply ({{%.*}} = {{%.*}} : !stencil.temp<64x1x60xf64>) -> !stencil.temp<64x1x60xf64> {
  // CHECK-NEXT: [[ACC1:%.*]] = stencil.access {{%.*}}[0, 0, 0] : (!stencil.temp<64x1x60xf64>) -> f64
  // CHECK-NEXT: [[RES1:%.*]] = stencil.store_result [[ACC1]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT: [[RES2:%.*]] = stencil.store_result : () -> !stencil.result<f64>
  // CHECK: stencil.return unroll [1, 4, 1] [[RES1]], [[RES2]], [[RES2]], [[RES2]] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>  
  // CHECK: } to ([0, 0, 0] : [64, 1, 60])
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<64x1x60xf64>) -> !stencil.temp<64x1x60xf64> {
    %4 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x1x60xf64>) -> f64
    %5 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
    %6 = stencil.access %arg2 [0, 1, 0] : (!stencil.temp<64x1x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    %8 = stencil.access %arg2 [0, 2, 0] : (!stencil.temp<64x1x60xf64>) -> f64
    %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    %10 = stencil.access %arg2 [0, 3, 0] : (!stencil.temp<64x1x60xf64>) -> f64
    %11 = stencil.store_result %10 : (f64) -> !stencil.result<f64>
    stencil.return unroll [1, 4, 1] %5, %7, %9, %11 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0] : [64, 1, 60])
  // CHECK: stencil.store [[PEEL]] to {{%.*}}([0, 0, 0] : [64, 1, 60]) : !stencil.temp<64x1x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %1([0, 0, 0] : [64, 1, 60]) : !stencil.temp<64x1x60xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @combine_tree
func @combine_tree(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0([0, 0, 0] : [64, 64, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<64x64x60xf64>
  // CHECK: [[LEFT:%.*]] = stencil.apply ({{%.*}} = {{%.*}} : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x60x60xf64> {
  // CHECK-NEXT: [[ACC1:%.*]] = stencil.access {{%.*}}[0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES1:%.*]] = stencil.store_result [[ACC1]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT: [[ACC2:%.*]] = stencil.access {{%.*}}[0, 1, 0] : (!stencil.temp<64x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES2:%.*]] = stencil.store_result [[ACC2]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT: [[ACC3:%.*]] = stencil.access {{%.*}}[0, 2, 0] : (!stencil.temp<64x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES3:%.*]] = stencil.store_result [[ACC3]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT: [[ACC4:%.*]] = stencil.access {{%.*}}[0, 3, 0] : (!stencil.temp<64x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES4:%.*]] = stencil.store_result [[ACC4]] : (f64) -> !stencil.result<f64>
  // CHECK: stencil.return unroll [1, 4, 1] [[RES1]], [[RES2]], [[RES3]], [[RES4]] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>  
  // CHECK: } to ([0, 0, 0] : [64, 60, 60])
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x62x60xf64> {
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    %8 = stencil.access %arg2 [0, 1, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    %10 = stencil.access %arg2 [0, 2, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %11 = stencil.store_result %10 : (f64) -> !stencil.result<f64>
    %12 = stencil.access %arg2 [0, 3, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %13 = stencil.store_result %12 : (f64) -> !stencil.result<f64>
    stencil.return unroll [1, 4, 1] %7, %9, %11, %13 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0] : [64, 62, 60])
  // CHECK: [[RIGHT:%.*]] = stencil.apply ({{%.*}} = {{%.*}} : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x3x60xf64> {
  // CHECK-NEXT: [[CST:%.*]] = constant 1.000000e+00 : f64
  // CHECK-NEXT: [[ACC1:%.*]] = stencil.access {{%.*}}[0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES1:%.*]] = stencil.store_result [[ACC1]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT: [[ACC2:%.*]] = stencil.access {{%.*}}[0, 1, 0] : (!stencil.temp<64x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES2:%.*]] = stencil.store_result [[ACC2]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT: [[RES3:%.*]] = stencil.store_result [[CST]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT: [[RES4:%.*]] = stencil.store_result : () -> !stencil.result<f64>
  // CHECK: stencil.return unroll [1, 4, 1] [[RES1]], [[RES2]], [[RES3]], [[RES4]] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>  
  // CHECK: } to ([0, 60, 0] : [64, 63, 60])
  %4 = stencil.apply (%arg2 = %2 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x1x60xf64> {
    %cst = constant 1.000000e+00 : f64
    %6 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    %7 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    %8 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    %9 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return unroll [1, 4, 1] %6, %7, %8, %9 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 62, 0] : [64, 63, 60])
  // CHECK: {{%.*}} = stencil.combine 1 at 60 lower = ([[LEFT]] : !stencil.temp<64x60x60xf64>) upper = ([[RIGHT]] : !stencil.temp<64x3x60xf64>) ([0, 0, 0] : [64, 63, 60]) : !stencil.temp<64x63x60xf64>
  %5 = stencil.combine 1 at 62 lower = (%3 : !stencil.temp<64x62x60xf64>) upper = (%4 : !stencil.temp<64x1x60xf64>) ([0, 0, 0] : [64, 63, 60]) : !stencil.temp<64x63x60xf64>
  stencil.store %5 to %1([0, 0, 0] : [64, 63, 60]) : !stencil.temp<64x63x60xf64> to !stencil.field<70x70x60xf64>
  return
}