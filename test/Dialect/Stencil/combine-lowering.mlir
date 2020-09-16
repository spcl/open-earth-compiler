// RUN: oec-opt %s -split-input-file --stencil-combine-lowering -cse | oec-opt | FileCheck %s

// CHECK-LABEL: func @simple
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0([0, 0, 0] : [65, 64, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<65x64x60xf64>
  // CHECK-DAG: [[IDX:%.*]] = stencil.index 0 [0, 0, 0] : index
  // CHECK-DAG: [[C32:%.*]] = constant 32 : index
  // CHECK-DAG: [[COND:%.*]] = cmpi "ult", [[IDX]], [[C32]] : index
  // CHECK: [[RES:%.*]] = scf.if [[COND]] -> (!stencil.result<f64>) {
  // CHECK-NEXT: [[ACC1:%.*]] = stencil.access {{%.*}}[0, 0, 0] : (!stencil.temp<65x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES1:%.*]] = stencil.store_result [[ACC1]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT:  scf.yield [[RES1]] : !stencil.result<f64>
  // CHECK-NEXT: } else {
  // CHECK-NEXT: [[ACC2:%.*]] = stencil.access {{%.*}}[1, 0, 0] : (!stencil.temp<65x64x60xf64>) -> f64
  // CHECK-NEXT: [[RES2:%.*]] = stencil.store_result [[ACC2]] : (f64) -> !stencil.result<f64>
  // CHECK-NEXT:  scf.yield [[RES2]] : !stencil.result<f64>
  // CHECK-NEXT: }
  // CHECK-NEXT: stencil.return [[RES]] : !stencil.result<f64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<65x64x60xf64>) -> !stencil.temp<32x64x60xf64> {
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<65x64x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [32, 64, 60])
  %4 = stencil.apply (%arg2 = %2 : !stencil.temp<65x64x60xf64>) -> !stencil.temp<32x64x60xf64> {
    %6 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<65x64x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([32, 0, 0] : [64, 64, 60])
  // CHECK-NOT: {{%.*}} = stencil.combine
  %5 = stencil.combine 0 at 32 lower = (%3 : !stencil.temp<32x64x60xf64>) upper = (%4 : !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>
  stencil.store %5 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  return
}



