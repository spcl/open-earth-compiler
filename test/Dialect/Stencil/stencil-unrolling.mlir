// RUN: oec-opt %s -split-input-file --stencil-unrolling='unroll-factor=4' -cse | oec-opt | FileCheck %s

// CHECK-LABEL: func @unroll_accesses
func @unroll_accesses(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes { stencil.program } {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    // CHECK: [[ACC1:%.*]] = stencil.access {{%.*}}[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK: [[ACC2:%.*]] = stencil.access {{%.*}}[0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK: [[ACC3:%.*]] = stencil.access {{%.*}}[0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK: [[ACC4:%.*]] = stencil.access {{%.*}}[0, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK: stencil.return unroll [1, 4, 1] [[ACC1]], [[ACC2]], [[ACC3]], [[ACC4]]  : f64
    stencil.return %4 : f64
	} to ([0, 0, 0]:[64, 64, 64])
  stencil.store %3 to %1([0, 0, 0]:[64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @unroll_index
func @unroll_index(%arg0 : f64, %arg1 : !stencil.field<?x?x?xf64>) attributes { stencil.program } {
  %0 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.apply (%arg2 = %arg0 : f64) -> !stencil.temp<?x?x?xf64> {
      // CHECK: [[IDX1:%.*]] = stencil.index 2 [0, 0, 0] : index
      // CHECK: [[IDX2:%.*]] = stencil.index 2 [0, 1, 0] : index
      // CHECK: [[IDX3:%.*]] = stencil.index 2 [0, 2, 0] : index
      // CHECK: [[IDX4:%.*]] = stencil.index 2 [0, 3, 0] : index
      %2 = stencil.index 2 [0, 0, 0] : index
      %3 = constant 20 : index
      %4 = constant 0.0 : f64
      %5 = cmpi "slt", %2, %3 : index
      %6 = select %5, %arg2, %4 : f64
      stencil.return %6 : f64
    } to ([0, 0, 0]:[64, 64, 64])
  stencil.store %1 to %0([0, 0, 0]:[64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

