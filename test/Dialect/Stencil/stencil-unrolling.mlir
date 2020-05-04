// RUN: oec-opt %s --stencil-unrolling='unroll-factor=4' -cse | oec-opt | FileCheck %s

// CHECK-LABEL: func @unroll_accesses
func @unroll_accesses(%arg0 : !stencil.field<ijk,f64>, %arg1 : !stencil.field<ijk,f64>) attributes { stencil.program } {
	stencil.assert %arg0 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %arg1 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  %0 = stencil.load %arg0 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %1 = stencil.apply (%arg2 = %0 : !stencil.temp<ijk,f64>) -> !stencil.temp<ijk,f64> {
      // CHECK: [[ACC1:%.*]] = stencil.access {{%.*}}[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      // CHECK: [[ACC2:%.*]] = stencil.access {{%.*}}[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      // CHECK: [[ACC3:%.*]] = stencil.access {{%.*}}[0, 2, 0] : (!stencil.temp<ijk,f64>) -> f64
      // CHECK: [[ACC4:%.*]] = stencil.access {{%.*}}[0, 3, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      // CHECK: stencil.return unroll [1, 4, 1] [[ACC1]], [[ACC2]], [[ACC3]], [[ACC4]]  : f64
      stencil.return %2 : f64
	} to ([0, 0, 0]:[64, 64, 64])
  stencil.store %1 to %arg1([0, 0, 0]:[64, 64, 60]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  return
}
