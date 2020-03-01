// RUN: oec-opt %s --stencil-unrolling -cse | oec-opt | FileCheck %s

// CHECK-LABEL: func @unroll_accesses
func @unroll_accesses(%arg0 : !stencil.field<ijk,f64>) attributes { stencil.program } {
	stencil.assert %arg0 ([0, 0, 0]:[7, 77, 777]) : !stencil.field<ijk,f64>
  %0 = stencil.load %arg0 : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %1 = stencil.apply %arg1 = %0 : !stencil.view<ijk,f64> {  
      // CHECK: [[ACC1:%.*]] = stencil.access {{%.*}}[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      // CHECK: [[ACC2:%.*]] = stencil.access {{%.*}}[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      // CHECK: stencil.return unroll [1, 2, 1] [[ACC1]], [[ACC2]] : f64
      stencil.return %2 : f64
	} : !stencil.view<ijk,f64>
  return
}