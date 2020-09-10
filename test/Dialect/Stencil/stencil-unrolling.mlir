// RUN: oec-opt %s -split-input-file --stencil-unrolling='unroll-factor=4' -cse | oec-opt | FileCheck %s

// CHECK-LABEL: func @access
func @access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes { stencil.program } {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    // CHECK-DAG: [[ACC1:%.*]] = stencil.access {{%.*}}[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK-DAG: [[ACC2:%.*]] = stencil.access {{%.*}}[0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK-DAG: [[ACC3:%.*]] = stencil.access {{%.*}}[0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK-DAG: [[ACC4:%.*]] = stencil.access {{%.*}}[0, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK-DAG: [[RES1:%.*]] = stencil.store_result [[ACC1]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES2:%.*]] = stencil.store_result [[ACC2]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES3:%.*]] = stencil.store_result [[ACC3]] : (f64) -> !stencil.result<f64>
    // CHECK-DAG: [[RES4:%.*]] = stencil.store_result [[ACC4]] : (f64) -> !stencil.result<f64>
    %5 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
    // CHECK: stencil.return unroll [1, 4, 1] [[RES1]], [[RES2]], [[RES3]], [[RES4]] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
    stencil.return %5 : !stencil.result<f64>
	} to ([0, 0, 0]:[64, 64, 64])
  stencil.store %3 to %1([0, 0, 0]:[64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @index
func @index(%arg0 : f64, %arg1 : !stencil.field<?x?x?xf64>) attributes { stencil.program } {
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
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0]:[64, 64, 64])
  stencil.store %1 to %0([0, 0, 0]:[64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @dyn_access
func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes { stencil.program } {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply ([[ARG2:%.*]] = %{{.*}} 
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    // CHECK: {{%.*}} = stencil.dyn_access [[ARG2]](%{{.*}}, %{{.*}}, %{{.*}}) in [0, 0, 0] : [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK: {{%.*}} = stencil.dyn_access [[ARG2]](%{{.*}}, %{{.*}}, %{{.*}}) in [0, 1, 0] : [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK: {{%.*}} = stencil.dyn_access [[ARG2]](%{{.*}}, %{{.*}}, %{{.*}}) in [0, 2, 0] : [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    // CHECK: {{%.*}} = stencil.dyn_access [[ARG2]](%{{.*}}, %{{.*}}, %{{.*}}) in [0, 3, 0] : [0, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.index 0 [0, 0, 0] : index
    %5 = stencil.index 1 [0, 0, 0] : index
    %6 = stencil.index 2 [0, 0, 0] : index
    %7 = stencil.dyn_access %arg2(%4, %5, %6) in [0, 0, 0] : [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 0, 0]:[64, 64, 64])
  stencil.store %3 to %1([0, 0, 0]:[64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}
