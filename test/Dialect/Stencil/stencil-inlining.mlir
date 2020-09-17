// RUN: oec-opt %s -split-input-file --stencil-inlining --cse | oec-opt | FileCheck %s

// CHECK-LABEL: func @simple(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x66x63xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<66x66x63xf64>) -> !stencil.temp<64x64x60xf64> {
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<66x66x63xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [0, 2, 3] : (!stencil.temp<66x66x63xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [2, 2, 3] : (!stencil.temp<66x66x63xf64>) -> f64
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %2 = stencil.load %0([0, 0, 0] : [66, 66, 63]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x66x63xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<66x66x63xf64>) -> !stencil.temp<64x64x60xf64> {
    %5 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<66x66x63xf64>) -> f64
    %6 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<66x66x63xf64>) -> f64
    %7 = addf %5, %6 : f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([1, 2, 3] : [65, 66, 63])
  %4 = stencil.apply (%arg2 = %2 : !stencil.temp<66x66x63xf64>, %arg3 = %3 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %5 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<66x66x63xf64>) -> f64
    %6 = stencil.access %arg3 [1, 2, 3] : (!stencil.temp<64x64x60xf64>) -> f64
    %7 = addf %5, %6 : f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %4 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
  return
}

// -----

//  CHECK-LABEL: func @simple_index(%{{.*}}: f64, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : f64) -> !stencil.temp<64x64x60xf64> {
//  CHECK: %{{.*}} = stencil.index 2 [3, 1, 4] : index
func @simple_index(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %1 = stencil.apply (%arg2 = %arg0 : f64) -> !stencil.temp<64x64x60xf64> {
    %3 = stencil.index 2 [2, -1, 1] : index
    %c20 = constant 20 : index
    %cst = constant 0.000000e+00 : f64
    %4 = cmpi "slt", %3, %c20 : index
    %5 = select %4, %arg2, %cst : f64
    %6 = stencil.store_result %5 : (f64) -> !stencil.result<f64>
    stencil.return %6 : !stencil.result<f64>
  } to ([1, 2, 3] : [65, 66, 63])
  %2 = stencil.apply (%arg2 = %arg0 : f64, %arg3 = %1 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %3 = stencil.access %arg3 [1, 2, 3] : (!stencil.temp<64x64x60xf64>) -> f64
    %4 = addf %3, %arg2 : f64
    %5 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
    stencil.return %5 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %2 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
  return
}

// -----

//  CHECK-LABEL: func @simple_ifelse(%{{.*}}: f64, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : f64) -> !stencil.temp<64x64x60xf64> {
//  CHECK-NEXT: [[TRUE:%.*]] = constant true
//  CHECK-NEXT: [[RES:%.*]] = scf.if [[TRUE]] -> (f64) {
//  CHECK-NEXT: scf.yield [[ARG0]] : f64
//  CHECK-NEXT: } else {
//  CHECK-NEXT: scf.yield [[ARG0]] : f64
//  CHECK-NEXT: }
//  CHECK-NEXT: %{{.*}} = stencil.store_result [[RES]] : (f64) -> !stencil.result<f64>
//  CHECK-NEXT: stencil.return %{{.*}} : !stencil.result<f64>
//  CHECK-NEXT: }
//  CHECK-NEXT: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
func @simple_ifelse(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %1 = stencil.apply (%arg2 = %arg0 : f64) -> !stencil.temp<64x64x60xf64> {
    %true = constant true
    %3 = scf.if %true -> (!stencil.result<f64>) {
      %4 = stencil.store_result %arg2 : (f64) -> !stencil.result<f64>
      scf.yield %4 : !stencil.result<f64>
    } else {
      %4 = stencil.store_result %arg2 : (f64) -> !stencil.result<f64>
      scf.yield %4 : !stencil.result<f64>
    }
    stencil.return %3 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  %2 = stencil.apply (%arg2 = %1 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %3 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %4 = stencil.store_result %3 : (f64) -> !stencil.result<f64>
    stencil.return %4 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %2 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
  return
}

// -----

// CHECK-LABEL: func @multiple_edges(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x64x60xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<64x64x60xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<66x64x60xf64>, [[ARG1:%.*]] = %{{.*}} : !stencil.temp<64x64x60xf64>) ->
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [-1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG1]] [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
func @multiple_edges(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %2 = stencil.cast %arg2([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %3 = stencil.load %0([-1, 0, 0] : [65, 64, 60]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x64x60xf64>
  %4:2 = stencil.apply (%arg3 = %3 : !stencil.temp<66x64x60xf64>) -> (!stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>) {
    %7 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
    %8 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
    %9 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    %10 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    stencil.return %9, %10 : !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  %5 = stencil.load %1([0, 0, 0] : [64, 64, 60]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<64x64x60xf64>
  %6 = stencil.apply (%arg3 = %3 : !stencil.temp<66x64x60xf64>, %arg4 = %4#0 : !stencil.temp<64x64x60xf64>, %arg5 = %4#1 : !stencil.temp<64x64x60xf64>, %arg6 = %5 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
    %8 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %9 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %10 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %11 = addf %7, %8 : f64
    %12 = addf %9, %10 : f64
    %13 = addf %11, %12 : f64
    %14 = stencil.store_result %13 : (f64) -> !stencil.result<f64>
    stencil.return %14 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %6 to %2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
  return
}

// -----

// CHECK-LABEL: func @avoid_redundant(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x64x60xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<66x64x60xf64>) -> 
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [-1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = addf %{{.*}}, %{{.*}} : f64
//  CHECK-NEXT: %{{.*}} = addf %{{.*}}, %{{.*}} : f64
//  CHECK-NEXT: %{{.*}} = stencil.store_result %{{.*}} : (f64) -> !stencil.result<f64>
//  CHECK-NEXT: stencil.return %{{.*}} : !stencil.result<f64>
func @avoid_redundant(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %2 = stencil.load %0([-1, 0, 0] : [65, 64, 60]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x64x60xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<66x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %5 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
    %6 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
    %7 = addf %5, %6 : f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  %4 = stencil.apply (%arg2 = %3 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %5 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %7 = addf %5, %6 : f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %4 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
  return
}

// -----

// CHECK-LABEL: func @reroute(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<67x66x63xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<67x66x63xf64>) ->
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<67x66x63xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [0, 2, 3] : (!stencil.temp<67x66x63xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [2, 2, 3] : (!stencil.temp<67x66x63xf64>) -> f64
func @reroute(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %2 = stencil.cast %arg2([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %3 = stencil.load %0([-1, 0, 0] : [66, 66, 63]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<67x66x63xf64>
  %4 = stencil.apply (%arg3 = %3 : !stencil.temp<67x66x63xf64>) -> !stencil.temp<65x66x63xf64> {
    %6 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<67x66x63xf64>) -> f64
    %7 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<67x66x63xf64>) -> f64
    %8 = addf %6, %7 : f64
    %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    stencil.return %9 : !stencil.result<f64>
  } to ([0, 0, 0] : [65, 66, 63])
  %5 = stencil.apply (%arg3 = %3 : !stencil.temp<67x66x63xf64>, %arg4 = %4 : !stencil.temp<65x66x63xf64>) -> !stencil.temp<64x64x60xf64> {
    %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<67x66x63xf64>) -> f64
    %7 = stencil.access %arg4 [1, 2, 3] : (!stencil.temp<65x66x63xf64>) -> f64
    %8 = addf %6, %7 : f64
    %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    stencil.return %9 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %4 to %1([0, 0, 0] : [65, 66, 63]) : !stencil.temp<65x66x63xf64> to !stencil.field<70x70x70xf64>
  stencil.store %5 to %2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
  return
}

// -----

// CHECK-LABEL: func @root(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<65x66x63xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<65x66x63xf64>) ->
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<65x66x63xf64>) -> f64
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [1, 2, 3] : (!stencil.temp<65x66x63xf64>) -> f64
func @root(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %2 = stencil.cast %arg2([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %3 = stencil.load %0([0, 0, 0] : [65, 66, 63]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<65x66x63xf64>
  %4 = stencil.apply (%arg3 = %3 : !stencil.temp<65x66x63xf64>) -> !stencil.temp<64x64x60xf64> {
    %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<65x66x63xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  %5 = stencil.apply (%arg3 = %3 : !stencil.temp<65x66x63xf64>) -> !stencil.temp<64x64x60xf64> {
    %6 = stencil.access %arg3 [1, 2, 3] : (!stencil.temp<65x66x63xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %4 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
  stencil.store %5 to %2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
  return
}

// -----

// CHECK-LABEL: func @dyn_access(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<68x66x62xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<68x66x62xf64>) ->
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [0, 0, -1] : (!stencil.temp<68x66x62xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.store_result %{{.*}} : (f64) -> !stencil.result<f64>
//  CHECK-NEXT: stencil.return %{{.*}} : !stencil.result<f64>
//  CHECK-NEXT: }
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG1:%.*]] = %{{.*}} : !stencil.temp<68x66x62xf64>) ->
//  CHECK-DAG: %{{.*}} = stencil.index 0 [-1, 0, 0] : index
//  CHECK-DAG: %{{.*}} = stencil.index 0 [1, 0, 0] : index
//  CHECK-DAG: %{{.*}} = stencil.dyn_access [[ARG1]](%{{.*}}, %{{.*}}, %{{.*}}) in [-2, -1, -1] : [0, 1, 1] : (!stencil.temp<68x66x62xf64>) -> f64
//  CHECK-DAG: %{{.*}} = stencil.dyn_access [[ARG1]](%{{.*}}, %{{.*}}, %{{.*}}) in [0, -1, -1] : [2, 1, 1] : (!stencil.temp<68x66x62xf64>) -> f64
//  CHECK-DAG: stencil.return %{{.*}} : !stencil.result<f64>
//  CHECK-NEXT: }
//  CHECK-NEXT: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
func @dyn_access(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %2 = stencil.load %0([-2, -1, -2] : [66, 65, 60]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<68x66x62xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<68x66x62xf64>) -> !stencil.temp<68x66x62xf64> {
    %6 = stencil.access %arg2 [0, 0, -1] : (!stencil.temp<68x66x62xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([-2, -1, -1] : [66, 65, 61])
  %4 = stencil.apply (%arg2 = %3 : !stencil.temp<68x66x62xf64>) -> !stencil.temp<66x64x60xf64> {
    %6 = stencil.index 0 [0, 0, 0] : index
    %7 = stencil.dyn_access %arg2(%6, %6, %6) in [-1, -1, -1] : [1, 1, 1] : (!stencil.temp<68x66x62xf64>) -> f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  } to ([-1, 0, 0] : [65, 64, 60])
  %5 = stencil.apply (%arg2 = %4 : !stencil.temp<66x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %6 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
    %7 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<66x64x60xf64>) -> f64
    %8 = addf %7, %6 : f64
    %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    stencil.return %9 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %5 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
  return
}

// -----

// CHECK-LABEL: func @simple_buffer(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.cast %{{.*}}([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<64x64x60xf64>
//  CHECK: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
//  CHECK: %{{.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
//  CHECK: %{{.*}} = stencil.buffer %{{.*}} : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
//  CHECK: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
//  CHECK: %{{.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
func @simple_buffer(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
  %2 = stencil.load %0([0, 0, 0] : [64, 64, 60]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<64x64x60xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  %4 = stencil.buffer %3([0, 0, 0] : [64, 64, 60]) : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
  %5 = stencil.apply (%arg2 = %4 : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<64x64x60xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %5 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
  return
}
