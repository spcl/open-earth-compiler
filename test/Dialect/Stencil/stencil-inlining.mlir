// RUN: oec-opt %s -split-input-file --stencil-inlining -cse | oec-opt | FileCheck %s

// CHECK-LABEL: func @simple(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [0, 2, 3] : (!stencil.temp<?x?x?xf64>) -> f64
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [2, 2, 3] : (!stencil.temp<?x?x?xf64>) -> f64
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %1 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5 : f64
  }
  %2 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x?xf64>, %arg3 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg3 [1, 2, 3] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5 : f64
  }
  stencil.store %2 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

//  CHECK-LABEL: func @simple_index(%{{.*}}: f64, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : f64) -> !stencil.temp<?x?x?xf64> {
//   CHECK: %{{.*}} = stencil.index 2 [3, 1, 4] : index
func @simple_index(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  %0 = stencil.apply (%arg2 = %arg0 : f64) -> !stencil.temp<?x?x?xf64> {
    %1 = stencil.index 2 [2, -1, 1] : index
    %2 = constant 20 : index
    %3 = constant 0.0 : f64
    %4 = cmpi "slt", %1, %2 : index
    %5 = select %4, %arg2, %3 : f64
    stencil.return %5 : f64
  }
  %1 = stencil.apply (%arg3 = %arg0 : f64, %arg4 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %2 = stencil.access %arg4 [1, 2, 3] : (!stencil.temp<?x?x?xf64>) -> f64
    %3 = addf %2, %arg3 : f64
    stencil.return %3 : f64
  }
  stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @multiple_edges(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<?x?x?xf64>, [[ARG1:%.*]] = %{{.*}} : !stencil.temp<?x?x?xf64>) ->
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG1]] [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
func @multiple_edges(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg2([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %1:2 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
    %4 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    stencil.return %4, %5 : f64, f64
  }
  %2 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %3 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %1#0 : !stencil.temp<?x?x?xf64>, %arg5 = %1#1 : !stencil.temp<?x?x?xf64>, %arg6 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %4 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %8 = addf %4, %5 : f64
    %9 = addf %6, %7 : f64
    %10 = addf %8, %9 : f64
    stencil.return %10 : f64
  }
  stencil.store %3 to %arg2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @avoid_redundant(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<?x?x?xf64>) -> 
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access [[ARG0]] [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//  CHECK-NEXT: %{{.*}} = addf %{{.*}}, %{{.*}} : f64
//  CHECK-NEXT: %{{.*}} = addf %{{.*}}, %{{.*}} : f64
//  CHECK-NEXT: stencil.return %{{.*}} : f64
func @avoid_redundant(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %1 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5 : f64
  }
  %2 = stencil.apply (%arg2 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5 : f64
  }
  stencil.store %2 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @reroute(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<?x?x?xf64>) ->
//       CHECK: %{{.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//       CHECK: %{{.*}} = stencil.access [[ARG0]] [0, 2, 3] : (!stencil.temp<?x?x?xf64>) -> f64
//       CHECK: %{{.*}} = stencil.access [[ARG0]] [2, 2, 3] : (!stencil.temp<?x?x?xf64>) -> f64
func @reroute(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg2([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %1 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5 : f64
  }
  %2 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg4 [1, 2, 3] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5 : f64
  }
  stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  stencil.store %2 to %arg2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @root(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: stencil.assert %{{.*}}([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply ([[ARG0:%.*]] = %{{.*}} : !stencil.temp<?x?x?xf64>) ->
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//   CHECK-DAG: %{{.*}} = stencil.access [[ARG0]] [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
func @root(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg2([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %1 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5 : f64
  }
  %2 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    stencil.return %3 : f64
  }
  stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  stencil.store %2 to %arg2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}
