// RUN: oec-opt %s -split-input-file --convert-stencil-to-std | FileCheck %s

// CHECK-LABEL: @func_lowering
// CHECK: (%{{.*}}: memref<?x?x?xf64>) {
func @func_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECK: %{{.*}} = memref_cast %{{.*}} : memref<?x?x?xf64> to memref<777x77x7xf64>
  stencil.assert %arg0 ([0, 0, 0]:[7, 77, 777]) : !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @parallel_loop
func @parallel_loop(%arg0 : f64) attributes {stencil.program} {
  // CHECK-DAG: [[C0_1:%.*]] = constant 0 : index
  // CHECK-DAG: [[C0_2:%.*]] = constant 0 : index
  // CHECK-DAG: [[C0_3:%.*]] = constant 0 : index
  // CHECK-DAG: [[C1_1:%.*]] = constant 1 : index
  // CHECK-DAG: [[C1_2:%.*]] = constant 1 : index
  // CHECK-DAG: [[C1_3:%.*]] = constant 1 : index
  // CHECK-DAG: [[C7:%.*]] = constant 7 : index
  // CHECK-DAG: [[C77:%.*]] = constant 77 : index
  // CHECK-DAG: [[C777:%.*]] = constant 777 : index
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) = ([[C0_1]], [[C0_2]], [[C0_3]]) to ([[C7]], [[C77]], [[C777]]) step ([[C1_1]], [[C1_2]], [[C1_3]]) {  
  %0 = stencil.apply (%arg1 = %arg0 : f64) -> !stencil.temp<7x77x777xf64> {
    // CHECK-DAG:  [[IV0:%.*]] = affine.apply [[MAP0]]([[ARG0]])
    // CHECK-DAG:  [[IV1:%.*]] = affine.apply [[MAP0]]([[ARG1]])
    // CHECK-DAG:  [[IV2:%.*]] = affine.apply [[MAP0]]([[ARG2]])
    // CHECK-DAG:  [[IDX2:%.*]] = affine.apply [[MAP1]]([[IV0]], %{{.*}})
    // CHECK-DAG:  [[IDX1:%.*]] = affine.apply [[MAP1]]([[IV1]], %{{.*}})
    // CHECK-DAG:  [[IDX0:%.*]] = affine.apply [[MAP1]]([[IV2]], %{{.*}})
    // CHECK: store %{{.*}}, %{{.*}}{{\[}}[[IDX0]], [[IDX1]], [[IDX2]]] 
    stencil.return %arg1 : f64
  } to ([0, 0, 0]:[7, 77, 777])
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @parallel_loop_unroll
func @parallel_loop_unroll(%arg0 : f64) attributes {stencil.program} {
  // CHECK-DAG: [[C0_1:%.*]] = constant 0 : index
  // CHECK-DAG: [[C0_2:%.*]] = constant 0 : index
  // CHECK-DAG: [[CM1:%.*]] = constant -1 : index
  // CHECK-DAG: [[C1_1:%.*]] = constant 1 : index
  // CHECK-DAG: [[C1_2:%.*]] = constant 1 : index
  // CHECK-DAG: [[C2:%.*]] = constant 2 : index
  // CHECK-DAG: [[C7:%.*]] = constant 7 : index
  // CHECK-DAG: [[C77:%.*]] = constant 77 : index
  // CHECK-DAG: [[C777:%.*]] = constant 777 : index
  // CHECK-NEXT: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) = ([[C0_1]], [[CM1]], [[C0_2]]) to ([[C7]], [[C77]], [[C777]]) step ([[C1_1]], [[C2]], [[C1_2]]) {  
  %0 = stencil.apply (%arg1 = %arg0 : f64) -> !stencil.temp<777x78x7xf64> {
    // CHECK-DAG:  [[IV1:%.*]] = affine.apply [[MAP0]]([[ARG1]])
    // CHECK-DAG:  [[U0O1:%.*]] = constant 1 : index
    // CHECK-DAG:  [[U0IDX1:%.*]] = affine.apply [[MAP1]]([[IV1]], [[U0O1]])
    // CHECK: store %{{.*}}, %{{.*}}{{\[}}%{{.*}}, [[U0IDX1]], %{{.*}}]
    // CHECK-DAG:  [[U1O1:%.*]] = constant 2 : index
    // CHECK-DAG:  [[U1IDX1:%.*]] = affine.apply [[MAP1]]([[IV1]], [[U1O1]])
    // CHECK: store %{{.*}}, %{{.*}}{{\[}}%{{.*}}, [[U1IDX1]], %{{.*}}]
    stencil.return unroll [1, 2, 1] %arg1, %arg1 : f64, f64
  } to ([0, -1, 0]:[7, 77, 777])
  return
}

// -----

// CHECK-LABEL: @alloc_temp
func @alloc_temp(%arg0 : f64) attributes {stencil.program} {
  // CHECK: [[TEMP:%.*]] = alloc() : memref<7x7x7xf64>
  %0 = stencil.apply (%arg1 = %arg0 : f64) -> !stencil.temp<7x7x7xf64> {
    // CHECK: store %{{.*}}, [[TEMP]]  
    stencil.return %arg1 : f64
  } to ([0, 0, 0]:[7, 7, 7]) 
  %1 = stencil.apply (%arg1 = %0 : !stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64> {
    // CHECK: load [[TEMP]]  
    %4 = stencil.access %arg1[0,0,0] : (!stencil.temp<7x7x7xf64>) -> f64
    stencil.return %4 : f64
  } to ([0, 0, 0]:[7, 7, 7])
  // CHECK: dealloc [[TEMP]] : memref<7x7x7xf64>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @access_lowering
func @access_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0 ([0, 0, 0]:[10, 10, 10]) : !stencil.field<?x?x?xf64>
  // CHECK: [[VIEW:%.*]] = subview %{{.*}}[0, 0, 0] [10, 10, 10] [1, 1, 1]
  %0 = stencil.load %arg0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<10x10x10xf64>
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %1 = stencil.apply (%arg1 = %0 : !stencil.temp<10x10x10xf64>) -> !stencil.temp<7x7x7xf64> {
    // CHECK-DAG: [[IV0:%.*]] = affine.apply [[MAP0]]([[ARG0]])
    // CHECK-DAG: [[IV1:%.*]] = affine.apply [[MAP0]]([[ARG1]])
    // CHECK-DAG: [[IV2:%.*]] = affine.apply [[MAP0]]([[ARG2]])
    // CHECK-DAG: [[C0:%.*]] = constant 0 : index
    // CHECK-DAG: [[O0:%.*]] = affine.apply [[MAP1]]([[IV0]], [[C0]])
    // CHECK-DAG: [[C1:%.*]] = constant 1 : index
    // CHECK-DAG: [[O1:%.*]] = affine.apply [[MAP1]]([[IV1]], [[C1]])
    // CHECK-DAG: [[C2:%.*]] = constant 2 : index
    // CHECK-DAG: [[O2:%.*]] = affine.apply [[MAP1]]([[IV2]], [[C2]])
    // CHECK: %{{.*}} = load [[VIEW:%.*]]{{\[}}[[O2]], [[O1]], [[O0]]{{[]]}}
    %2 = stencil.access %arg1[0, 1, 2] : (!stencil.temp<10x10x10xf64>) -> f64
    stencil.return %2 : f64
  } to ([0, 0, 0]:[7, 7, 7])
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @index_lowering
func @index_lowering(%arg0 : f64) attributes {stencil.program} {
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %0 = stencil.apply (%arg1 = %arg0 : f64) -> !stencil.temp<7x7x7xf64> {
    // CHECK-DAG: [[IV2:%.*]] = affine.apply [[MAP0]]([[ARG2]])
    // CHECK-DAG: [[C0:%.*]] = constant 2 : index
    // CHECK-DAG: [[O0:%.*]] = affine.apply [[MAP1]]([[IV2]], [[C0]])
    %1 = stencil.index 2 [0, 1, 2] : index
    %2 = constant 0 : index
    %3 = constant 0.0 : f64
    %4 = cmpi "slt", %1, %2 : index
    %5 = select %4, %arg1, %3 : f64
    stencil.return %5 : f64
  } to ([0, 0, 0]:[7, 7, 7])
  return
}

// -----

// CHECK-LABEL: @return_lowering
func @return_lowering(%arg0: f64) attributes {stencil.program} {
  // CHECK: scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}) =
  %0:2 = stencil.apply (%arg1 = %arg0 : f64) -> (!stencil.temp<7x7x7xf64>, !stencil.temp<7x7x7xf64>) {
    // CHECK-COUNT-2: store %{{.*}}, %{{.*}}{{\[}}%{{.*}}, %{{.*}}, %{{.*}} : memref<7x7x7xf64> 
    stencil.return %arg1, %arg1 : f64, f64
  } to ([0, 0, 0]:[7, 7, 7])
  return
}

// -----

// CHECK-LABEL: @load_lowering
func @load_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0 ([0, 0, 0]:[11, 12, 13]) : !stencil.field<?x?x?xf64>
  // CHECK: %{{.*}} = subview %{{.*}}[3, 2, 1] [9, 9, 9] [1, 1, 1] : memref<13x12x11xf64> to memref<9x9x9xf64, #map{{[0-9]+}}>
  %0 = stencil.load %arg0 ([1, 2, 3]:[10, 11, 12]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<9x9x9xf64>
  return
}

// -----

// CHECK-LABEL: @store_lowering
func @store_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0 ([0, 0, 0]:[10, 10, 10]) : !stencil.field<?x?x?xf64>
  // CHECK: [[VIEW:%.*]] = subview %{{.*}}[3, 2, 1] [7, 7, 7] [1, 1, 1] : memref<10x10x10xf64> to memref<7x7x7xf64, #map{{[0-9]+}}>
  %cst = constant 1.0 : f64
  %0 = stencil.apply (%arg1 = %cst : f64) -> !stencil.temp<7x7x7xf64> {
    // CHECK: store %{{.*}} [[VIEW]]
    stencil.return %arg1 : f64
  } to ([0, 0, 0]:[7, 7, 7]) 
  stencil.store %0 to %arg0 ([1, 2, 3]:[8, 9, 10]) : !stencil.temp<7x7x7xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: @if_lowering
func @if_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0 ([0, 0, 0]:[10, 10, 10]) : !stencil.field<?x?x?xf64>
  %0 = stencil.load %arg0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<10x10x10xf64>
  %1 = stencil.apply (%arg1 = %0 : !stencil.temp<10x10x10xf64>) -> !stencil.temp<7x7x7xf64> {
    %2 = constant 1 : i1
    // CHECK: [[RES:%.*]] = scf.if %{{.*}} -> (f64) {
    %3 = scf.if %2 -> (f64) {
      // CHECK: [[IF:%.*]] = load
      %4 = stencil.access %arg1[0, 1, 2] : (!stencil.temp<10x10x10xf64>) -> f64
      // CHECK: scf.yield [[IF]] : f64
      scf.yield %4 : f64
    // CHECK: } else {
    } else {
      // CHECK: [[ELSE:%.*]] = load
      %5 = stencil.access %arg1[0, 2, 1] : (!stencil.temp<10x10x10xf64>) -> f64
      // CHECK: scf.yield [[ELSE]] : f64
      scf.yield %5 : f64
    }
    // CHECK: store [[RES]]
    stencil.return %3 : f64
  } to ([0, 0, 0]:[7, 7, 7])
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @lowerdim
// CHECK: (%{{.*}}: memref<?x?xf64>) {
func @lowerdim(%arg0: !stencil.field<?x?x0xf64>) attributes {stencil.program} {
  // CHECK: %{{.*}} = memref_cast %{{.*}} : memref<?x?xf64> to memref<11x10xf64>
  stencil.assert %arg0 ([0, 0, 0]:[10, 11, 12]) : !stencil.field<?x?x0xf64>
  // CHECK: [[VIEW:%.*]] = subview %{{.*}}[0, 0] [8, 7] [1, 1] : memref<11x10xf64> to memref<8x7xf64, #map{{[0-9]+}}>
  %0 = stencil.load %arg0 ([0, 0, 0]:[7, 8, 9]) : (!stencil.field<?x?x0xf64>) -> !stencil.temp<7x8x0xf64>
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %1 = stencil.apply (%arg1 = %0 : !stencil.temp<7x8x0xf64>) -> !stencil.temp<7x8x9xf64> {
    // CHECK-DAG: [[IV0:%.*]] = affine.apply [[MAP0]]([[ARG0]])
    // CHECK-DAG: [[IV1:%.*]] = affine.apply [[MAP0]]([[ARG1]])
    // CHECK-DAG: [[C0:%.*]] = constant 0 : index
    // CHECK-DAG: [[O0:%.*]] = affine.apply [[MAP1]]([[IV0]], [[C0]])
    // CHECK-DAG: [[C1:%.*]] = constant 1 : index
    // CHECK-DAG: [[O1:%.*]] = affine.apply [[MAP1]]([[IV1]], [[C1]])
    // CHECK: %{{.*}} = load [[VIEW:%.*]]{{\[}}[[O1]], [[O0]]{{[]]}}
    %2 = stencil.access %arg1[0, 1, 2]: (!stencil.temp<7x8x0xf64>) -> f64
    stencil.return %2 : f64
  } to ([0, 0, 0]:[7, 7, 7])
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK: [[MAP2:#map[0-9]+]] = affine_map<(d0, d1, d2) -> (d0 + d1 - d2 - 1)>

// CHECK-LABEL: @sequential_loop
func @sequential_loop(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0 ([0, 0, 0]:[10, 10, 10]) : !stencil.field<?x?x?xf64>
  %0 = stencil.load %arg0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<10x10x10xf64>
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  // CHECK: scf.for [[ARG3:%.*]] =
  %1 = stencil.apply seq(dim = 2, range = 0 to 7, dir = 1) (%arg1 = %0 : !stencil.temp<10x10x10xf64>) -> !stencil.temp<7x7x7xf64> {
    // CHECK-DAG: [[IV2:%.*]] = affine.apply [[MAP0]]([[ARG3]])
    // CHECK-DAG: [[C2:%.*]] = constant 2 : index
    // CHECK-DAG: [[O2:%.*]] = affine.apply [[MAP1]]([[IV2]], [[C2]])
    %2 = stencil.access %arg1[0, 1, 2] : (!stencil.temp<10x10x10xf64>) -> f64
    stencil.return %2 : f64
  } to ([0, 0, 0]:[7, 7, 7])
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  // CHECK: scf.for [[ARG3:%.*]] = [[LB:%.*]] to [[UB:%.*]]
  %3 = stencil.apply seq(dim = 2, range = 0 to 7, dir = -1) (%arg1 = %0 : !stencil.temp<10x10x10xf64>) -> !stencil.temp<7x7x7xf64> {
    // CHECK-DAG: [[IV2:%.*]] = affine.apply [[MAP2]]([[UB:%.*]], [[LB:%.*]], [[ARG3]])
    // CHECK-DAG: [[C2:%.*]] = constant 2 : index
    // CHECK-DAG: [[O2:%.*]] = affine.apply [[MAP1]]([[IV2]], [[C2]])
    %4 = stencil.access %arg1[0, 1, 2] : (!stencil.temp<10x10x10xf64>) -> f64
    stencil.return %4 : f64
  } to ([0, 0, 0]:[7, 7, 7])
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @depend_op
func @depend_op(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0 ([0, 0, 0]:[10, 10, 10]) : !stencil.field<?x?x?xf64>
  %0 = stencil.load %arg0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<10x10x10xf64>
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  // CHECK: scf.for [[ARG3:%.*]] =
  %1 = stencil.apply seq(dim = 2, range = 0 to 7, dir = 1) (%arg1 = %0 : !stencil.temp<10x10x10xf64>) -> !stencil.temp<7x7x7xf64> {
    // CHECK-DAG: [[IV2:%.*]] = affine.apply [[MAP0]]([[ARG3]])
    // CHECK-DAG: [[C2:%.*]] = constant -1 : index
    // CHECK-DAG: [[O2:%.*]] = affine.apply [[MAP1]]([[IV2]], [[C2]])
    // CHECK: %{{.*}} = load %{{.*}}{{\[}}[[O2]], %{{.*}}, %{{.*}}{{[]]}}
    %2 = stencil.depend 0 [0, 0, -1] : f64
    stencil.return %2 : f64
  } to ([0, 0, 0]:[7, 7, 7])
  return
}