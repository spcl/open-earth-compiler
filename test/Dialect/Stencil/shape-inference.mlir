// RUN: oec-opt %s -split-input-file --stencil-shape-inference | oec-opt | FileCheck %s

// -----

// CHECK-LABEL: func @simple(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<66x68x60xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %1 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5 : f64
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<?x?x?xf64>
  stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @unroll(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @unroll(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<66x68x60xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %1 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    %6 = stencil.access %arg2 [-1, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.access %arg2 [1, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %8 = addf %6, %7 : f64
    stencil.return unroll [1, 2, 1] %5, %8 : f64, f64
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<?x?x?xf64>
  stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @multiple(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @multiple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.temp<66x68x60xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x60xf64>) -> !stencil.temp<66x64x60xf64> {
  %1 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg2 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %4 = stencil.access %arg2 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5 : f64
  //  CHECK: } to ([-1, 0, 0] : [65, 64, 60])
  }
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %2 = stencil.apply (%arg2 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %6 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %8 = addf %6, %7 : f64
    stencil.return %8 : f64
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<?x?x?xf64>
  stencil.store %2 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @lower(%{{.*}}: !stencil.field<?x?x0xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @lower(%arg0: !stencil.field<?x?x0xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x0xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<?x?x0xf64>) -> !stencil.temp<66x68x0xf64>
  %0 = stencil.load %arg0 : (!stencil.field<?x?x0xf64>) -> !stencil.temp<?x?x0xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x0xf64>) -> !stencil.temp<64x64x60xf64> {
  %1 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
    %3 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<?x?x0xf64>) -> f64
    %4 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<?x?x0xf64>) -> f64
    %5 = addf %3, %4 : f64
    stencil.return %5 : f64
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<?x?x?xf64>
  stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}

// -----

// CHECK-LABEL: func @twostores(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @twostores(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  stencil.assert %arg0([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  stencil.assert %arg1([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  //  CHECK: %{{.*}}:2 = stencil.apply -> (!stencil.temp<64x66x60xf64>, !stencil.temp<64x66x60xf64>) {
  %1,%2 = stencil.apply -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
    %3 = constant 1.0 : f64
    stencil.return %3, %3 : f64, f64
  //  CHECK: } to ([0, -1, 0] : [64, 65, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, -1, 0] : [64, 65, 60]) : !stencil.temp<64x66x60xf64> to !stencil.field<?x?x?xf64>
  stencil.store %1 to %arg0([0, 0, 0] : [64, 65, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, -1, 0] : [64, 65, 60]) : !stencil.temp<64x66x60xf64> to !stencil.field<?x?x?xf64>
  stencil.store %2 to %arg1([0, -1, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
  return
}