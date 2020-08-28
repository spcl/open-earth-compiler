// RUN: oec-opt %s -split-input-file --stencil-shape-inference | oec-opt | FileCheck %s

// -----

// CHECK-LABEL: func @simple(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x68x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %4 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = addf %4, %5 : f64
    stencil.return %6 : f64
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @unroll(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @unroll(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x68x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %4 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = addf %4, %5 : f64
    %7 = stencil.access %arg2 [-1, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %8 = stencil.access %arg2 [1, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %9 = addf %7, %8 : f64
    stencil.return unroll [1, 2, 1] %6, %9 : f64, f64
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @multiple(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @multiple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x68x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x60xf64>) -> !stencil.temp<66x64x60xf64> {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %5 = stencil.access %arg2 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = stencil.access %arg2 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = addf %5, %6 : f64
    stencil.return %7 : f64
  //  CHECK: } to ([-1, 0, 0] : [65, 64, 60])
  }
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %4 = stencil.apply (%arg2 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %8 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %9 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %10 = addf %8, %9 : f64
    stencil.return %10 : f64
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %4 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @lower(%{{.*}}: !stencil.field<?x?x0xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @lower(%arg0: !stencil.field<?x?x0xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x0xf64>) -> !stencil.field<70x70x0xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x0xf64>) -> !stencil.temp<66x68x0xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x0xf64>) -> !stencil.temp<?x?x0xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x0xf64>) -> !stencil.temp<64x64x60xf64> {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
    %4 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<?x?x0xf64>) -> f64
    %5 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<?x?x0xf64>) -> f64
    %6 = addf %4, %5 : f64
    stencil.return %6 : f64
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @twostores(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @twostores(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  //  CHECK: %{{.*}}:2 = stencil.apply -> (!stencil.temp<64x66x60xf64>, !stencil.temp<64x66x60xf64>) {
  %2,%3 = stencil.apply -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
    %4 = constant 1.0 : f64
    stencil.return %4, %4 : f64, f64
  //  CHECK: } to ([0, -1, 0] : [64, 65, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, -1, 0] : [64, 65, 60]) : !stencil.temp<64x66x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %2 to %0([0, 0, 0] : [64, 65, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, -1, 0] : [64, 65, 60]) : !stencil.temp<64x66x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %1([0, -1, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}