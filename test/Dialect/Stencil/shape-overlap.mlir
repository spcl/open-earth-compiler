// RUN: oec-opt %s -split-input-file --stencil-shape-overlap | oec-opt | FileCheck %s

// CHECK-LABEL: func @ioverlap(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @ioverlap(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.cast %arg2([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %3 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  %4 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  }
  %5 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  }
  stencil.store %4 to %1([-1, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  stencil.store %5 to %2([0, 0, 0] : [65, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// func @test(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
//   %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
//   %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
//   %2 = stencil.cast %arg2([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
//   %3 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
//   %4 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
//     %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//     %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
//     stencil.return %7 : !stencil.result<f64>
//   }
//   %5 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
//     %6 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
//     %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
//     stencil.return %7 : !stencil.result<f64>
//   }


//   %6,%7 = stencil.combine 0 at 0 lower = (%4 : !stencil.temp<?x?x?xf64>) 
//                               upper = (%4 : !stencil.temp<?x?x?xf64>)
//                               upperext = (%5 : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>

//   %8,%9 = stencil.combine 0 at 64 lower = (%7 : !stencil.temp<?x?x?xf64>) 
//                               upper = (%5 : !stencil.temp<?x?x?xf64>)
//                               lowerext = (%6 : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
  
//   stencil.store %9 to %1([-1, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
//   stencil.store %8 to %2([0, 0, 0] : [65, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
//   return
// }