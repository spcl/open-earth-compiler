// RUN: oec-opt %s --canonicalize | oec-opt | FileCheck %s

module {
  func @canonical(%arg0: !stencil.field<ijk,f64>, %arg1: !stencil.field<ijk,f64>, %arg2: !stencil.field<ijk,f64>) attributes {stencil.program} {
    stencil.assert %arg0 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
    stencil.assert %arg1 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
    stencil.assert %arg2 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
    %0 = stencil.load %arg0 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %1,%2 = stencil.apply %arg3 = %0, %arg4 = %0 : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %3 = stencil.access %arg3[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %4 = stencil.access %arg4[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %5 = addf %3, %4 : f64
      stencil.return %5, %5  : f64, f64
    } : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>
    stencil.store %1 to %arg1 ([0, 0, 0]:[64, 64, 60]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
    stencil.store %2 to %arg2 ([0, 0, 0]:[64, 64, 60]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
    return
  }
}

//       CHECK: %{{.*}} = stencil.apply %{{.*}} = %{{.*}} : !stencil.temp<ijk,f64> {
//       CHECK: stencil.return %{{.*}} : f64
