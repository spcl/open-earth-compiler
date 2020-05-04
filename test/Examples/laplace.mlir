

module {
  func @laplace(%arg0: !stencil.field<ijk,f64>, %arg1: !stencil.field<ijk,f64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    %0 = stencil.load %arg0 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %1 = stencil.apply (%arg2 = %0 : !stencil.temp<ijk,f64>) -> !stencil.temp<ijk,f64> {
      %2 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %3 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %4 = stencil.access %arg2 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %5 = stencil.access %arg2 [0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %7 = addf %2, %3 : f64
      %8 = addf %4, %5 : f64
      %9 = addf %7, %8 : f64
      %cst = constant -4.000000e+00 : f64
      %10 = mulf %6, %cst : f64
      %11 = addf %10, %9 : f64
      stencil.return %11 : f64
    }
    stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
    return
  }
}
