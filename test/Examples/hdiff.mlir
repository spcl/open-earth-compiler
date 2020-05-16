

module {
  func @hdiff(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %6 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = stencil.access %arg3 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg3 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = addf %6, %7 : f64
      %12 = addf %8, %9 : f64
      %13 = addf %11, %12 : f64
      %cst = constant -4.000000e+00 : f64
      %14 = mulf %10, %cst : f64
      %15 = addf %14, %13 : f64
      stencil.return %15 : f64
    }
    %3 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %6 = stencil.access %arg4 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = subf %6, %7 : f64
      %9 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = subf %9, %10 : f64
      %12 = mulf %8, %11 : f64
      %cst = constant 0.000000e+00 : f64
      %13 = cmpf "ogt", %12, %cst : f64
      %14 = select %13, %cst, %8 : f64
      stencil.return %14 : f64
    }
    %4 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %6 = stencil.access %arg4 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = subf %6, %7 : f64
      %9 = stencil.access %arg3 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = subf %9, %10 : f64
      %12 = mulf %8, %11 : f64
      %cst = constant 0.000000e+00 : f64
      %13 = cmpf "ogt", %12, %cst : f64
      %14 = select %13, %cst, %8 : f64
      stencil.return %14 : f64
    }
    %5 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %3 : !stencil.temp<?x?x?xf64>, %arg5 = %4 : !stencil.temp<?x?x?xf64>, %arg6 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %6 = stencil.access %arg4 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = subf %6, %7 : f64
      %9 = stencil.access %arg5 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = subf %9, %10 : f64
      %12 = addf %8, %11 : f64
      %13 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = mulf %13, %12 : f64
      %15 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = addf %14, %15 : f64
      stencil.return %16 : f64
    }
    stencil.store %5 to %arg2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
