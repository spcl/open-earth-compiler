

module {
  func @hdiff(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %4 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %5 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %9 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = stencil.access %arg3 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = stencil.access %arg3 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = addf %9, %10 : f64
      %15 = addf %11, %12 : f64
      %16 = addf %14, %15 : f64
      %cst = constant -4.000000e+00 : f64
      %17 = mulf %13, %cst : f64
      %18 = addf %17, %16 : f64
      stencil.return %18 : f64
    }
    %6 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>, %arg4 = %5 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %9 = stencil.access %arg4 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = subf %9, %10 : f64
      %12 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = subf %12, %13 : f64
      %15 = mulf %11, %14 : f64
      %cst = constant 0.000000e+00 : f64
      %16 = cmpf "ogt", %15, %cst : f64
      %17 = select %16, %cst, %11 : f64
      stencil.return %17 : f64
    }
    %7 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>, %arg4 = %5 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %9 = stencil.access %arg4 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = subf %9, %10 : f64
      %12 = stencil.access %arg3 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = subf %12, %13 : f64
      %15 = mulf %11, %14 : f64
      %cst = constant 0.000000e+00 : f64
      %16 = cmpf "ogt", %15, %cst : f64
      %17 = select %16, %cst, %11 : f64
      stencil.return %17 : f64
    }
    %8 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>, %arg4 = %6 : !stencil.temp<?x?x?xf64>, %arg5 = %7 : !stencil.temp<?x?x?xf64>, %arg6 = %4 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %9 = stencil.access %arg4 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = subf %9, %10 : f64
      %12 = stencil.access %arg5 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = subf %12, %13 : f64
      %15 = addf %11, %14 : f64
      %16 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %16, %15 : f64
      %18 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = addf %17, %18 : f64
      stencil.return %19 : f64
    }
    stencil.store %8 to %2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
