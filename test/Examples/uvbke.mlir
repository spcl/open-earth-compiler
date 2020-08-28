

module {
  func @uvbke(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg2 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.load %arg3 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %4 = stencil.apply (%arg7 = %0 : !stencil.temp<?x?x?xf64>, %arg8 = %1 : !stencil.temp<?x?x?xf64>, %arg9 = %2 : !stencil.temp<?x?x?xf64>, %arg10 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %dt5 = constant 112.5 : f64
      %6 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = addf %6, %7 : f64
      %9 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = mulf %8, %9 : f64
      %11 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = addf %11, %12 : f64
      %14 = subf %13, %10 : f64
      %15 = mulf %dt5, %14 : f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %16, %15 : f64
      stencil.return %17 : f64
    }
    %5 = stencil.apply (%arg7 = %0 : !stencil.temp<?x?x?xf64>, %arg8 = %1 : !stencil.temp<?x?x?xf64>, %arg9 = %2 : !stencil.temp<?x?x?xf64>, %arg10 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %dt5 = constant 112.5 : f64
      %6 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = addf %6, %7 : f64
      %9 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = mulf %8, %9 : f64
      %11 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = addf %11, %12 : f64
      %14 = subf %13, %10 : f64
      %15 = mulf %dt5, %14 : f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %16, %15 : f64
      stencil.return %17 : f64
    }
    stencil.store %4 to %arg4([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %5 to %arg5([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
