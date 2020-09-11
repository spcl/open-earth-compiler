

module {
  func @uvbke(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.cast %arg3([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %4 = stencil.cast %arg4([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %5 = stencil.cast %arg5([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %6 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %7 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %8 = stencil.load %2 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %9 = stencil.load %3 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %10 = stencil.apply (%arg6 = %6 : !stencil.temp<?x?x?xf64>, %arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %9 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.125000e+02 : f64
      %12 = stencil.access %arg7 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = addf %12, %13 : f64
      %15 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = mulf %14, %15 : f64
      %17 = stencil.access %arg6 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = addf %17, %18 : f64
      %20 = subf %19, %16 : f64
      %21 = mulf %cst, %20 : f64
      %22 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = mulf %22, %21 : f64
      %24 = stencil.store_result %23 : (f64) -> !stencil.result<f64>
      stencil.return %24 : !stencil.result<f64>
    }
    %11 = stencil.apply (%arg6 = %6 : !stencil.temp<?x?x?xf64>, %arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %9 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.125000e+02 : f64
      %12 = stencil.access %arg6 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = addf %12, %13 : f64
      %15 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = mulf %14, %15 : f64
      %17 = stencil.access %arg7 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = addf %17, %18 : f64
      %20 = subf %19, %16 : f64
      %21 = mulf %cst, %20 : f64
      %22 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = mulf %22, %21 : f64
      %24 = stencil.store_result %23 : (f64) -> !stencil.result<f64>
      stencil.return %24 : !stencil.result<f64>
    }
    stencil.store %10 to %4([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %11 to %5([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
