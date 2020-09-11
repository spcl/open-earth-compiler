

module {
  func @hdiffsa(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<0x?x0xf64>, %arg4: !stencil.field<0x?x0xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.cast %arg3([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %4 = stencil.cast %arg4([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %5 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %6 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %7 = stencil.load %3 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %8 = stencil.load %4 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %9 = stencil.apply (%arg5 = %5 : !stencil.temp<?x?x?xf64>, %arg6 = %7 : !stencil.temp<0x?x0xf64>, %arg7 = %8 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %13 = stencil.access %arg5 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg5 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = addf %13, %14 : f64
      %cst = constant -2.000000e+00 : f64
      %17 = mulf %15, %cst : f64
      %18 = addf %16, %17 : f64
      %19 = stencil.access %arg5 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg5 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %22 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %23 = subf %19, %15 : f64
      %24 = subf %20, %15 : f64
      %25 = mulf %23, %21 : f64
      %26 = mulf %24, %22 : f64
      %27 = addf %25, %18 : f64
      %28 = addf %27, %26 : f64
      %29 = stencil.store_result %28 : (f64) -> !stencil.result<f64>
      stencil.return %29 : !stencil.result<f64>
    }
    %10 = stencil.apply (%arg5 = %5 : !stencil.temp<?x?x?xf64>, %arg6 = %9 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %13 = stencil.access %arg6 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg5 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = mulf %15, %18 : f64
      %cst = constant 0.000000e+00 : f64
      %20 = cmpf "ogt", %19, %cst : f64
      %21 = select %20, %cst, %15 : f64
      %22 = stencil.store_result %21 : (f64) -> !stencil.result<f64>
      stencil.return %22 : !stencil.result<f64>
    }
    %11 = stencil.apply (%arg5 = %5 : !stencil.temp<?x?x?xf64>, %arg6 = %9 : !stencil.temp<?x?x?xf64>, %arg7 = %7 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %13 = stencil.access %arg6 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %17 = mulf %15, %16 : f64
      %18 = stencil.access %arg5 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = subf %18, %19 : f64
      %21 = mulf %17, %20 : f64
      %cst = constant 0.000000e+00 : f64
      %22 = cmpf "ogt", %21, %cst : f64
      %23 = select %22, %cst, %17 : f64
      %24 = stencil.store_result %23 : (f64) -> !stencil.result<f64>
      stencil.return %24 : !stencil.result<f64>
    }
    %12 = stencil.apply (%arg5 = %5 : !stencil.temp<?x?x?xf64>, %arg6 = %10 : !stencil.temp<?x?x?xf64>, %arg7 = %11 : !stencil.temp<?x?x?xf64>, %arg8 = %6 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %13 = stencil.access %arg6 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = addf %15, %18 : f64
      %20 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = mulf %20, %19 : f64
      %22 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = addf %21, %22 : f64
      %24 = stencil.store_result %23 : (f64) -> !stencil.result<f64>
      stencil.return %24 : !stencil.result<f64>
    }
    stencil.store %12 to %2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
