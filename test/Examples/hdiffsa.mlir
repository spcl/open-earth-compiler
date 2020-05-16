

module {
  func @hdiffsa(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<0x?x0xf64>, %arg4: !stencil.field<0x?x0xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg3 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %3 = stencil.load %arg4 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %4 = stencil.apply (%arg5 = %0 : !stencil.temp<?x?x?xf64>, %arg6 = %2 : !stencil.temp<0x?x0xf64>, %arg7 = %3 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %8 = stencil.access %arg5 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg5 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = addf %8, %9 : f64
      %cst = constant -2.000000e+00 : f64
      %12 = mulf %10, %cst : f64
      %13 = addf %11, %12 : f64
      %14 = stencil.access %arg5 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg5 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %17 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %18 = subf %14, %10 : f64
      %19 = subf %15, %10 : f64
      %20 = mulf %18, %16 : f64
      %21 = mulf %19, %17 : f64
      %22 = addf %20, %13 : f64
      %23 = addf %22, %21 : f64
      stencil.return %23 : f64
    }
    %5 = stencil.apply (%arg5 = %0 : !stencil.temp<?x?x?xf64>, %arg6 = %4 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %8 = stencil.access %arg6 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = subf %8, %9 : f64
      %11 = stencil.access %arg5 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = subf %11, %12 : f64
      %14 = mulf %10, %13 : f64
      %cst = constant 0.000000e+00 : f64
      %15 = cmpf "ogt", %14, %cst : f64
      %16 = select %15, %cst, %10 : f64
      stencil.return %16 : f64
    }
    %6 = stencil.apply (%arg5 = %0 : !stencil.temp<?x?x?xf64>, %arg6 = %4 : !stencil.temp<?x?x?xf64>, %arg7 = %2 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %8 = stencil.access %arg6 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = subf %8, %9 : f64
      %11 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %12 = mulf %10, %11 : f64
      %13 = stencil.access %arg5 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = mulf %12, %15 : f64
      %cst = constant 0.000000e+00 : f64
      %17 = cmpf "ogt", %16, %cst : f64
      %18 = select %17, %cst, %12 : f64
      stencil.return %18 : f64
    }
    %7 = stencil.apply (%arg5 = %0 : !stencil.temp<?x?x?xf64>, %arg6 = %5 : !stencil.temp<?x?x?xf64>, %arg7 = %6 : !stencil.temp<?x?x?xf64>, %arg8 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %8 = stencil.access %arg6 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = subf %8, %9 : f64
      %11 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = subf %11, %12 : f64
      %14 = addf %10, %13 : f64
      %15 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = mulf %15, %14 : f64
      %17 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %16, %17 : f64
      stencil.return %18 : f64
    }
    stencil.store %7 to %arg2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
