

module {
  func @hdiffsmag(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<0x?x0xf64>, %arg6: !stencil.field<0x?x0xf64>, %arg7: !stencil.field<0x?x0xf64>, %arg8: !stencil.field<0x?x0xf64>, %arg9: !stencil.field<0x?x0xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.cast %arg3([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %4 = stencil.cast %arg4([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %5 = stencil.cast %arg5([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %6 = stencil.cast %arg6([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %7 = stencil.cast %arg7([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %8 = stencil.cast %arg8([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %9 = stencil.cast %arg9([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %10 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %11 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %12 = stencil.load %2 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %13 = stencil.load %5 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %14 = stencil.load %6 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %15 = stencil.load %7 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %16 = stencil.load %8 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %17 = stencil.load %9 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %18 = stencil.apply (%arg10 = %10 : !stencil.temp<?x?x?xf64>, %arg11 = %11 : !stencil.temp<?x?x?xf64>, %arg12 = %17 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 6.371229e+06 : f64
      %cst_1 = constant 4.8828125E-4 : f64
      %cst_2 = constant 7.32421875E-4 : f64
      %24 = divf %cst, %cst_0 : f64
      %25 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %26 = mulf %25, %cst_2 : f64
      %27 = mulf %cst_1, %24 : f64
      %28 = stencil.access %arg11 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = subf %28, %29 : f64
      %31 = mulf %30, %27 : f64
      %32 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = subf %32, %33 : f64
      %35 = mulf %34, %26 : f64
      %36 = subf %31, %35 : f64
      %37 = mulf %36, %36 : f64
      %38 = stencil.store_result %37 : (f64) -> !stencil.result<f64>
      stencil.return %38 : !stencil.result<f64>
    }
    %19 = stencil.apply (%arg10 = %10 : !stencil.temp<?x?x?xf64>, %arg11 = %11 : !stencil.temp<?x?x?xf64>, %arg12 = %17 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 6.371229e+06 : f64
      %cst_1 = constant 4.8828125E-4 : f64
      %cst_2 = constant 7.32421875E-4 : f64
      %24 = divf %cst, %cst_0 : f64
      %25 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %26 = mulf %25, %cst_2 : f64
      %27 = mulf %cst_1, %24 : f64
      %28 = stencil.access %arg11 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = subf %28, %29 : f64
      %31 = mulf %30, %26 : f64
      %32 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = subf %32, %33 : f64
      %35 = mulf %34, %27 : f64
      %36 = addf %35, %31 : f64
      %37 = mulf %36, %36 : f64
      %38 = stencil.store_result %37 : (f64) -> !stencil.result<f64>
      stencil.return %38 : !stencil.result<f64>
    }
    %20 = stencil.apply (%arg10 = %10 : !stencil.temp<?x?x?xf64>, %arg11 = %15 : !stencil.temp<0x?x0xf64>, %arg12 = %16 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %24 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %24, %25 : f64
      %cst = constant -2.000000e+00 : f64
      %28 = mulf %26, %cst : f64
      %29 = addf %27, %28 : f64
      %30 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %33 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %34 = subf %30, %26 : f64
      %35 = subf %31, %26 : f64
      %36 = mulf %34, %32 : f64
      %37 = mulf %35, %33 : f64
      %38 = addf %36, %29 : f64
      %39 = addf %38, %37 : f64
      %40 = stencil.store_result %39 : (f64) -> !stencil.result<f64>
      stencil.return %40 : !stencil.result<f64>
    }
    %21 = stencil.apply (%arg10 = %11 : !stencil.temp<?x?x?xf64>, %arg11 = %13 : !stencil.temp<0x?x0xf64>, %arg12 = %14 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %24 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %24, %25 : f64
      %cst = constant -2.000000e+00 : f64
      %28 = mulf %26, %cst : f64
      %29 = addf %27, %28 : f64
      %30 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %33 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %34 = subf %30, %26 : f64
      %35 = subf %31, %26 : f64
      %36 = mulf %34, %32 : f64
      %37 = mulf %35, %33 : f64
      %38 = addf %36, %29 : f64
      %39 = addf %38, %37 : f64
      %40 = stencil.store_result %39 : (f64) -> !stencil.result<f64>
      stencil.return %40 : !stencil.result<f64>
    }
    %22 = stencil.apply (%arg10 = %10 : !stencil.temp<?x?x?xf64>, %arg11 = %18 : !stencil.temp<?x?x?xf64>, %arg12 = %19 : !stencil.temp<?x?x?xf64>, %arg13 = %20 : !stencil.temp<?x?x?xf64>, %arg14 = %12 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e-02 : f64
      %cst_0 = constant 2.500000e-02 : f64
      %24 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = mulf %cst_0, %24 : f64
      %cst_1 = constant 5.000000e-01 : f64
      %cst_2 = constant 0.000000e+00 : f64
      %26 = stencil.access %arg11 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = addf %26, %27 : f64
      %29 = mulf %28, %cst_1 : f64
      %30 = stencil.access %arg12 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = addf %30, %31 : f64
      %33 = mulf %32, %cst_1 : f64
      %34 = addf %29, %33 : f64
      %35 = sqrt %34 : f64
      %36 = mulf %35, %cst : f64
      %37 = subf %36, %25 : f64
      %38 = cmpf "ogt", %37, %cst_2 : f64
      %39 = select %38, %37, %cst_2 : f64
      %40 = cmpf "olt", %39, %cst_1 : f64
      %41 = select %40, %39, %cst_1 : f64
      %42 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %43 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = mulf %41, %42 : f64
      %45 = addf %44, %43 : f64
      %46 = stencil.store_result %45 : (f64) -> !stencil.result<f64>
      stencil.return %46 : !stencil.result<f64>
    }
    %23 = stencil.apply (%arg10 = %11 : !stencil.temp<?x?x?xf64>, %arg11 = %18 : !stencil.temp<?x?x?xf64>, %arg12 = %19 : !stencil.temp<?x?x?xf64>, %arg13 = %21 : !stencil.temp<?x?x?xf64>, %arg14 = %12 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e-02 : f64
      %cst_0 = constant 2.500000e-02 : f64
      %24 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = mulf %cst_0, %24 : f64
      %cst_1 = constant 5.000000e-01 : f64
      %cst_2 = constant 0.000000e+00 : f64
      %26 = stencil.access %arg11 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = addf %26, %27 : f64
      %29 = mulf %28, %cst_1 : f64
      %30 = stencil.access %arg12 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = addf %30, %31 : f64
      %33 = mulf %32, %cst_1 : f64
      %34 = addf %29, %33 : f64
      %35 = sqrt %34 : f64
      %36 = mulf %35, %cst : f64
      %37 = subf %36, %25 : f64
      %38 = cmpf "ogt", %37, %cst_2 : f64
      %39 = select %38, %37, %cst_2 : f64
      %40 = cmpf "olt", %39, %cst_1 : f64
      %41 = select %40, %39, %cst_1 : f64
      %42 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %43 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = mulf %41, %42 : f64
      %45 = addf %44, %43 : f64
      %46 = stencil.store_result %45 : (f64) -> !stencil.result<f64>
      stencil.return %46 : !stencil.result<f64>
    }
    stencil.store %22 to %3([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %23 to %4([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
