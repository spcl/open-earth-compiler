

module {
  func @fastwavesuv(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>, %arg9: !stencil.field<?x?x?xf64>, %arg10: !stencil.field<0x?x0xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.cast %arg3([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %4 = stencil.cast %arg4([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %5 = stencil.cast %arg5([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %6 = stencil.cast %arg6([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %7 = stencil.cast %arg7([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %8 = stencil.cast %arg8([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %9 = stencil.cast %arg9([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %10 = stencil.cast %arg10([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %11 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %12 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %13 = stencil.load %2 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %14 = stencil.load %3 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %15 = stencil.load %4 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %16 = stencil.load %5 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %17 = stencil.load %6 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %18 = stencil.load %7 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %19 = stencil.load %10 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %20 = stencil.apply (%arg11 = %15 : !stencil.temp<?x?x?xf64>, %arg12 = %16 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %26 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = mulf %26, %27 : f64
      %cst = constant 1.000000e+00 : f64
      %29 = subf %cst, %26 : f64
      %30 = stencil.access %arg12 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = mulf %30, %29 : f64
      %32 = addf %28, %31 : f64
      %33 = stencil.store_result %32 : (f64) -> !stencil.result<f64>
      stencil.return %33 : !stencil.result<f64>
    }
    %21 = stencil.apply (%arg11 = %20 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %26 = stencil.access %arg11 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = stencil.store_result %28 : (f64) -> !stencil.result<f64>
      stencil.return %29 : !stencil.result<f64>
    }
    %22 = stencil.apply (%arg11 = %16 : !stencil.temp<?x?x?xf64>, %arg12 = %21 : !stencil.temp<?x?x?xf64>, %arg13 = %17 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %26 = stencil.access %arg11 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %cst = constant 5.000000e-01 : f64
      %29 = stencil.access %arg12 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = addf %29, %30 : f64
      %32 = mulf %cst, %31 : f64
      %33 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = stencil.access %arg13 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %36 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %37 = addf %33, %34 : f64
      %38 = addf %35, %36 : f64
      %39 = subf %37, %38 : f64
      %40 = subf %33, %34 : f64
      %41 = subf %35, %36 : f64
      %42 = addf %40, %41 : f64
      %43 = divf %39, %42 : f64
      %44 = mulf %32, %43 : f64
      %45 = addf %28, %44 : f64
      %46 = stencil.store_result %45 : (f64) -> !stencil.result<f64>
      stencil.return %46 : !stencil.result<f64>
    }
    %23 = stencil.apply (%arg11 = %16 : !stencil.temp<?x?x?xf64>, %arg12 = %21 : !stencil.temp<?x?x?xf64>, %arg13 = %17 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %26 = stencil.access %arg11 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %cst = constant 5.000000e-01 : f64
      %29 = stencil.access %arg12 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = addf %29, %30 : f64
      %32 = mulf %cst, %31 : f64
      %33 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = stencil.access %arg13 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %36 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %37 = addf %33, %34 : f64
      %38 = addf %35, %36 : f64
      %39 = subf %37, %38 : f64
      %40 = subf %33, %34 : f64
      %41 = subf %35, %36 : f64
      %42 = addf %40, %41 : f64
      %43 = divf %39, %42 : f64
      %44 = mulf %32, %43 : f64
      %45 = addf %28, %44 : f64
      %46 = stencil.store_result %45 : (f64) -> !stencil.result<f64>
      stencil.return %46 : !stencil.result<f64>
    }
    %24 = stencil.apply (%arg11 = %11 : !stencil.temp<?x?x?xf64>, %arg12 = %12 : !stencil.temp<?x?x?xf64>, %arg13 = %22 : !stencil.temp<?x?x?xf64>, %arg14 = %18 : !stencil.temp<?x?x?xf64>, %arg15 = %19 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+01 : f64
      %cst_0 = constant 2.000000e+00 : f64
      %26 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %30 = addf %27, %28 : f64
      %31 = mulf %cst_0, %29 : f64
      %32 = divf %31, %30 : f64
      %33 = mulf %26, %32 : f64
      %34 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = subf %34, %33 : f64
      %36 = mulf %cst, %35 : f64
      %37 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %38 = addf %36, %37 : f64
      %39 = stencil.store_result %38 : (f64) -> !stencil.result<f64>
      stencil.return %39 : !stencil.result<f64>
    }
    %25 = stencil.apply (%arg11 = %13 : !stencil.temp<?x?x?xf64>, %arg12 = %14 : !stencil.temp<?x?x?xf64>, %arg13 = %23 : !stencil.temp<?x?x?xf64>, %arg14 = %18 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+01 : f64
      %cst_0 = constant 4.8828125E-4 : f64
      %cst_1 = constant 2.000000e+00 : f64
      %26 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = addf %27, %28 : f64
      %30 = mulf %cst_1, %cst_0 : f64
      %31 = divf %30, %29 : f64
      %32 = mulf %26, %31 : f64
      %33 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = subf %33, %32 : f64
      %35 = mulf %cst, %34 : f64
      %36 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %37 = addf %35, %36 : f64
      %38 = stencil.store_result %37 : (f64) -> !stencil.result<f64>
      stencil.return %38 : !stencil.result<f64>
    }
    stencil.store %24 to %8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %25 to %9([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
