

module {
  func @nh_p_grad(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>, %arg9: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
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
    %10 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %11 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %12 = stencil.load %2 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %13 = stencil.load %3 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %14 = stencil.load %4 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %15 = stencil.load %5 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %16 = stencil.load %6 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %17 = stencil.load %7 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %18 = stencil.apply (%arg10 = %16 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %23 = stencil.access %arg10 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = stencil.store_result %25 : (f64) -> !stencil.result<f64>
      stencil.return %26 : !stencil.result<f64>
    }
    %19 = stencil.apply (%arg10 = %18 : !stencil.temp<?x?x?xf64>, %arg11 = %14 : !stencil.temp<?x?x?xf64>, %arg12 = %16 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e-01 : f64
      %23 = stencil.access %arg11 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg11 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = stencil.access %arg12 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = mulf %25, %28 : f64
      %30 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg11 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = subf %30, %31 : f64
      %33 = stencil.access %arg12 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = stencil.access %arg12 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = subf %33, %34 : f64
      %36 = mulf %32, %35 : f64
      %37 = addf %29, %36 : f64
      %38 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %39 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %40 = addf %38, %39 : f64
      %41 = divf %cst, %40 : f64
      %42 = mulf %37, %41 : f64
      %43 = stencil.store_result %42 : (f64) -> !stencil.result<f64>
      stencil.return %43 : !stencil.result<f64>
    }
    %20 = stencil.apply (%arg10 = %18 : !stencil.temp<?x?x?xf64>, %arg11 = %14 : !stencil.temp<?x?x?xf64>, %arg12 = %16 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e-01 : f64
      %23 = stencil.access %arg11 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg11 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = stencil.access %arg12 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = mulf %25, %28 : f64
      %30 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg11 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = subf %30, %31 : f64
      %33 = stencil.access %arg12 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = stencil.access %arg12 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = subf %33, %34 : f64
      %36 = mulf %32, %35 : f64
      %37 = addf %29, %36 : f64
      %38 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %39 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %40 = addf %38, %39 : f64
      %41 = divf %cst, %40 : f64
      %42 = mulf %37, %41 : f64
      %43 = stencil.store_result %42 : (f64) -> !stencil.result<f64>
      stencil.return %43 : !stencil.result<f64>
    }
    %21 = stencil.apply (%arg10 = %10 : !stencil.temp<?x?x?xf64>, %arg11 = %17 : !stencil.temp<?x?x?xf64>, %arg12 = %14 : !stencil.temp<?x?x?xf64>, %arg13 = %15 : !stencil.temp<?x?x?xf64>, %arg14 = %19 : !stencil.temp<?x?x?xf64>, %arg15 = %12 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e-01 : f64
      %23 = stencil.access %arg12 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg12 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = stencil.access %arg13 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = mulf %25, %28 : f64
      %30 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg12 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = subf %30, %31 : f64
      %33 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = subf %33, %34 : f64
      %36 = mulf %32, %35 : f64
      %37 = addf %29, %36 : f64
      %38 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %39 = stencil.access %arg11 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %40 = addf %38, %39 : f64
      %41 = divf %cst, %40 : f64
      %42 = mulf %37, %41 : f64
      %43 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %45 = addf %43, %44 : f64
      %46 = addf %42, %45 : f64
      %47 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %48 = mulf %46, %47 : f64
      %49 = stencil.store_result %48 : (f64) -> !stencil.result<f64>
      stencil.return %49 : !stencil.result<f64>
    }
    %22 = stencil.apply (%arg10 = %11 : !stencil.temp<?x?x?xf64>, %arg11 = %17 : !stencil.temp<?x?x?xf64>, %arg12 = %14 : !stencil.temp<?x?x?xf64>, %arg13 = %15 : !stencil.temp<?x?x?xf64>, %arg14 = %20 : !stencil.temp<?x?x?xf64>, %arg15 = %13 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e-01 : f64
      %23 = stencil.access %arg12 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg12 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = stencil.access %arg13 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = mulf %25, %28 : f64
      %30 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg12 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = subf %30, %31 : f64
      %33 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = subf %33, %34 : f64
      %36 = mulf %32, %35 : f64
      %37 = addf %29, %36 : f64
      %38 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %39 = stencil.access %arg11 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %40 = addf %38, %39 : f64
      %41 = divf %cst, %40 : f64
      %42 = mulf %37, %41 : f64
      %43 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %45 = addf %43, %44 : f64
      %46 = addf %42, %45 : f64
      %47 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %48 = mulf %46, %47 : f64
      %49 = stencil.store_result %48 : (f64) -> !stencil.result<f64>
      stencil.return %49 : !stencil.result<f64>
    }
    stencil.store %21 to %8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %22 to %9([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
