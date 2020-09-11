

module {
  func @p_grad_c(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.cast %arg3([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %4 = stencil.cast %arg4([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %5 = stencil.cast %arg5([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %6 = stencil.cast %arg6([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %7 = stencil.cast %arg7([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %8 = stencil.cast %arg8([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %9 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %10 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %11 = stencil.load %2 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %12 = stencil.load %3 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %13 = stencil.load %4 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %14 = stencil.load %5 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %15 = stencil.load %6 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %16 = stencil.apply (%arg9 = %13 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %19 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.store_result %19 : (f64) -> !stencil.result<f64>
      stencil.return %20 : !stencil.result<f64>
    }
    %17 = stencil.apply (%arg9 = %9 : !stencil.temp<?x?x?xf64>, %arg10 = %11 : !stencil.temp<?x?x?xf64>, %arg11 = %16 : !stencil.temp<?x?x?xf64>, %arg12 = %14 : !stencil.temp<?x?x?xf64>, %arg13 = %15 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e-01 : f64
      %19 = stencil.access %arg12 [-1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = subf %19, %20 : f64
      %22 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg13 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = subf %22, %23 : f64
      %25 = mulf %21, %24 : f64
      %26 = stencil.access %arg12 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg12 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = stencil.access %arg13 [-1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = subf %29, %30 : f64
      %32 = mulf %28, %31 : f64
      %33 = addf %25, %32 : f64
      %34 = stencil.access %arg11 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %36 = addf %34, %35 : f64
      %37 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %38 = mulf %37, %cst : f64
      %39 = divf %38, %36 : f64
      %40 = mulf %39, %33 : f64
      %41 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = addf %40, %41 : f64
      %43 = stencil.store_result %42 : (f64) -> !stencil.result<f64>
      stencil.return %43 : !stencil.result<f64>
    }
    %18 = stencil.apply (%arg9 = %10 : !stencil.temp<?x?x?xf64>, %arg10 = %12 : !stencil.temp<?x?x?xf64>, %arg11 = %16 : !stencil.temp<?x?x?xf64>, %arg12 = %14 : !stencil.temp<?x?x?xf64>, %arg13 = %15 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e-01 : f64
      %19 = stencil.access %arg12 [0, -1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = subf %19, %20 : f64
      %22 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg13 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = subf %22, %23 : f64
      %25 = mulf %21, %24 : f64
      %26 = stencil.access %arg12 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg12 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = stencil.access %arg13 [0, -1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = subf %29, %30 : f64
      %32 = mulf %28, %31 : f64
      %33 = addf %25, %32 : f64
      %34 = stencil.access %arg11 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %36 = addf %34, %35 : f64
      %37 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %38 = mulf %37, %cst : f64
      %39 = divf %38, %36 : f64
      %40 = mulf %39, %33 : f64
      %41 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = addf %40, %41 : f64
      %43 = stencil.store_result %42 : (f64) -> !stencil.result<f64>
      stencil.return %43 : !stencil.result<f64>
    }
    stencil.store %17 to %7([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %18 to %8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
