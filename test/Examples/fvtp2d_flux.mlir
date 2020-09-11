

module {
  func @fvtp2d_flux(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
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
    %16 = stencil.apply (%arg9 = %12 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %21 = divf %cst_0, %cst_1 : f64
      %22 = divf %cst, %cst_1 : f64
      %23 = stencil.access %arg9 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = addf %23, %24 : f64
      %26 = stencil.access %arg9 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg9 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = addf %26, %27 : f64
      %29 = mulf %21, %25 : f64
      %30 = mulf %22, %28 : f64
      %31 = addf %29, %30 : f64
      %32 = stencil.store_result %31 : (f64) -> !stencil.result<f64>
      stencil.return %32 : !stencil.result<f64>
    }
    %17:4 = stencil.apply (%arg9 = %12 : !stencil.temp<?x?x?xf64>, %arg10 = %16 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %21 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = subf %21, %22 : f64
      %24 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %24, %22 : f64
      %26 = addf %23, %25 : f64
      %27 = mulf %23, %25 : f64
      %28 = cmpf "olt", %27, %cst : f64
      %29 = select %28, %cst_0, %cst : f64
      %30 = stencil.store_result %23 : (f64) -> !stencil.result<f64>
      %31 = stencil.store_result %25 : (f64) -> !stencil.result<f64>
      %32 = stencil.store_result %26 : (f64) -> !stencil.result<f64>
      %33 = stencil.store_result %29 : (f64) -> !stencil.result<f64>
      stencil.return %30, %31, %32, %33 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
    }
    %18 = stencil.apply (%arg9 = %12 : !stencil.temp<?x?x?xf64>, %arg10 = %9 : !stencil.temp<?x?x?xf64>, %arg11 = %17#0 : !stencil.temp<?x?x?xf64>, %arg12 = %17#1 : !stencil.temp<?x?x?xf64>, %arg13 = %17#2 : !stencil.temp<?x?x?xf64>, %arg14 = %17#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %21 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = cmpf "oeq", %21, %cst : f64
      %23 = select %22, %cst_0, %cst : f64
      %24 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = mulf %24, %23 : f64
      %26 = addf %21, %25 : f64
      %27 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = cmpf "ogt", %27, %cst : f64
      %29 = scf.if %28 -> (f64) {
        %33 = stencil.access %arg12 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %34 = stencil.access %arg13 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %35 = mulf %27, %34 : f64
        %36 = subf %33, %35 : f64
        %37 = subf %cst_0, %27 : f64
        %38 = mulf %37, %36 : f64
        scf.yield %38 : f64
      } else {
        %33 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %34 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %35 = mulf %27, %34 : f64
        %36 = addf %33, %35 : f64
        %37 = addf %cst_0, %27 : f64
        %38 = mulf %37, %36 : f64
        scf.yield %38 : f64
      }
      %30 = mulf %29, %26 : f64
      %31 = scf.if %28 -> (f64) {
        %33 = stencil.access %arg9 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %34 = addf %33, %30 : f64
        scf.yield %34 : f64
      } else {
        %33 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %34 = addf %33, %30 : f64
        scf.yield %34 : f64
      }
      %32 = stencil.store_result %31 : (f64) -> !stencil.result<f64>
      stencil.return %32 : !stencil.result<f64>
    }
    %19 = stencil.apply (%arg9 = %13 : !stencil.temp<?x?x?xf64>, %arg10 = %14 : !stencil.temp<?x?x?xf64>, %arg11 = %10 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %21 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = addf %21, %22 : f64
      %24 = mulf %23, %cst : f64
      %25 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = mulf %25, %24 : f64
      %27 = stencil.store_result %26 : (f64) -> !stencil.result<f64>
      stencil.return %27 : !stencil.result<f64>
    }
    %20 = stencil.apply (%arg9 = %18 : !stencil.temp<?x?x?xf64>, %arg10 = %15 : !stencil.temp<?x?x?xf64>, %arg11 = %11 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %21 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = addf %21, %22 : f64
      %24 = mulf %23, %cst : f64
      %25 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = mulf %25, %24 : f64
      %27 = stencil.store_result %26 : (f64) -> !stencil.result<f64>
      stencil.return %27 : !stencil.result<f64>
    }
    stencil.store %19 to %7([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %20 to %8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
