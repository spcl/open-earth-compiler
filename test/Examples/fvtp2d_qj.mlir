

module {
  func @fvtp2d_qj(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
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
    %15 = stencil.apply (%arg9 = %14 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %23 = divf %cst_0, %cst_1 : f64
      %24 = divf %cst, %cst_1 : f64
      %25 = stencil.access %arg9 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %25, %26 : f64
      %28 = stencil.access %arg9 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg9 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = mulf %23, %27 : f64
      %32 = mulf %24, %30 : f64
      %33 = addf %31, %32 : f64
      %34 = stencil.store_result %33 : (f64) -> !stencil.result<f64>
      stencil.return %34 : !stencil.result<f64>
    }
    %16:4 = stencil.apply (%arg9 = %14 : !stencil.temp<?x?x?xf64>, %arg10 = %15 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %23 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = subf %26, %24 : f64
      %28 = addf %25, %27 : f64
      %29 = mulf %25, %27 : f64
      %30 = cmpf "olt", %29, %cst : f64
      %31 = select %30, %cst_0, %cst : f64
      %32 = stencil.store_result %25 : (f64) -> !stencil.result<f64>
      %33 = stencil.store_result %27 : (f64) -> !stencil.result<f64>
      %34 = stencil.store_result %28 : (f64) -> !stencil.result<f64>
      %35 = stencil.store_result %31 : (f64) -> !stencil.result<f64>
      stencil.return %32, %33, %34, %35 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
    }
    %17 = stencil.apply (%arg9 = %14 : !stencil.temp<?x?x?xf64>, %arg10 = %10 : !stencil.temp<?x?x?xf64>, %arg11 = %16#0 : !stencil.temp<?x?x?xf64>, %arg12 = %16#1 : !stencil.temp<?x?x?xf64>, %arg13 = %16#2 : !stencil.temp<?x?x?xf64>, %arg14 = %16#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %23 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = cmpf "oeq", %23, %cst : f64
      %25 = select %24, %cst_0, %cst : f64
      %26 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = mulf %26, %25 : f64
      %28 = addf %23, %27 : f64
      %29 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = cmpf "ogt", %29, %cst : f64
      %31 = scf.if %30 -> (f64) {
        %35 = stencil.access %arg12 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %36 = stencil.access %arg13 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %37 = mulf %29, %36 : f64
        %38 = subf %35, %37 : f64
        %39 = subf %cst_0, %29 : f64
        %40 = mulf %39, %38 : f64
        scf.yield %40 : f64
      } else {
        %35 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %36 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %37 = mulf %29, %36 : f64
        %38 = addf %35, %37 : f64
        %39 = addf %cst_0, %29 : f64
        %40 = mulf %39, %38 : f64
        scf.yield %40 : f64
      }
      %32 = mulf %31, %28 : f64
      %33 = scf.if %30 -> (f64) {
        %35 = stencil.access %arg9 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %36 = addf %35, %32 : f64
        scf.yield %36 : f64
      } else {
        %35 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %36 = addf %35, %32 : f64
        scf.yield %36 : f64
      }
      %34 = stencil.store_result %33 : (f64) -> !stencil.result<f64>
      stencil.return %34 : !stencil.result<f64>
    }
    %18 = stencil.apply (%arg9 = %9 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %23 = divf %cst_0, %cst_1 : f64
      %24 = divf %cst, %cst_1 : f64
      %25 = stencil.access %arg9 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %25, %26 : f64
      %28 = stencil.access %arg9 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg9 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = mulf %23, %27 : f64
      %32 = mulf %24, %30 : f64
      %33 = addf %31, %32 : f64
      %34 = stencil.store_result %33 : (f64) -> !stencil.result<f64>
      stencil.return %34 : !stencil.result<f64>
    }
    %19:4 = stencil.apply (%arg9 = %9 : !stencil.temp<?x?x?xf64>, %arg10 = %18 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %23 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = subf %26, %24 : f64
      %28 = addf %25, %27 : f64
      %29 = mulf %25, %27 : f64
      %30 = cmpf "olt", %29, %cst : f64
      %31 = select %30, %cst_0, %cst : f64
      %32 = stencil.store_result %25 : (f64) -> !stencil.result<f64>
      %33 = stencil.store_result %27 : (f64) -> !stencil.result<f64>
      %34 = stencil.store_result %28 : (f64) -> !stencil.result<f64>
      %35 = stencil.store_result %31 : (f64) -> !stencil.result<f64>
      stencil.return %32, %33, %34, %35 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
    }
    %20 = stencil.apply (%arg9 = %9 : !stencil.temp<?x?x?xf64>, %arg10 = %10 : !stencil.temp<?x?x?xf64>, %arg11 = %19#0 : !stencil.temp<?x?x?xf64>, %arg12 = %19#1 : !stencil.temp<?x?x?xf64>, %arg13 = %19#2 : !stencil.temp<?x?x?xf64>, %arg14 = %19#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %23 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = cmpf "oeq", %23, %cst : f64
      %25 = select %24, %cst_0, %cst : f64
      %26 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = mulf %26, %25 : f64
      %28 = addf %23, %27 : f64
      %29 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = cmpf "ogt", %29, %cst : f64
      %31 = scf.if %30 -> (f64) {
        %35 = stencil.access %arg12 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %36 = stencil.access %arg13 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %37 = mulf %29, %36 : f64
        %38 = subf %35, %37 : f64
        %39 = subf %cst_0, %29 : f64
        %40 = mulf %39, %38 : f64
        scf.yield %40 : f64
      } else {
        %35 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %36 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %37 = mulf %29, %36 : f64
        %38 = addf %35, %37 : f64
        %39 = addf %cst_0, %29 : f64
        %40 = mulf %39, %38 : f64
        scf.yield %40 : f64
      }
      %32 = mulf %31, %28 : f64
      %33 = scf.if %30 -> (f64) {
        %35 = stencil.access %arg9 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %36 = addf %35, %32 : f64
        scf.yield %36 : f64
      } else {
        %35 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %36 = addf %35, %32 : f64
        scf.yield %36 : f64
      }
      %34 = stencil.store_result %33 : (f64) -> !stencil.result<f64>
      stencil.return %34 : !stencil.result<f64>
    }
    %21 = stencil.apply (%arg9 = %12 : !stencil.temp<?x?x?xf64>, %arg10 = %20 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %23 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = mulf %23, %24 : f64
      %26 = stencil.store_result %25 : (f64) -> !stencil.result<f64>
      stencil.return %26 : !stencil.result<f64>
    }
    %22 = stencil.apply (%arg9 = %9 : !stencil.temp<?x?x?xf64>, %arg10 = %13 : !stencil.temp<?x?x?xf64>, %arg11 = %21 : !stencil.temp<?x?x?xf64>, %arg12 = %11 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %23 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = mulf %23, %24 : f64
      %26 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg11 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = addf %25, %28 : f64
      %30 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = divf %29, %30 : f64
      %32 = stencil.store_result %31 : (f64) -> !stencil.result<f64>
      stencil.return %32 : !stencil.result<f64>
    }
    stencil.store %22 to %6([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %17 to %7([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %20 to %8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
