

module {
  func @fvtp2d_qi(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.cast %arg3([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %4 = stencil.cast %arg4([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %5 = stencil.cast %arg5([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %6 = stencil.cast %arg6([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %7 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %8 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %9 = stencil.load %2 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %10 = stencil.load %3 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %11 = stencil.load %4 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %12 = stencil.apply (%arg7 = %7 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %17 = divf %cst_0, %cst_1 : f64
      %18 = divf %cst, %cst_1 : f64
      %19 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = addf %19, %20 : f64
      %22 = stencil.access %arg7 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg7 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = addf %22, %23 : f64
      %25 = mulf %17, %21 : f64
      %26 = mulf %18, %24 : f64
      %27 = addf %25, %26 : f64
      %28 = stencil.store_result %27 : (f64) -> !stencil.result<f64>
      stencil.return %28 : !stencil.result<f64>
    }
    %13:4 = stencil.apply (%arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %12 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %17 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = subf %17, %18 : f64
      %20 = stencil.access %arg8 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = subf %20, %18 : f64
      %22 = addf %19, %21 : f64
      %23 = mulf %19, %21 : f64
      %24 = cmpf "olt", %23, %cst : f64
      %25 = select %24, %cst_0, %cst : f64
      %26 = stencil.store_result %19 : (f64) -> !stencil.result<f64>
      %27 = stencil.store_result %21 : (f64) -> !stencil.result<f64>
      %28 = stencil.store_result %22 : (f64) -> !stencil.result<f64>
      %29 = stencil.store_result %25 : (f64) -> !stencil.result<f64>
      stencil.return %26, %27, %28, %29 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
    }
    %14 = stencil.apply (%arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %13#0 : !stencil.temp<?x?x?xf64>, %arg10 = %13#1 : !stencil.temp<?x?x?xf64>, %arg11 = %13#2 : !stencil.temp<?x?x?xf64>, %arg12 = %13#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %17 = stencil.access %arg12 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = cmpf "oeq", %17, %cst : f64
      %19 = select %18, %cst_0, %cst : f64
      %20 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = mulf %20, %19 : f64
      %22 = addf %17, %21 : f64
      %23 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = cmpf "ogt", %23, %cst : f64
      %25 = scf.if %24 -> (f64) {
        %29 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %30 = stencil.access %arg11 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %31 = mulf %23, %30 : f64
        %32 = subf %29, %31 : f64
        %33 = subf %cst_0, %23 : f64
        %34 = mulf %33, %32 : f64
        scf.yield %34 : f64
      } else {
        %29 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %30 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %31 = mulf %23, %30 : f64
        %32 = addf %29, %31 : f64
        %33 = addf %cst_0, %23 : f64
        %34 = mulf %33, %32 : f64
        scf.yield %34 : f64
      }
      %26 = mulf %25, %22 : f64
      %27 = scf.if %24 -> (f64) {
        %29 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %30 = addf %29, %26 : f64
        scf.yield %30 : f64
      } else {
        %29 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %30 = addf %29, %26 : f64
        scf.yield %30 : f64
      }
      %28 = stencil.store_result %27 : (f64) -> !stencil.result<f64>
      stencil.return %28 : !stencil.result<f64>
    }
    %15 = stencil.apply (%arg7 = %10 : !stencil.temp<?x?x?xf64>, %arg8 = %14 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %17 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = mulf %17, %18 : f64
      %20 = stencil.store_result %19 : (f64) -> !stencil.result<f64>
      stencil.return %20 : !stencil.result<f64>
    }
    %16 = stencil.apply (%arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %11 : !stencil.temp<?x?x?xf64>, %arg9 = %15 : !stencil.temp<?x?x?xf64>, %arg10 = %9 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %17 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = mulf %17, %18 : f64
      %20 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg9 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = addf %19, %22 : f64
      %24 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = divf %23, %24 : f64
      %26 = stencil.store_result %25 : (f64) -> !stencil.result<f64>
      stencil.return %26 : !stencil.result<f64>
    }
    stencil.store %14 to %6([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %16 to %5([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
