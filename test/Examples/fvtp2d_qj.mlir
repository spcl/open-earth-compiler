

module {
  func @fvtp2d_qj(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg6([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg7([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg8([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg2 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.load %arg3 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %4 = stencil.load %arg4 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %5 = stencil.load %arg5 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %6 = stencil.apply (%arg9 = %5 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %14 = divf %cst_0, %cst_1 : f64
      %15 = divf %cst, %cst_1 : f64
      %16 = stencil.access %arg9 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %16, %17 : f64
      %19 = stencil.access %arg9 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg9 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = addf %19, %20 : f64
      %22 = mulf %14, %18 : f64
      %23 = mulf %15, %21 : f64
      %24 = addf %22, %23 : f64
      stencil.return %24 : f64
    }
    %7:4 = stencil.apply (%arg9 = %5 : !stencil.temp<?x?x?xf64>, %arg10 = %6 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %14 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = subf %14, %15 : f64
      %17 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = subf %17, %15 : f64
      %19 = addf %16, %18 : f64
      %20 = mulf %16, %18 : f64
      %21 = cmpf "olt", %20, %cst : f64
      %22 = select %21, %cst_0, %cst : f64
      stencil.return %16, %18, %19, %22 : f64, f64, f64, f64
    }
    %8 = stencil.apply (%arg9 = %5 : !stencil.temp<?x?x?xf64>, %arg10 = %1 : !stencil.temp<?x?x?xf64>, %arg11 = %7#0 : !stencil.temp<?x?x?xf64>, %arg12 = %7#1 : !stencil.temp<?x?x?xf64>, %arg13 = %7#2 : !stencil.temp<?x?x?xf64>, %arg14 = %7#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %14 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = cmpf "oeq", %14, %cst : f64
      %16 = select %15, %cst_0, %cst : f64
      %17 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = mulf %17, %16 : f64
      %19 = addf %14, %18 : f64
      %20 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = cmpf "ogt", %20, %cst : f64
      %22 = scf.if %21 -> (f64) {
        %25 = stencil.access %arg12 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %26 = stencil.access %arg13 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %27 = mulf %20, %26 : f64
        %28 = subf %25, %27 : f64
        %29 = subf %cst_0, %20 : f64
        %30 = mulf %29, %28 : f64
        scf.yield %30 : f64
      } else {
        %25 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %26 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %27 = mulf %20, %26 : f64
        %28 = addf %25, %27 : f64
        %29 = addf %cst_0, %20 : f64
        %30 = mulf %29, %28 : f64
        scf.yield %30 : f64
      }
      %23 = mulf %22, %19 : f64
      %24 = scf.if %21 -> (f64) {
        %25 = stencil.access %arg9 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %26 = addf %25, %23 : f64
        scf.yield %26 : f64
      } else {
        %25 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %26 = addf %25, %23 : f64
        scf.yield %26 : f64
      }
      stencil.return %24 : f64
    }
    %9 = stencil.apply (%arg9 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %14 = divf %cst_0, %cst_1 : f64
      %15 = divf %cst, %cst_1 : f64
      %16 = stencil.access %arg9 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %16, %17 : f64
      %19 = stencil.access %arg9 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg9 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = addf %19, %20 : f64
      %22 = mulf %14, %18 : f64
      %23 = mulf %15, %21 : f64
      %24 = addf %22, %23 : f64
      stencil.return %24 : f64
    }
    %10:4 = stencil.apply (%arg9 = %0 : !stencil.temp<?x?x?xf64>, %arg10 = %9 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %14 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = subf %14, %15 : f64
      %17 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = subf %17, %15 : f64
      %19 = addf %16, %18 : f64
      %20 = mulf %16, %18 : f64
      %21 = cmpf "olt", %20, %cst : f64
      %22 = select %21, %cst_0, %cst : f64
      stencil.return %16, %18, %19, %22 : f64, f64, f64, f64
    }
    %11 = stencil.apply (%arg9 = %0 : !stencil.temp<?x?x?xf64>, %arg10 = %1 : !stencil.temp<?x?x?xf64>, %arg11 = %10#0 : !stencil.temp<?x?x?xf64>, %arg12 = %10#1 : !stencil.temp<?x?x?xf64>, %arg13 = %10#2 : !stencil.temp<?x?x?xf64>, %arg14 = %10#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %14 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = cmpf "oeq", %14, %cst : f64
      %16 = select %15, %cst_0, %cst : f64
      %17 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = mulf %17, %16 : f64
      %19 = addf %14, %18 : f64
      %20 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = cmpf "ogt", %20, %cst : f64
      %22 = scf.if %21 -> (f64) {
        %25 = stencil.access %arg12 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %26 = stencil.access %arg13 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %27 = mulf %20, %26 : f64
        %28 = subf %25, %27 : f64
        %29 = subf %cst_0, %20 : f64
        %30 = mulf %29, %28 : f64
        scf.yield %30 : f64
      } else {
        %25 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %26 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %27 = mulf %20, %26 : f64
        %28 = addf %25, %27 : f64
        %29 = addf %cst_0, %20 : f64
        %30 = mulf %29, %28 : f64
        scf.yield %30 : f64
      }
      %23 = mulf %22, %19 : f64
      %24 = scf.if %21 -> (f64) {
        %25 = stencil.access %arg9 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %26 = addf %25, %23 : f64
        scf.yield %26 : f64
      } else {
        %25 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %26 = addf %25, %23 : f64
        scf.yield %26 : f64
      }
      stencil.return %24 : f64
    }
    %12 = stencil.apply (%arg9 = %3 : !stencil.temp<?x?x?xf64>, %arg10 = %11 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = mulf %14, %15 : f64
      stencil.return %16 : f64
    }
    %13 = stencil.apply (%arg9 = %0 : !stencil.temp<?x?x?xf64>, %arg10 = %4 : !stencil.temp<?x?x?xf64>, %arg11 = %12 : !stencil.temp<?x?x?xf64>, %arg12 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = mulf %14, %15 : f64
      %17 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg11 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = subf %17, %18 : f64
      %20 = addf %16, %19 : f64
      %21 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = divf %20, %21 : f64
      stencil.return %22 : f64
    }
    stencil.store %13 to %arg6([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %8 to %arg7([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %11 to %arg8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
