

module {
  func @fvtp2d_flux(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
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
    %6 = stencil.load %arg6 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %7 = stencil.apply (%arg9 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %12 = divf %cst_0, %cst_1 : f64
      %13 = divf %cst, %cst_1 : f64
      %14 = stencil.access %arg9 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = addf %14, %15 : f64
      %17 = stencil.access %arg9 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg9 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = addf %17, %18 : f64
      %20 = mulf %12, %16 : f64
      %21 = mulf %13, %19 : f64
      %22 = addf %20, %21 : f64
      stencil.return %22 : f64
    }
    %8:4 = stencil.apply (%arg9 = %3 : !stencil.temp<?x?x?xf64>, %arg10 = %7 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %12 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = subf %12, %13 : f64
      %15 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = subf %15, %13 : f64
      %17 = addf %14, %16 : f64
      %18 = mulf %14, %16 : f64
      %19 = cmpf "olt", %18, %cst : f64
      %20 = select %19, %cst_0, %cst : f64
      stencil.return %14, %16, %17, %20 : f64, f64, f64, f64
    }
    %9 = stencil.apply (%arg9 = %3 : !stencil.temp<?x?x?xf64>, %arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %8#0 : !stencil.temp<?x?x?xf64>, %arg12 = %8#1 : !stencil.temp<?x?x?xf64>, %arg13 = %8#2 : !stencil.temp<?x?x?xf64>, %arg14 = %8#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %12 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = cmpf "oeq", %12, %cst : f64
      %14 = select %13, %cst_0, %cst : f64
      %15 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = mulf %15, %14 : f64
      %17 = addf %12, %16 : f64
      %18 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = cmpf "ogt", %18, %cst : f64
      %20 = scf.if %19 -> (f64) {
        %23 = stencil.access %arg12 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %24 = stencil.access %arg13 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %25 = mulf %18, %24 : f64
        %26 = subf %23, %25 : f64
        %27 = subf %cst_0, %18 : f64
        %28 = mulf %27, %26 : f64
        scf.yield %28 : f64
      } else {
        %23 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %24 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %25 = mulf %18, %24 : f64
        %26 = addf %23, %25 : f64
        %27 = addf %cst_0, %18 : f64
        %28 = mulf %27, %26 : f64
        scf.yield %28 : f64
      }
      %21 = mulf %20, %17 : f64
      %22 = scf.if %19 -> (f64) {
        %23 = stencil.access %arg9 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %24 = addf %23, %21 : f64
        scf.yield %24 : f64
      } else {
        %23 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %24 = addf %23, %21 : f64
        scf.yield %24 : f64
      }
      stencil.return %22 : f64
    }
    %10 = stencil.apply (%arg9 = %4 : !stencil.temp<?x?x?xf64>, %arg10 = %5 : !stencil.temp<?x?x?xf64>, %arg11 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %12 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = addf %12, %13 : f64
      %15 = mulf %14, %cst : f64
      %16 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %16, %15 : f64
      stencil.return %17 : f64
    }
    %11 = stencil.apply (%arg9 = %9 : !stencil.temp<?x?x?xf64>, %arg10 = %6 : !stencil.temp<?x?x?xf64>, %arg11 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %12 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = addf %12, %13 : f64
      %15 = mulf %14, %cst : f64
      %16 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %16, %15 : f64
      stencil.return %17 : f64
    }
    stencil.store %10 to %arg7([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %11 to %arg8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
