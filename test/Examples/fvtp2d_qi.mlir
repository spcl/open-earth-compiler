

module {
  func @fvtp2d_qi(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg6([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg2 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.load %arg3 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %4 = stencil.load %arg4 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %5 = stencil.apply (%arg7 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %10 = divf %cst_0, %cst_1 : f64
      %11 = divf %cst, %cst_1 : f64
      %12 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = addf %12, %13 : f64
      %15 = stencil.access %arg7 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg7 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = addf %15, %16 : f64
      %18 = mulf %10, %14 : f64
      %19 = mulf %11, %17 : f64
      %20 = addf %18, %19 : f64
      stencil.return %20 : f64
    }
    %6:4 = stencil.apply (%arg7 = %0 : !stencil.temp<?x?x?xf64>, %arg8 = %5 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %10 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = subf %10, %11 : f64
      %13 = stencil.access %arg8 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = subf %13, %11 : f64
      %15 = addf %12, %14 : f64
      %16 = mulf %12, %14 : f64
      %17 = cmpf "olt", %16, %cst : f64
      %18 = select %17, %cst_0, %cst : f64
      stencil.return %12, %14, %15, %18 : f64, f64, f64, f64
    }
    %7 = stencil.apply (%arg7 = %0 : !stencil.temp<?x?x?xf64>, %arg8 = %1 : !stencil.temp<?x?x?xf64>, %arg9 = %6#0 : !stencil.temp<?x?x?xf64>, %arg10 = %6#1 : !stencil.temp<?x?x?xf64>, %arg11 = %6#2 : !stencil.temp<?x?x?xf64>, %arg12 = %6#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %10 = stencil.access %arg12 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = cmpf "oeq", %10, %cst : f64
      %12 = select %11, %cst_0, %cst : f64
      %13 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = mulf %13, %12 : f64
      %15 = addf %10, %14 : f64
      %16 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = cmpf "ogt", %16, %cst : f64
      %18 = scf.if %17 -> (f64) {
        %21 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %22 = stencil.access %arg11 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %23 = mulf %16, %22 : f64
        %24 = subf %21, %23 : f64
        %25 = subf %cst_0, %16 : f64
        %26 = mulf %25, %24 : f64
        scf.yield %26 : f64
      } else {
        %21 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %22 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %23 = mulf %16, %22 : f64
        %24 = addf %21, %23 : f64
        %25 = addf %cst_0, %16 : f64
        %26 = mulf %25, %24 : f64
        scf.yield %26 : f64
      }
      %19 = mulf %18, %15 : f64
      %20 = scf.if %17 -> (f64) {
        %21 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %22 = addf %21, %19 : f64
        scf.yield %22 : f64
      } else {
        %21 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %22 = addf %21, %19 : f64
        scf.yield %22 : f64
      }
      stencil.return %20 : f64
    }
    %8 = stencil.apply (%arg7 = %3 : !stencil.temp<?x?x?xf64>, %arg8 = %7 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %10 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = mulf %10, %11 : f64
      stencil.return %12 : f64
    }
    %9 = stencil.apply (%arg7 = %0 : !stencil.temp<?x?x?xf64>, %arg8 = %4 : !stencil.temp<?x?x?xf64>, %arg9 = %8 : !stencil.temp<?x?x?xf64>, %arg10 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %10 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = mulf %10, %11 : f64
      %13 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg9 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = addf %12, %15 : f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = divf %16, %17 : f64
      stencil.return %18 : f64
    }
    stencil.store %7 to %arg6([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %9 to %arg5([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
