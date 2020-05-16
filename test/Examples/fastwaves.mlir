

module {
  func @fastwaves(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>, %arg9: !stencil.field<?x?x?xf64>, %arg10: !stencil.field<?x?x?xf64>, %arg11: !stencil.field<?x?x?xf64>, %arg12: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg6([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg7([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg8([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg9([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg10([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg11([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg12([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg2 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.load %arg3 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %4 = stencil.load %arg4 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %5 = stencil.load %arg5 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %6 = stencil.load %arg6 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %7 = stencil.load %arg7 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %8 = stencil.load %arg8 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %9 = stencil.load %arg9 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %10 = stencil.apply (%arg13 = %4 : !stencil.temp<?x?x?xf64>, %arg14 = %5 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %19 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = mulf %19, %20 : f64
      %cst = constant 1.000000e+00 : f64
      %22 = subf %cst, %19 : f64
      %23 = stencil.access %arg14 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = mulf %23, %22 : f64
      %25 = addf %21, %24 : f64
      stencil.return %25 : f64
    }
    %11 = stencil.apply (%arg13 = %10 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %19 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = subf %19, %20 : f64
      stencil.return %21 : f64
    }
    %12 = stencil.apply (%arg13 = %5 : !stencil.temp<?x?x?xf64>, %arg14 = %11 : !stencil.temp<?x?x?xf64>, %arg15 = %6 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %19 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = subf %19, %20 : f64
      %cst = constant 5.000000e-01 : f64
      %22 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = addf %22, %23 : f64
      %25 = mulf %cst, %24 : f64
      %26 = stencil.access %arg15 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = stencil.access %arg15 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg15 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %26, %27 : f64
      %31 = addf %28, %29 : f64
      %32 = subf %30, %31 : f64
      %33 = subf %26, %27 : f64
      %34 = subf %28, %29 : f64
      %35 = addf %33, %34 : f64
      %36 = divf %32, %35 : f64
      %37 = mulf %25, %36 : f64
      %38 = addf %21, %37 : f64
      stencil.return %38 : f64
    }
    %13 = stencil.apply (%arg13 = %5 : !stencil.temp<?x?x?xf64>, %arg14 = %11 : !stencil.temp<?x?x?xf64>, %arg15 = %6 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %19 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = subf %19, %20 : f64
      %cst = constant 5.000000e-01 : f64
      %22 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = addf %22, %23 : f64
      %25 = mulf %cst, %24 : f64
      %26 = stencil.access %arg15 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = stencil.access %arg15 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg15 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %26, %27 : f64
      %31 = addf %28, %29 : f64
      %32 = subf %30, %31 : f64
      %33 = subf %26, %27 : f64
      %34 = subf %28, %29 : f64
      %35 = addf %33, %34 : f64
      %36 = divf %32, %35 : f64
      %37 = mulf %25, %36 : f64
      %38 = addf %21, %37 : f64
      stencil.return %38 : f64
    }
    %14 = stencil.apply (%arg13 = %0 : !stencil.temp<?x?x?xf64>, %arg14 = %1 : !stencil.temp<?x?x?xf64>, %arg15 = %12 : !stencil.temp<?x?x?xf64>, %arg16 = %7 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 2.000000e+00 : f64
      %cst_0 = constant 1.000000e-02 : f64
      %19 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg16 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = addf %20, %21 : f64
      %23 = divf %cst, %22 : f64
      %24 = mulf %19, %23 : f64
      %25 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = subf %25, %24 : f64
      %27 = mulf %cst_0, %26 : f64
      %28 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = addf %27, %28 : f64
      stencil.return %29 : f64
    }
    %15 = stencil.apply (%arg13 = %2 : !stencil.temp<?x?x?xf64>, %arg14 = %3 : !stencil.temp<?x?x?xf64>, %arg15 = %13 : !stencil.temp<?x?x?xf64>, %arg16 = %7 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 2.000000e+00 : f64
      %cst_0 = constant 1.000000e-02 : f64
      %19 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg16 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = addf %20, %21 : f64
      %23 = divf %cst, %22 : f64
      %24 = mulf %19, %23 : f64
      %25 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = subf %25, %24 : f64
      %27 = mulf %cst_0, %26 : f64
      %28 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = addf %27, %28 : f64
      stencil.return %29 : f64
    }
    %16 = stencil.apply (%arg13 = %4 : !stencil.temp<?x?x?xf64>, %arg14 = %14 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %19 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = addf %19, %20 : f64
      %22 = mulf %cst, %21 : f64
      %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg14 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = mulf %22, %23 : f64
      %26 = subf %cst_0, %22 : f64
      %27 = mulf %26, %24 : f64
      %28 = addf %25, %27 : f64
      stencil.return %28 : f64
    }
    %17 = stencil.apply (%arg13 = %4 : !stencil.temp<?x?x?xf64>, %arg14 = %15 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %19 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = addf %19, %20 : f64
      %22 = mulf %cst, %21 : f64
      %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg14 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = mulf %22, %23 : f64
      %26 = subf %cst_0, %22 : f64
      %27 = mulf %26, %24 : f64
      %28 = addf %25, %27 : f64
      stencil.return %28 : f64
    }
    %18 = stencil.apply (%arg13 = %14 : !stencil.temp<?x?x?xf64>, %arg14 = %16 : !stencil.temp<?x?x?xf64>, %arg15 = %15 : !stencil.temp<?x?x?xf64>, %arg16 = %17 : !stencil.temp<?x?x?xf64>, %arg17 = %8 : !stencil.temp<?x?x?xf64>, %arg18 = %9 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e-01 : f64
      %cst_0 = constant 1.000000e-01 : f64
      %cst_1 = constant 2.000000e-01 : f64
      %cst_2 = constant 3.000000e-01 : f64
      %19 = stencil.access %arg14 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = subf %19, %20 : f64
      %22 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = mulf %21, %22 : f64
      %24 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = addf %24, %23 : f64
      %26 = mulf %cst, %25 : f64
      %27 = stencil.access %arg14 [-1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = subf %27, %28 : f64
      %30 = stencil.access %arg17 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = mulf %29, %30 : f64
      %32 = stencil.access %arg13 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = subf %31, %32 : f64
      %34 = mulf %cst_0, %33 : f64
      %35 = stencil.access %arg16 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %36 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %37 = subf %35, %36 : f64
      %38 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %39 = mulf %37, %38 : f64
      %40 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = addf %40, %39 : f64
      %42 = mulf %cst_1, %41 : f64
      %43 = stencil.access %arg16 [0, -1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = stencil.access %arg16 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %45 = subf %43, %44 : f64
      %46 = stencil.access %arg18 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = mulf %45, %46 : f64
      %48 = stencil.access %arg15 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %49 = subf %47, %48 : f64
      %50 = mulf %cst_2, %49 : f64
      %51 = addf %26, %34 : f64
      %52 = addf %51, %42 : f64
      %53 = addf %50, %52 : f64
      stencil.return %53 : f64
    }
    stencil.store %14 to %arg10([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %15 to %arg11([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %18 to %arg12([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
