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

  func @fastwavesuv(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>, %arg9: !stencil.field<?x?x?xf64>, %arg10: !stencil.field<0x?x0xf64>, %arg11: f64, %arg12: f64) attributes {stencil.program} {
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
    stencil.assert %arg10([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg2 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.load %arg3 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %4 = stencil.load %arg4 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %5 = stencil.load %arg5 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %6 = stencil.load %arg6 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %7 = stencil.load %arg7 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %8 = stencil.load %arg10 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %9 = stencil.apply (%arg13 = %4 : !stencil.temp<?x?x?xf64>, %arg14 = %5 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %15, %16 : f64
      %cst = constant 1.000000e+00 : f64
      %18 = subf %cst, %15 : f64
      %19 = stencil.access %arg14 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = mulf %19, %18 : f64
      %21 = addf %17, %20 : f64
      stencil.return %21 : f64
    }
    %10 = stencil.apply (%arg13 = %9 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = subf %15, %16 : f64
      stencil.return %17 : f64
    }
    %11 = stencil.apply (%arg13 = %5 : !stencil.temp<?x?x?xf64>, %arg14 = %10 : !stencil.temp<?x?x?xf64>, %arg15 = %6 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = subf %15, %16 : f64
      %cst = constant 5.000000e-01 : f64
      %18 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = addf %18, %19 : f64
      %21 = mulf %cst, %20 : f64
      %22 = stencil.access %arg15 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg15 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg15 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = addf %22, %23 : f64
      %27 = addf %24, %25 : f64
      %28 = subf %26, %27 : f64
      %29 = subf %22, %23 : f64
      %30 = subf %24, %25 : f64
      %31 = addf %29, %30 : f64
      %32 = divf %28, %31 : f64
      %33 = mulf %21, %32 : f64
      %34 = addf %17, %33 : f64
      stencil.return %34 : f64
    }
    %12 = stencil.apply (%arg13 = %5 : !stencil.temp<?x?x?xf64>, %arg14 = %10 : !stencil.temp<?x?x?xf64>, %arg15 = %6 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = subf %15, %16 : f64
      %cst = constant 5.000000e-01 : f64
      %18 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = addf %18, %19 : f64
      %21 = mulf %cst, %20 : f64
      %22 = stencil.access %arg15 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg15 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg15 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = addf %22, %23 : f64
      %27 = addf %24, %25 : f64
      %28 = subf %26, %27 : f64
      %29 = subf %22, %23 : f64
      %30 = subf %24, %25 : f64
      %31 = addf %29, %30 : f64
      %32 = divf %28, %31 : f64
      %33 = mulf %21, %32 : f64
      %34 = addf %17, %33 : f64
      stencil.return %34 : f64
    }
    %13 = stencil.apply (%arg13 = %0 : !stencil.temp<?x?x?xf64>, %arg14 = %1 : !stencil.temp<?x?x?xf64>, %arg15 = %11 : !stencil.temp<?x?x?xf64>, %arg16 = %7 : !stencil.temp<?x?x?xf64>, %arg17 = %8 : !stencil.temp<0x?x0xf64>, %arg18 = %arg11 : f64) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 2.000000e+00 : f64
      %15 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg16 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %19 = addf %16, %17 : f64
      %20 = mulf %cst, %18 : f64
      %21 = divf %20, %19 : f64
      %22 = mulf %15, %21 : f64
      %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = subf %23, %22 : f64
      %25 = mulf %arg18, %24 : f64
      %26 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %25, %26 : f64
      stencil.return %27 : f64
    }
    %14 = stencil.apply (%arg13 = %2 : !stencil.temp<?x?x?xf64>, %arg14 = %3 : !stencil.temp<?x?x?xf64>, %arg15 = %12 : !stencil.temp<?x?x?xf64>, %arg16 = %7 : !stencil.temp<?x?x?xf64>, %arg17 = %arg11 : f64, %arg18 = %arg12 : f64) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 2.000000e+00 : f64
      %15 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg16 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %16, %17 : f64
      %19 = mulf %cst, %arg18 : f64
      %20 = divf %19, %18 : f64
      %21 = mulf %15, %20 : f64
      %22 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = subf %22, %21 : f64
      %24 = mulf %arg17, %23 : f64
      %25 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = addf %24, %25 : f64
      stencil.return %26 : f64
    }
    stencil.store %13 to %arg8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %14 to %arg9([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

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

  func @fvtp2d(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>, %arg9: !stencil.field<?x?x?xf64>, %arg10: !stencil.field<?x?x?xf64>, %arg11: !stencil.field<?x?x?xf64>, %arg12: !stencil.field<?x?x?xf64>, %arg13: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
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
    stencil.assert %arg13([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg2 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.load %arg3 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %4 = stencil.load %arg4 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %5 = stencil.load %arg5 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %6 = stencil.load %arg6 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %7 = stencil.load %arg7 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %8 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %26 = divf %cst_0, %cst_1 : f64
      %27 = divf %cst, %cst_1 : f64
      %28 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = stencil.access %arg14 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = addf %31, %32 : f64
      %34 = mulf %26, %30 : f64
      %35 = mulf %27, %33 : f64
      %36 = addf %34, %35 : f64
      stencil.return %36 : f64
    }
    %9:4 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>, %arg15 = %8 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %26 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = stencil.access %arg15 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = subf %29, %27 : f64
      %31 = addf %28, %30 : f64
      %32 = mulf %28, %30 : f64
      %33 = cmpf "olt", %32, %cst : f64
      %34 = select %33, %cst_0, %cst : f64
      stencil.return %28, %30, %31, %34 : f64, f64, f64, f64
    }
    %10 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>, %arg15 = %2 : !stencil.temp<?x?x?xf64>, %arg16 = %9#0 : !stencil.temp<?x?x?xf64>, %arg17 = %9#1 : !stencil.temp<?x?x?xf64>, %arg18 = %9#2 : !stencil.temp<?x?x?xf64>, %arg19 = %9#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %26 = stencil.access %arg19 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = cmpf "oeq", %26, %cst : f64
      %28 = select %27, %cst_0, %cst : f64
      %29 = stencil.access %arg19 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = mulf %29, %28 : f64
      %31 = addf %26, %30 : f64
      %32 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = cmpf "ogt", %32, %cst : f64
      %34 = scf.if %33 -> (f64) {
        %37 = stencil.access %arg17 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = stencil.access %arg18 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %32, %38 : f64
        %40 = subf %37, %39 : f64
        %41 = subf %cst_0, %32 : f64
        %42 = mulf %41, %40 : f64
        scf.yield %42 : f64
      } else {
        %37 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %32, %38 : f64
        %40 = addf %37, %39 : f64
        %41 = addf %cst_0, %32 : f64
        %42 = mulf %41, %40 : f64
        scf.yield %42 : f64
      }
      %35 = mulf %34, %31 : f64
      %36 = scf.if %33 -> (f64) {
        %37 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = addf %37, %35 : f64
        scf.yield %38 : f64
      } else {
        %37 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = addf %37, %35 : f64
        scf.yield %38 : f64
      }
      stencil.return %36 : f64
    }
    %11 = stencil.apply (%arg14 = %6 : !stencil.temp<?x?x?xf64>, %arg15 = %10 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %26 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = mulf %26, %27 : f64
      stencil.return %28 : f64
    }
    %12 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>, %arg15 = %7 : !stencil.temp<?x?x?xf64>, %arg16 = %11 : !stencil.temp<?x?x?xf64>, %arg17 = %4 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %26 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = mulf %26, %27 : f64
      %29 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg16 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = subf %29, %30 : f64
      %32 = addf %28, %31 : f64
      %33 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = divf %32, %33 : f64
      stencil.return %34 : f64
    }
    %13 = stencil.apply (%arg14 = %12 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %26 = divf %cst_0, %cst_1 : f64
      %27 = divf %cst, %cst_1 : f64
      %28 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = stencil.access %arg14 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = addf %31, %32 : f64
      %34 = mulf %26, %30 : f64
      %35 = mulf %27, %33 : f64
      %36 = addf %34, %35 : f64
      stencil.return %36 : f64
    }
    %14:4 = stencil.apply (%arg14 = %12 : !stencil.temp<?x?x?xf64>, %arg15 = %13 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %26 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = stencil.access %arg15 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = subf %29, %27 : f64
      %31 = addf %28, %30 : f64
      %32 = mulf %28, %30 : f64
      %33 = cmpf "olt", %32, %cst : f64
      %34 = select %33, %cst_0, %cst : f64
      stencil.return %28, %30, %31, %34 : f64, f64, f64, f64
    }
    %15 = stencil.apply (%arg14 = %12 : !stencil.temp<?x?x?xf64>, %arg15 = %1 : !stencil.temp<?x?x?xf64>, %arg16 = %14#0 : !stencil.temp<?x?x?xf64>, %arg17 = %14#1 : !stencil.temp<?x?x?xf64>, %arg18 = %14#2 : !stencil.temp<?x?x?xf64>, %arg19 = %14#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %26 = stencil.access %arg19 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = cmpf "oeq", %26, %cst : f64
      %28 = select %27, %cst_0, %cst : f64
      %29 = stencil.access %arg19 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = mulf %29, %28 : f64
      %31 = addf %26, %30 : f64
      %32 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = cmpf "ogt", %32, %cst : f64
      %34 = scf.if %33 -> (f64) {
        %37 = stencil.access %arg17 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = stencil.access %arg18 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %32, %38 : f64
        %40 = subf %37, %39 : f64
        %41 = subf %cst_0, %32 : f64
        %42 = mulf %41, %40 : f64
        scf.yield %42 : f64
      } else {
        %37 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %32, %38 : f64
        %40 = addf %37, %39 : f64
        %41 = addf %cst_0, %32 : f64
        %42 = mulf %41, %40 : f64
        scf.yield %42 : f64
      }
      %35 = mulf %34, %31 : f64
      %36 = scf.if %33 -> (f64) {
        %37 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = addf %37, %35 : f64
        scf.yield %38 : f64
      } else {
        %37 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = addf %37, %35 : f64
        scf.yield %38 : f64
      }
      stencil.return %36 : f64
    }
    %16 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %26 = divf %cst_0, %cst_1 : f64
      %27 = divf %cst, %cst_1 : f64
      %28 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = stencil.access %arg14 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = addf %31, %32 : f64
      %34 = mulf %26, %30 : f64
      %35 = mulf %27, %33 : f64
      %36 = addf %34, %35 : f64
      stencil.return %36 : f64
    }
    %17:4 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>, %arg15 = %16 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %26 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = stencil.access %arg15 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = subf %29, %27 : f64
      %31 = addf %28, %30 : f64
      %32 = mulf %28, %30 : f64
      %33 = cmpf "olt", %32, %cst : f64
      %34 = select %33, %cst_0, %cst : f64
      stencil.return %28, %30, %31, %34 : f64, f64, f64, f64
    }
    %18 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>, %arg15 = %1 : !stencil.temp<?x?x?xf64>, %arg16 = %17#0 : !stencil.temp<?x?x?xf64>, %arg17 = %17#1 : !stencil.temp<?x?x?xf64>, %arg18 = %17#2 : !stencil.temp<?x?x?xf64>, %arg19 = %17#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %26 = stencil.access %arg19 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = cmpf "oeq", %26, %cst : f64
      %28 = select %27, %cst_0, %cst : f64
      %29 = stencil.access %arg19 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = mulf %29, %28 : f64
      %31 = addf %26, %30 : f64
      %32 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = cmpf "ogt", %32, %cst : f64
      %34 = scf.if %33 -> (f64) {
        %37 = stencil.access %arg17 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = stencil.access %arg18 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %32, %38 : f64
        %40 = subf %37, %39 : f64
        %41 = subf %cst_0, %32 : f64
        %42 = mulf %41, %40 : f64
        scf.yield %42 : f64
      } else {
        %37 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %32, %38 : f64
        %40 = addf %37, %39 : f64
        %41 = addf %cst_0, %32 : f64
        %42 = mulf %41, %40 : f64
        scf.yield %42 : f64
      }
      %35 = mulf %34, %31 : f64
      %36 = scf.if %33 -> (f64) {
        %37 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = addf %37, %35 : f64
        scf.yield %38 : f64
      } else {
        %37 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = addf %37, %35 : f64
        scf.yield %38 : f64
      }
      stencil.return %36 : f64
    }
    %19 = stencil.apply (%arg14 = %5 : !stencil.temp<?x?x?xf64>, %arg15 = %18 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %26 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = mulf %26, %27 : f64
      stencil.return %28 : f64
    }
    %20 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>, %arg15 = %7 : !stencil.temp<?x?x?xf64>, %arg16 = %19 : !stencil.temp<?x?x?xf64>, %arg17 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %26 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = mulf %26, %27 : f64
      %29 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg16 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = subf %29, %30 : f64
      %32 = addf %28, %31 : f64
      %33 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = divf %32, %33 : f64
      stencil.return %34 : f64
    }
    %21 = stencil.apply (%arg14 = %20 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %26 = divf %cst_0, %cst_1 : f64
      %27 = divf %cst, %cst_1 : f64
      %28 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = stencil.access %arg14 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = addf %31, %32 : f64
      %34 = mulf %26, %30 : f64
      %35 = mulf %27, %33 : f64
      %36 = addf %34, %35 : f64
      stencil.return %36 : f64
    }
    %22:4 = stencil.apply (%arg14 = %20 : !stencil.temp<?x?x?xf64>, %arg15 = %21 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %26 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = subf %26, %27 : f64
      %29 = stencil.access %arg15 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = subf %29, %27 : f64
      %31 = addf %28, %30 : f64
      %32 = mulf %28, %30 : f64
      %33 = cmpf "olt", %32, %cst : f64
      %34 = select %33, %cst_0, %cst : f64
      stencil.return %28, %30, %31, %34 : f64, f64, f64, f64
    }
    %23 = stencil.apply (%arg14 = %20 : !stencil.temp<?x?x?xf64>, %arg15 = %2 : !stencil.temp<?x?x?xf64>, %arg16 = %22#0 : !stencil.temp<?x?x?xf64>, %arg17 = %22#1 : !stencil.temp<?x?x?xf64>, %arg18 = %22#2 : !stencil.temp<?x?x?xf64>, %arg19 = %22#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %26 = stencil.access %arg19 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = cmpf "oeq", %26, %cst : f64
      %28 = select %27, %cst_0, %cst : f64
      %29 = stencil.access %arg19 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = mulf %29, %28 : f64
      %31 = addf %26, %30 : f64
      %32 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = cmpf "ogt", %32, %cst : f64
      %34 = scf.if %33 -> (f64) {
        %37 = stencil.access %arg17 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = stencil.access %arg18 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %32, %38 : f64
        %40 = subf %37, %39 : f64
        %41 = subf %cst_0, %32 : f64
        %42 = mulf %41, %40 : f64
        scf.yield %42 : f64
      } else {
        %37 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %32, %38 : f64
        %40 = addf %37, %39 : f64
        %41 = addf %cst_0, %32 : f64
        %42 = mulf %41, %40 : f64
        scf.yield %42 : f64
      }
      %35 = mulf %34, %31 : f64
      %36 = scf.if %33 -> (f64) {
        %37 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = addf %37, %35 : f64
        scf.yield %38 : f64
      } else {
        %37 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %38 = addf %37, %35 : f64
        scf.yield %38 : f64
      }
      stencil.return %36 : f64
    }
    %24 = stencil.apply (%arg14 = %15 : !stencil.temp<?x?x?xf64>, %arg15 = %18 : !stencil.temp<?x?x?xf64>, %arg16 = %5 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %26 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = addf %26, %27 : f64
      %29 = mulf %28, %cst : f64
      %30 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = mulf %30, %29 : f64
      stencil.return %31 : f64
    }
    %25 = stencil.apply (%arg14 = %23 : !stencil.temp<?x?x?xf64>, %arg15 = %10 : !stencil.temp<?x?x?xf64>, %arg16 = %6 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %26 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = addf %26, %27 : f64
      %29 = mulf %28, %cst : f64
      %30 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = mulf %30, %29 : f64
      stencil.return %31 : f64
    }
    stencil.store %12 to %arg8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %20 to %arg9([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %24 to %arg10([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %18 to %arg11([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %25 to %arg12([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %10 to %arg13([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @hadvuv(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<0x?x0xf64>, %arg5: !stencil.field<0x?x0xf64>, %arg6: !stencil.field<0x?x0xf64>, %arg7: !stencil.field<0x?x0xf64>, %arg8: f64, %arg9: f64) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg6([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg7([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg4 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %3 = stencil.load %arg5 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %4 = stencil.load %arg6 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %5 = stencil.load %arg7 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %6:2 = stencil.apply (%arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %2 : !stencil.temp<0x?x0xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 3.000000e+00 : f64
      %14 = divf %cst, %cst_0 : f64
      %15 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %15, %16 : f64
      %19 = addf %18, %17 : f64
      %20 = mulf %19, %14 : f64
      %21 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %22 = mulf %20, %21 : f64
      stencil.return %20, %22 : f64, f64
    }
    %7:2 = stencil.apply (%arg10 = %1 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 0x41584DE740000000 : f64
      %14 = divf %cst, %cst_0 : f64
      %cst_1 = constant 2.500000e-01 : f64
      %15 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [1, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = addf %15, %16 : f64
      %20 = addf %17, %18 : f64
      %21 = addf %19, %20 : f64
      %22 = mulf %21, %cst_1 : f64
      %23 = mulf %22, %14 : f64
      stencil.return %22, %23 : f64, f64
    }
    %8 = stencil.apply (%arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %6#1 : !stencil.temp<?x?x?xf64>, %arg12 = %7#1 : !stencil.temp<?x?x?xf64>, %arg13 = %arg8 : f64, %arg14 = %arg9 : f64) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant -1.000000e+00 : f64
      %cst_1 = constant 3.000000e+00 : f64
      %cst_2 = constant 6.000000e+00 : f64
      %14 = divf %cst_0, %cst_2 : f64
      %cst_3 = constant -5.000000e-01 : f64
      %15 = divf %cst_0, %cst_1 : f64
      %16 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = cmpf "ogt", %16, %cst : f64
      %21 = mulf %cst_3, %17 : f64
      %22 = scf.if %20 -> (f64) {
        %31 = stencil.access %arg10 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %32 = addf %21, %18 : f64
        %33 = mulf %15, %19 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %33, %34 : f64
        %36 = addf %32, %35 : f64
        %37 = mulf %36, %16 : f64
        scf.yield %37 : f64
      } else {
        %31 = stencil.access %arg10 [2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %32 = mulf %15, %18 : f64
        %33 = addf %21, %19 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %32, %34 : f64
        %36 = addf %33, %35 : f64
        %37 = negf %16 : f64
        %38 = mulf %37, %36 : f64
        scf.yield %38 : f64
      }
      %23 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = cmpf "ogt", %23, %cst : f64
      %27 = scf.if %26 -> (f64) {
        %31 = stencil.access %arg10 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %32 = addf %21, %24 : f64
        %33 = mulf %15, %25 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %33, %34 : f64
        %36 = addf %32, %35 : f64
        %37 = mulf %36, %23 : f64
        scf.yield %37 : f64
      } else {
        %31 = stencil.access %arg10 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %32 = mulf %15, %24 : f64
        %33 = addf %21, %25 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %32, %34 : f64
        %36 = addf %33, %35 : f64
        %37 = negf %23 : f64
        %38 = mulf %37, %36 : f64
        scf.yield %38 : f64
      }
      %28 = mulf %22, %arg13 : f64
      %29 = mulf %27, %arg14 : f64
      %30 = addf %28, %29 : f64
      stencil.return %30 : f64
    }
    %9 = stencil.apply (%arg10 = %8 : !stencil.temp<?x?x?xf64>, %arg11 = %0 : !stencil.temp<?x?x?xf64>, %arg12 = %7#0 : !stencil.temp<?x?x?xf64>, %arg13 = %4 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = mulf %14, %15 : f64
      %19 = mulf %18, %16 : f64
      %20 = addf %19, %17 : f64
      stencil.return %20 : f64
    }
    %10:2 = stencil.apply (%arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %3 : !stencil.temp<0x?x0xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 2.500000e-01 : f64
      %14 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg10 [-1, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %14, %15 : f64
      %19 = addf %16, %17 : f64
      %20 = addf %18, %19 : f64
      %21 = mulf %20, %cst : f64
      %22 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %23 = mulf %21, %22 : f64
      stencil.return %21, %23 : f64, f64
    }
    %11:2 = stencil.apply (%arg10 = %1 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 0x41584DE740000000 : f64
      %14 = divf %cst, %cst_0 : f64
      %cst_1 = constant 3.000000e+00 : f64
      %15 = divf %cst, %cst_1 : f64
      %16 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = addf %16, %17 : f64
      %20 = addf %19, %18 : f64
      %21 = mulf %20, %15 : f64
      %22 = mulf %21, %14 : f64
      stencil.return %21, %22 : f64, f64
    }
    %12 = stencil.apply (%arg10 = %1 : !stencil.temp<?x?x?xf64>, %arg11 = %10#1 : !stencil.temp<?x?x?xf64>, %arg12 = %11#1 : !stencil.temp<?x?x?xf64>, %arg13 = %arg8 : f64, %arg14 = %arg9 : f64) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant -1.000000e+00 : f64
      %cst_1 = constant 3.000000e+00 : f64
      %cst_2 = constant 6.000000e+00 : f64
      %14 = divf %cst_0, %cst_2 : f64
      %cst_3 = constant -5.000000e-01 : f64
      %15 = divf %cst_0, %cst_1 : f64
      %16 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = cmpf "ogt", %16, %cst : f64
      %21 = mulf %cst_3, %17 : f64
      %22 = scf.if %20 -> (f64) {
        %31 = stencil.access %arg10 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %32 = addf %21, %18 : f64
        %33 = mulf %15, %19 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %33, %34 : f64
        %36 = addf %32, %35 : f64
        %37 = mulf %36, %16 : f64
        scf.yield %37 : f64
      } else {
        %31 = stencil.access %arg10 [2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %32 = mulf %15, %18 : f64
        %33 = addf %21, %19 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %32, %34 : f64
        %36 = addf %33, %35 : f64
        %37 = negf %16 : f64
        %38 = mulf %37, %36 : f64
        scf.yield %38 : f64
      }
      %23 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = cmpf "ogt", %23, %cst : f64
      %27 = scf.if %26 -> (f64) {
        %31 = stencil.access %arg10 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %32 = addf %21, %24 : f64
        %33 = mulf %15, %25 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %33, %34 : f64
        %36 = addf %32, %35 : f64
        %37 = mulf %36, %23 : f64
        scf.yield %37 : f64
      } else {
        %31 = stencil.access %arg10 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %32 = mulf %15, %24 : f64
        %33 = addf %21, %25 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %32, %34 : f64
        %36 = addf %33, %35 : f64
        %37 = negf %23 : f64
        %38 = mulf %37, %36 : f64
        scf.yield %38 : f64
      }
      %28 = mulf %22, %arg13 : f64
      %29 = mulf %27, %arg14 : f64
      %30 = addf %28, %29 : f64
      stencil.return %30 : f64
    }
    %13 = stencil.apply (%arg10 = %12 : !stencil.temp<?x?x?xf64>, %arg11 = %10#0 : !stencil.temp<?x?x?xf64>, %arg12 = %5 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %14, %14 : f64
      %18 = mulf %17, %15 : f64
      %19 = subf %16, %18 : f64
      stencil.return %19 : f64
    }
    stencil.store %9 to %arg2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %13 to %arg3([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @hadvuv5th(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<0x?x0xf64>, %arg5: !stencil.field<0x?x0xf64>, %arg6: !stencil.field<0x?x0xf64>, %arg7: !stencil.field<0x?x0xf64>, %arg8: f64, %arg9: f64) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg6([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg7([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg4 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %3 = stencil.load %arg5 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %4 = stencil.load %arg6 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %5 = stencil.load %arg7 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %6:2 = stencil.apply (%arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %2 : !stencil.temp<0x?x0xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 3.000000e+00 : f64
      %14 = divf %cst, %cst_0 : f64
      %15 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %15, %16 : f64
      %19 = addf %18, %17 : f64
      %20 = mulf %19, %14 : f64
      %21 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %22 = mulf %20, %21 : f64
      stencil.return %20, %22 : f64, f64
    }
    %7:2 = stencil.apply (%arg10 = %1 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 0x41584DE740000000 : f64
      %14 = divf %cst, %cst_0 : f64
      %cst_1 = constant 2.500000e-01 : f64
      %15 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [1, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = addf %15, %16 : f64
      %20 = addf %17, %18 : f64
      %21 = addf %19, %20 : f64
      %22 = mulf %21, %cst_1 : f64
      %23 = mulf %22, %14 : f64
      stencil.return %22, %23 : f64, f64
    }
    %8 = stencil.apply (%arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %6#1 : !stencil.temp<?x?x?xf64>, %arg12 = %7#1 : !stencil.temp<?x?x?xf64>, %arg13 = %arg8 : f64, %arg14 = %arg9 : f64) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %cst_1 = constant -1.000000e+00 : f64
      %cst_2 = constant 2.000000e+00 : f64
      %cst_3 = constant 3.000000e+00 : f64
      %cst_4 = constant 4.000000e+00 : f64
      %cst_5 = constant 2.000000e+01 : f64
      %cst_6 = constant 3.000000e+01 : f64
      %14 = divf %cst_0, %cst_6 : f64
      %15 = divf %cst_1, %cst_4 : f64
      %16 = divf %cst_1, %cst_3 : f64
      %17 = divf %cst_1, %cst_2 : f64
      %18 = divf %cst_0, %cst_5 : f64
      %19 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg10 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg10 [2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = cmpf "ogt", %19, %cst : f64
      %26 = mulf %16, %20 : f64
      %27 = scf.if %25 -> (f64) {
        %38 = stencil.access %arg10 [-3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %15, %21 : f64
        %40 = addf %26, %22 : f64
        %41 = mulf %17, %23 : f64
        %42 = mulf %18, %24 : f64
        %43 = mulf %14, %38 : f64
        %44 = addf %39, %41 : f64
        %45 = addf %42, %43 : f64
        %46 = addf %40, %44 : f64
        %47 = addf %45, %46 : f64
        %48 = mulf %47, %19 : f64
        scf.yield %48 : f64
      } else {
        %38 = stencil.access %arg10 [3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %18, %21 : f64
        %40 = mulf %17, %22 : f64
        %41 = addf %26, %23 : f64
        %42 = mulf %15, %24 : f64
        %43 = mulf %14, %38 : f64
        %44 = addf %39, %40 : f64
        %45 = addf %42, %43 : f64
        %46 = addf %41, %44 : f64
        %47 = addf %45, %46 : f64
        %48 = negf %19 : f64
        %49 = mulf %48, %47 : f64
        scf.yield %49 : f64
      }
      %28 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg10 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg10 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = cmpf "ogt", %28, %cst : f64
      %34 = scf.if %33 -> (f64) {
        %38 = stencil.access %arg10 [0, -3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %15, %29 : f64
        %40 = addf %26, %30 : f64
        %41 = mulf %17, %31 : f64
        %42 = mulf %18, %32 : f64
        %43 = mulf %14, %38 : f64
        %44 = addf %39, %41 : f64
        %45 = addf %42, %43 : f64
        %46 = addf %40, %44 : f64
        %47 = addf %45, %46 : f64
        %48 = mulf %47, %28 : f64
        scf.yield %48 : f64
      } else {
        %38 = stencil.access %arg10 [0, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %18, %29 : f64
        %40 = mulf %17, %30 : f64
        %41 = addf %26, %31 : f64
        %42 = mulf %15, %32 : f64
        %43 = mulf %14, %38 : f64
        %44 = addf %39, %40 : f64
        %45 = addf %42, %43 : f64
        %46 = addf %41, %44 : f64
        %47 = addf %45, %46 : f64
        %48 = negf %28 : f64
        %49 = mulf %48, %47 : f64
        scf.yield %49 : f64
      }
      %35 = mulf %27, %arg13 : f64
      %36 = mulf %34, %arg14 : f64
      %37 = addf %35, %36 : f64
      stencil.return %37 : f64
    }
    %9 = stencil.apply (%arg10 = %8 : !stencil.temp<?x?x?xf64>, %arg11 = %0 : !stencil.temp<?x?x?xf64>, %arg12 = %7#0 : !stencil.temp<?x?x?xf64>, %arg13 = %4 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = mulf %14, %15 : f64
      %19 = mulf %18, %16 : f64
      %20 = addf %19, %17 : f64
      stencil.return %20 : f64
    }
    %10:2 = stencil.apply (%arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %3 : !stencil.temp<0x?x0xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 2.500000e-01 : f64
      %14 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg10 [-1, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %14, %15 : f64
      %19 = addf %16, %17 : f64
      %20 = addf %18, %19 : f64
      %21 = mulf %20, %cst : f64
      %22 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %23 = mulf %21, %22 : f64
      stencil.return %21, %23 : f64, f64
    }
    %11:2 = stencil.apply (%arg10 = %1 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 0x41584DE740000000 : f64
      %14 = divf %cst, %cst_0 : f64
      %cst_1 = constant 3.000000e+00 : f64
      %15 = divf %cst, %cst_1 : f64
      %16 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = addf %16, %17 : f64
      %20 = addf %19, %18 : f64
      %21 = mulf %20, %15 : f64
      %22 = mulf %21, %14 : f64
      stencil.return %21, %22 : f64, f64
    }
    %12 = stencil.apply (%arg10 = %1 : !stencil.temp<?x?x?xf64>, %arg11 = %10#1 : !stencil.temp<?x?x?xf64>, %arg12 = %11#1 : !stencil.temp<?x?x?xf64>, %arg13 = %arg8 : f64, %arg14 = %arg9 : f64) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %cst_1 = constant -1.000000e+00 : f64
      %cst_2 = constant 2.000000e+00 : f64
      %cst_3 = constant 3.000000e+00 : f64
      %cst_4 = constant 4.000000e+00 : f64
      %cst_5 = constant 2.000000e+01 : f64
      %cst_6 = constant 3.000000e+01 : f64
      %14 = divf %cst_0, %cst_6 : f64
      %15 = divf %cst_1, %cst_4 : f64
      %16 = divf %cst_1, %cst_3 : f64
      %17 = divf %cst_1, %cst_2 : f64
      %18 = divf %cst_0, %cst_5 : f64
      %19 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg10 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg10 [2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = cmpf "ogt", %19, %cst : f64
      %26 = mulf %16, %20 : f64
      %27 = scf.if %25 -> (f64) {
        %38 = stencil.access %arg10 [-3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %15, %21 : f64
        %40 = addf %26, %22 : f64
        %41 = mulf %17, %23 : f64
        %42 = mulf %18, %24 : f64
        %43 = mulf %14, %38 : f64
        %44 = addf %39, %41 : f64
        %45 = addf %42, %43 : f64
        %46 = addf %40, %44 : f64
        %47 = addf %45, %46 : f64
        %48 = mulf %47, %19 : f64
        scf.yield %48 : f64
      } else {
        %38 = stencil.access %arg10 [3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = mulf %18, %21 : f64
        %40 = mulf %17, %22 : f64
        %41 = addf %26, %23 : f64
        %42 = mulf %15, %24 : f64
        %43 = mulf %14, %38 : f64
        %44 = addf %39, %40 : f64
        %45 = addf %42, %43 : f64
        %46 = addf %41, %44 : f64
        %47 = addf %45, %46 : f64
        %48 = negf %19 : f64
        %49 = mulf %48, %47 : f64
        scf.yield %49 : f64
      }
      %28 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg10 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg10 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = cmpf "ogt", %28, %cst : f64
      %34 = scf.if %33 -> (f64) {
        %38 = stencil.access %arg10 [0, -3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = addf %26, %30 : f64
        %40 = mulf %15, %29 : f64
        %41 = mulf %17, %31 : f64
        %42 = mulf %18, %32 : f64
        %43 = mulf %14, %38 : f64
        %44 = addf %40, %41 : f64
        %45 = addf %42, %43 : f64
        %46 = addf %39, %44 : f64
        %47 = addf %45, %46 : f64
        %48 = mulf %47, %28 : f64
        scf.yield %48 : f64
      } else {
        %38 = stencil.access %arg10 [0, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %39 = addf %26, %31 : f64
        %40 = mulf %18, %29 : f64
        %41 = mulf %17, %30 : f64
        %42 = mulf %15, %32 : f64
        %43 = mulf %14, %38 : f64
        %44 = addf %40, %41 : f64
        %45 = addf %42, %43 : f64
        %46 = addf %39, %44 : f64
        %47 = addf %45, %46 : f64
        %48 = negf %28 : f64
        %49 = mulf %48, %47 : f64
        scf.yield %49 : f64
      }
      %35 = mulf %27, %arg13 : f64
      %36 = mulf %34, %arg14 : f64
      %37 = addf %35, %36 : f64
      stencil.return %37 : f64
    }
    %13 = stencil.apply (%arg10 = %12 : !stencil.temp<?x?x?xf64>, %arg11 = %10#0 : !stencil.temp<?x?x?xf64>, %arg12 = %5 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %14, %14 : f64
      %18 = mulf %17, %15 : f64
      %19 = subf %16, %18 : f64
      stencil.return %19 : f64
    }
    stencil.store %9 to %arg2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %13 to %arg3([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @hdiff(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %6 = stencil.access %arg3 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = stencil.access %arg3 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg3 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = addf %6, %7 : f64
      %12 = addf %8, %9 : f64
      %13 = addf %11, %12 : f64
      %cst = constant -4.000000e+00 : f64
      %14 = mulf %10, %cst : f64
      %15 = addf %14, %13 : f64
      stencil.return %15 : f64
    }
    %3 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %6 = stencil.access %arg4 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = subf %6, %7 : f64
      %9 = stencil.access %arg3 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = subf %9, %10 : f64
      %12 = mulf %8, %11 : f64
      %cst = constant 0.000000e+00 : f64
      %13 = cmpf "ogt", %12, %cst : f64
      %14 = select %13, %cst, %8 : f64
      stencil.return %14 : f64
    }
    %4 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %6 = stencil.access %arg4 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = subf %6, %7 : f64
      %9 = stencil.access %arg3 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = subf %9, %10 : f64
      %12 = mulf %8, %11 : f64
      %cst = constant 0.000000e+00 : f64
      %13 = cmpf "ogt", %12, %cst : f64
      %14 = select %13, %cst, %8 : f64
      stencil.return %14 : f64
    }
    %5 = stencil.apply (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %3 : !stencil.temp<?x?x?xf64>, %arg5 = %4 : !stencil.temp<?x?x?xf64>, %arg6 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %6 = stencil.access %arg4 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = subf %6, %7 : f64
      %9 = stencil.access %arg5 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = subf %9, %10 : f64
      %12 = addf %8, %11 : f64
      %13 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = mulf %13, %12 : f64
      %15 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = addf %14, %15 : f64
      stencil.return %16 : f64
    }
    stencil.store %5 to %arg2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @hdiffsa(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<0x?x0xf64>, %arg4: !stencil.field<0x?x0xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg3 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %3 = stencil.load %arg4 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %4 = stencil.apply (%arg5 = %0 : !stencil.temp<?x?x?xf64>, %arg6 = %2 : !stencil.temp<0x?x0xf64>, %arg7 = %3 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %8 = stencil.access %arg5 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg5 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = addf %8, %9 : f64
      %cst = constant -2.000000e+00 : f64
      %12 = mulf %10, %cst : f64
      %13 = addf %11, %12 : f64
      %14 = stencil.access %arg5 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg5 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %17 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %18 = subf %14, %10 : f64
      %19 = subf %15, %10 : f64
      %20 = mulf %18, %16 : f64
      %21 = mulf %19, %17 : f64
      %22 = addf %20, %13 : f64
      %23 = addf %22, %21 : f64
      stencil.return %23 : f64
    }
    %5 = stencil.apply (%arg5 = %0 : !stencil.temp<?x?x?xf64>, %arg6 = %4 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %8 = stencil.access %arg6 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = subf %8, %9 : f64
      %11 = stencil.access %arg5 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = subf %11, %12 : f64
      %14 = mulf %10, %13 : f64
      %cst = constant 0.000000e+00 : f64
      %15 = cmpf "ogt", %14, %cst : f64
      %16 = select %15, %cst, %10 : f64
      stencil.return %16 : f64
    }
    %6 = stencil.apply (%arg5 = %0 : !stencil.temp<?x?x?xf64>, %arg6 = %4 : !stencil.temp<?x?x?xf64>, %arg7 = %2 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %8 = stencil.access %arg6 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = subf %8, %9 : f64
      %11 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %12 = mulf %10, %11 : f64
      %13 = stencil.access %arg5 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = mulf %12, %15 : f64
      %cst = constant 0.000000e+00 : f64
      %17 = cmpf "ogt", %16, %cst : f64
      %18 = select %17, %cst, %12 : f64
      stencil.return %18 : f64
    }
    %7 = stencil.apply (%arg5 = %0 : !stencil.temp<?x?x?xf64>, %arg6 = %5 : !stencil.temp<?x?x?xf64>, %arg7 = %6 : !stencil.temp<?x?x?xf64>, %arg8 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %8 = stencil.access %arg6 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg6 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = subf %8, %9 : f64
      %11 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = subf %11, %12 : f64
      %14 = addf %10, %13 : f64
      %15 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = mulf %15, %14 : f64
      %17 = stencil.access %arg5 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %16, %17 : f64
      stencil.return %18 : f64
    }
    stencil.store %7 to %arg2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @hdiffsmag(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<0x?x0xf64>, %arg6: !stencil.field<0x?x0xf64>, %arg7: !stencil.field<0x?x0xf64>, %arg8: !stencil.field<0x?x0xf64>, %arg9: !stencil.field<0x?x0xf64>, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg6([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg7([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg8([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg9([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg2 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.load %arg5 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %4 = stencil.load %arg6 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %5 = stencil.load %arg7 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %6 = stencil.load %arg8 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %7 = stencil.load %arg9 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %8 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>, %arg15 = %1 : !stencil.temp<?x?x?xf64>, %arg16 = %7 : !stencil.temp<0x?x0xf64>, %arg17 = %arg10 : f64, %arg18 = %arg11 : f64) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 0x41584DE740000000 : f64
      %14 = divf %cst, %cst_0 : f64
      %15 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %16 = mulf %15, %arg18 : f64
      %17 = mulf %arg17, %14 : f64
      %18 = stencil.access %arg15 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = subf %18, %19 : f64
      %21 = mulf %20, %17 : f64
      %22 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = subf %22, %23 : f64
      %25 = mulf %24, %16 : f64
      %26 = subf %21, %25 : f64
      %27 = mulf %26, %26 : f64
      stencil.return %27 : f64
    }
    %9 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>, %arg15 = %1 : !stencil.temp<?x?x?xf64>, %arg16 = %7 : !stencil.temp<0x?x0xf64>, %arg17 = %arg10 : f64, %arg18 = %arg11 : f64) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 0x41584DE740000000 : f64
      %14 = divf %cst, %cst_0 : f64
      %15 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %16 = mulf %15, %arg18 : f64
      %17 = mulf %arg17, %14 : f64
      %18 = stencil.access %arg15 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = subf %18, %19 : f64
      %21 = mulf %20, %16 : f64
      %22 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = subf %22, %23 : f64
      %25 = mulf %24, %17 : f64
      %26 = addf %25, %21 : f64
      %27 = mulf %26, %26 : f64
      stencil.return %27 : f64
    }
    %10 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>, %arg15 = %5 : !stencil.temp<0x?x0xf64>, %arg16 = %6 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = addf %14, %15 : f64
      %cst = constant -2.000000e+00 : f64
      %18 = mulf %16, %cst : f64
      %19 = addf %17, %18 : f64
      %20 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %23 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %24 = subf %20, %16 : f64
      %25 = subf %21, %16 : f64
      %26 = mulf %24, %22 : f64
      %27 = mulf %25, %23 : f64
      %28 = addf %26, %19 : f64
      %29 = addf %28, %27 : f64
      stencil.return %29 : f64
    }
    %11 = stencil.apply (%arg14 = %1 : !stencil.temp<?x?x?xf64>, %arg15 = %3 : !stencil.temp<0x?x0xf64>, %arg16 = %4 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = addf %14, %15 : f64
      %cst = constant -2.000000e+00 : f64
      %18 = mulf %16, %cst : f64
      %19 = addf %17, %18 : f64
      %20 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %23 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %24 = subf %20, %16 : f64
      %25 = subf %21, %16 : f64
      %26 = mulf %24, %22 : f64
      %27 = mulf %25, %23 : f64
      %28 = addf %26, %19 : f64
      %29 = addf %28, %27 : f64
      stencil.return %29 : f64
    }
    %12 = stencil.apply (%arg14 = %0 : !stencil.temp<?x?x?xf64>, %arg15 = %8 : !stencil.temp<?x?x?xf64>, %arg16 = %9 : !stencil.temp<?x?x?xf64>, %arg17 = %10 : !stencil.temp<?x?x?xf64>, %arg18 = %2 : !stencil.temp<?x?x?xf64>, %arg19 = %arg13 : f64, %arg20 = %arg12 : f64) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = mulf %arg19, %14 : f64
      %cst = constant 5.000000e-01 : f64
      %cst_0 = constant 0.000000e+00 : f64
      %16 = stencil.access %arg15 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %16, %17 : f64
      %19 = mulf %18, %cst : f64
      %20 = stencil.access %arg16 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = addf %20, %21 : f64
      %23 = mulf %22, %cst : f64
      %24 = addf %19, %23 : f64
      %25 = sqrt %24 : f64
      %26 = mulf %25, %arg20 : f64
      %27 = subf %26, %15 : f64
      %28 = cmpf "ogt", %27, %cst_0 : f64
      %29 = select %28, %27, %cst_0 : f64
      %30 = cmpf "olt", %29, %cst : f64
      %31 = select %30, %29, %cst : f64
      %32 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = mulf %31, %32 : f64
      %35 = addf %34, %33 : f64
      stencil.return %35 : f64
    }
    %13 = stencil.apply (%arg14 = %1 : !stencil.temp<?x?x?xf64>, %arg15 = %8 : !stencil.temp<?x?x?xf64>, %arg16 = %9 : !stencil.temp<?x?x?xf64>, %arg17 = %11 : !stencil.temp<?x?x?xf64>, %arg18 = %2 : !stencil.temp<?x?x?xf64>, %arg19 = %arg13 : f64, %arg20 = %arg12 : f64) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = mulf %arg19, %14 : f64
      %cst = constant 5.000000e-01 : f64
      %cst_0 = constant 0.000000e+00 : f64
      %16 = stencil.access %arg15 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %16, %17 : f64
      %19 = mulf %18, %cst : f64
      %20 = stencil.access %arg16 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = addf %20, %21 : f64
      %23 = mulf %22, %cst : f64
      %24 = addf %19, %23 : f64
      %25 = sqrt %24 : f64
      %26 = mulf %25, %arg20 : f64
      %27 = subf %26, %15 : f64
      %28 = cmpf "ogt", %27, %cst_0 : f64
      %29 = select %28, %27, %cst_0 : f64
      %30 = cmpf "olt", %29, %cst : f64
      %31 = select %30, %29, %cst : f64
      %32 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = mulf %31, %32 : f64
      %35 = addf %34, %33 : f64
      stencil.return %35 : f64
    }
    stencil.store %12 to %arg3([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %13 to %arg4([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @laplace1(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %2 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %3 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %4 = stencil.access %arg2 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %5 = stencil.access %arg2 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = addf %2, %3 : f64
      %8 = addf %4, %5 : f64
      %9 = addf %7, %8 : f64
      %cst = constant -4.000000e+00 : f64
      %10 = mulf %6, %cst : f64
      %11 = addf %10, %9 : f64
      stencil.return %11 : f64
    }
    stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @nh_p_grad(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>, %arg9: !stencil.field<?x?x?xf64>, %arg10: f64) attributes {stencil.program} {
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
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg2 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.load %arg3 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %4 = stencil.load %arg4 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %5 = stencil.load %arg5 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %6 = stencil.load %arg6 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %7 = stencil.load %arg7 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %8 = stencil.apply (%arg11 = %6 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %13 = stencil.access %arg11 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      stencil.return %15 : f64
    }
    %9 = stencil.apply (%arg11 = %8 : !stencil.temp<?x?x?xf64>, %arg12 = %4 : !stencil.temp<?x?x?xf64>, %arg13 = %6 : !stencil.temp<?x?x?xf64>, %arg14 = %arg10 : f64) -> !stencil.temp<?x?x?xf64> {
      %13 = stencil.access %arg12 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg12 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg13 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = mulf %15, %18 : f64
      %20 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg12 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = mulf %22, %25 : f64
      %27 = addf %19, %26 : f64
      %28 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg11 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = divf %arg14, %30 : f64
      %32 = mulf %27, %31 : f64
      stencil.return %32 : f64
    }
    %10 = stencil.apply (%arg11 = %8 : !stencil.temp<?x?x?xf64>, %arg12 = %4 : !stencil.temp<?x?x?xf64>, %arg13 = %6 : !stencil.temp<?x?x?xf64>, %arg14 = %arg10 : f64) -> !stencil.temp<?x?x?xf64> {
      %13 = stencil.access %arg12 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg12 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg13 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = mulf %15, %18 : f64
      %20 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg12 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = mulf %22, %25 : f64
      %27 = addf %19, %26 : f64
      %28 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg11 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = divf %arg14, %30 : f64
      %32 = mulf %27, %31 : f64
      stencil.return %32 : f64
    }
    %11 = stencil.apply (%arg11 = %0 : !stencil.temp<?x?x?xf64>, %arg12 = %7 : !stencil.temp<?x?x?xf64>, %arg13 = %4 : !stencil.temp<?x?x?xf64>, %arg14 = %5 : !stencil.temp<?x?x?xf64>, %arg15 = %9 : !stencil.temp<?x?x?xf64>, %arg16 = %2 : !stencil.temp<?x?x?xf64>, %arg17 = %arg10 : f64) -> !stencil.temp<?x?x?xf64> {
      %13 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg14 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = mulf %15, %18 : f64
      %20 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg13 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = stencil.access %arg14 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = mulf %22, %25 : f64
      %27 = addf %19, %26 : f64
      %28 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg12 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = divf %arg17, %30 : f64
      %32 = mulf %27, %31 : f64
      %33 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = addf %33, %34 : f64
      %36 = addf %32, %35 : f64
      %37 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %38 = mulf %36, %37 : f64
      stencil.return %38 : f64
    }
    %12 = stencil.apply (%arg11 = %1 : !stencil.temp<?x?x?xf64>, %arg12 = %7 : !stencil.temp<?x?x?xf64>, %arg13 = %4 : !stencil.temp<?x?x?xf64>, %arg14 = %5 : !stencil.temp<?x?x?xf64>, %arg15 = %10 : !stencil.temp<?x?x?xf64>, %arg16 = %3 : !stencil.temp<?x?x?xf64>, %arg17 = %arg10 : f64) -> !stencil.temp<?x?x?xf64> {
      %13 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg14 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = mulf %15, %18 : f64
      %20 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg13 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = stencil.access %arg14 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = mulf %22, %25 : f64
      %27 = addf %19, %26 : f64
      %28 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg12 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = divf %arg17, %30 : f64
      %32 = mulf %27, %31 : f64
      %33 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = addf %33, %34 : f64
      %36 = addf %32, %35 : f64
      %37 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %38 = mulf %36, %37 : f64
      stencil.return %38 : f64
    }
    stencil.store %11 to %arg8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %12 to %arg9([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @p_grad_c(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>, %arg9: f64) attributes {stencil.program} {
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
    %7 = stencil.apply (%arg10 = %4 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %10 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      stencil.return %10 : f64
    }
    %8 = stencil.apply (%arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %2 : !stencil.temp<?x?x?xf64>, %arg12 = %7 : !stencil.temp<?x?x?xf64>, %arg13 = %5 : !stencil.temp<?x?x?xf64>, %arg14 = %6 : !stencil.temp<?x?x?xf64>, %arg15 = %arg9 : f64) -> !stencil.temp<?x?x?xf64> {
      %10 = stencil.access %arg13 [-1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = subf %10, %11 : f64
      %13 = stencil.access %arg14 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = mulf %12, %15 : f64
      %17 = stencil.access %arg13 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = subf %17, %18 : f64
      %20 = stencil.access %arg14 [-1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = mulf %19, %22 : f64
      %24 = addf %16, %23 : f64
      %25 = stencil.access %arg12 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %25, %26 : f64
      %28 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = mulf %28, %arg15 : f64
      %30 = divf %29, %27 : f64
      %31 = mulf %30, %24 : f64
      %32 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = addf %31, %32 : f64
      stencil.return %33 : f64
    }
    %9 = stencil.apply (%arg10 = %1 : !stencil.temp<?x?x?xf64>, %arg11 = %3 : !stencil.temp<?x?x?xf64>, %arg12 = %7 : !stencil.temp<?x?x?xf64>, %arg13 = %5 : !stencil.temp<?x?x?xf64>, %arg14 = %6 : !stencil.temp<?x?x?xf64>, %arg15 = %arg9 : f64) -> !stencil.temp<?x?x?xf64> {
      %10 = stencil.access %arg13 [0, -1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = subf %10, %11 : f64
      %13 = stencil.access %arg14 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = mulf %12, %15 : f64
      %17 = stencil.access %arg13 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = subf %17, %18 : f64
      %20 = stencil.access %arg14 [0, -1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = mulf %19, %22 : f64
      %24 = addf %16, %23 : f64
      %25 = stencil.access %arg12 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %25, %26 : f64
      %28 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = mulf %28, %arg15 : f64
      %30 = divf %29, %27 : f64
      %31 = mulf %30, %24 : f64
      %32 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = addf %31, %32 : f64
      stencil.return %33 : f64
    }
    stencil.store %8 to %arg7([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %9 to %arg8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @laplace2(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.apply (%arg2 = %0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %2 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %3 = cmpf "ogt", %2, %cst : f64
      %4 = scf.if %3 -> (f64) {
        %5 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %6 = subf %5, %2 : f64
        scf.yield %6 : f64
      } else {
        %5 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %6 = subf %2, %5 : f64
        scf.yield %6 : f64
      }
      stencil.return %4 : f64
    }
    stencil.store %1 to %arg1([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }

  func @uvbke(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: f64) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg2 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.load %arg3 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %4 = stencil.apply (%arg7 = %0 : !stencil.temp<?x?x?xf64>, %arg8 = %1 : !stencil.temp<?x?x?xf64>, %arg9 = %2 : !stencil.temp<?x?x?xf64>, %arg10 = %3 : !stencil.temp<?x?x?xf64>, %arg11 = %arg6 : f64) -> !stencil.temp<?x?x?xf64> {
      %6 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = addf %6, %7 : f64
      %9 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = mulf %8, %9 : f64
      %11 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = addf %11, %12 : f64
      %14 = subf %13, %10 : f64
      %15 = mulf %arg11, %14 : f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %16, %15 : f64
      stencil.return %17 : f64
    }
    %5 = stencil.apply (%arg7 = %0 : !stencil.temp<?x?x?xf64>, %arg8 = %1 : !stencil.temp<?x?x?xf64>, %arg9 = %2 : !stencil.temp<?x?x?xf64>, %arg10 = %3 : !stencil.temp<?x?x?xf64>, %arg11 = %arg6 : f64) -> !stencil.temp<?x?x?xf64> {
      %6 = stencil.access %arg7 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg7 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = addf %6, %7 : f64
      %9 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %10 = mulf %8, %9 : f64
      %11 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = addf %11, %12 : f64
      %14 = subf %13, %10 : f64
      %15 = mulf %arg11, %14 : f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %16, %15 : f64
      stencil.return %17 : f64
    }
    stencil.store %4 to %arg4([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %5 to %arg5([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
