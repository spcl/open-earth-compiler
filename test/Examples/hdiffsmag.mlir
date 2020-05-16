

module {
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
}
