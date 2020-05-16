

module {
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
}
