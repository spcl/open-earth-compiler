

module {
  func @hadvuv(%arg0: !stencil.field<ijk,f64>, %arg1: !stencil.field<ijk,f64>, %arg2: !stencil.field<ijk,f64>, %arg3: !stencil.field<ijk,f64>, %arg4: !stencil.field<j,f64>, %arg5: !stencil.field<j,f64>, %arg6: !stencil.field<j,f64>, %arg7: !stencil.field<j,f64>, %arg8: f64, %arg9: f64) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<j,f64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<j,f64>
    stencil.assert %arg6([-4, -4, -4] : [68, 68, 68]) : !stencil.field<j,f64>
    stencil.assert %arg7([-4, -4, -4] : [68, 68, 68]) : !stencil.field<j,f64>
    %0 = stencil.load %arg0 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %1 = stencil.load %arg1 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %2 = stencil.load %arg4 : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
    %3 = stencil.load %arg5 : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
    %4 = stencil.load %arg6 : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
    %5 = stencil.load %arg7 : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
    %6:2 = stencil.apply (%arg10 = %0 : !stencil.temp<ijk,f64>, %arg11 = %2 : !stencil.temp<j,f64>) -> (!stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 3.000000e+00 : f64
      %14 = divf %cst, %cst_0 : f64
      %15 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = addf %15, %16 : f64
      %19 = addf %18, %17 : f64
      %20 = mulf %19, %14 : f64
      %21 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %22 = mulf %20, %21 : f64
      stencil.return %20, %22 : f64, f64
    }
    %7:2 = stencil.apply (%arg10 = %1 : !stencil.temp<ijk,f64>) -> (!stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 0x41584DE740000000 : f64
      %14 = divf %cst, %cst_0 : f64
      %cst_1 = constant 2.500000e-01 : f64
      %15 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg10 [1, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %19 = addf %15, %16 : f64
      %20 = addf %17, %18 : f64
      %21 = addf %19, %20 : f64
      %22 = mulf %21, %cst_1 : f64
      %23 = mulf %22, %14 : f64
      stencil.return %22, %23 : f64, f64
    }
    %8 = stencil.apply (%arg10 = %0 : !stencil.temp<ijk,f64>, %arg11 = %6#1 : !stencil.temp<ijk,f64>, %arg12 = %7#1 : !stencil.temp<ijk,f64>, %arg13 = %arg8 : f64, %arg14 = %arg9 : f64) -> !stencil.temp<ijk,f64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant -1.000000e+00 : f64
      %cst_1 = constant 3.000000e+00 : f64
      %cst_2 = constant 6.000000e+00 : f64
      %14 = divf %cst_0, %cst_2 : f64
      %cst_3 = constant -5.000000e-01 : f64
      %15 = divf %cst_0, %cst_1 : f64
      %16 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %19 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %20 = cmpf "ogt", %16, %cst : f64
      %21 = mulf %cst_3, %17 : f64
      %22 = loop.if %20 -> (f64) {
        %31 = stencil.access %arg10 [-2, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
        %32 = addf %21, %18 : f64
        %33 = mulf %15, %19 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %33, %34 : f64
        %36 = addf %32, %35 : f64
        %37 = mulf %36, %16 : f64
        loop.yield %37 : f64
      } else {
        %31 = stencil.access %arg10 [2, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
        %32 = mulf %15, %18 : f64
        %33 = addf %21, %19 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %32, %34 : f64
        %36 = addf %33, %35 : f64
        %37 = negf %16 : f64
        %38 = mulf %37, %36 : f64
        loop.yield %38 : f64
      }
      %23 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %24 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %25 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %26 = cmpf "ogt", %23, %cst : f64
      %27 = loop.if %26 -> (f64) {
        %31 = stencil.access %arg10 [0, -2, 0] : (!stencil.temp<ijk,f64>) -> f64
        %32 = addf %21, %24 : f64
        %33 = mulf %15, %25 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %33, %34 : f64
        %36 = addf %32, %35 : f64
        %37 = mulf %36, %23 : f64
        loop.yield %37 : f64
      } else {
        %31 = stencil.access %arg10 [0, 2, 0] : (!stencil.temp<ijk,f64>) -> f64
        %32 = mulf %15, %24 : f64
        %33 = addf %21, %25 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %32, %34 : f64
        %36 = addf %33, %35 : f64
        %37 = negf %23 : f64
        %38 = mulf %37, %36 : f64
        loop.yield %38 : f64
      }
      %28 = mulf %22, %arg13 : f64
      %29 = mulf %27, %arg14 : f64
      %30 = addf %28, %29 : f64
      stencil.return %30 : f64
    }
    %9 = stencil.apply (%arg10 = %8 : !stencil.temp<ijk,f64>, %arg11 = %0 : !stencil.temp<ijk,f64>, %arg12 = %7#0 : !stencil.temp<ijk,f64>, %arg13 = %4 : !stencil.temp<j,f64>) -> !stencil.temp<ijk,f64> {
      %14 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %15 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = mulf %14, %15 : f64
      %19 = mulf %18, %16 : f64
      %20 = addf %19, %17 : f64
      stencil.return %20 : f64
    }
    %10:2 = stencil.apply (%arg10 = %0 : !stencil.temp<ijk,f64>, %arg11 = %3 : !stencil.temp<j,f64>) -> (!stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>) {
      %cst = constant 2.500000e-01 : f64
      %14 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %15 = stencil.access %arg10 [-1, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = addf %14, %15 : f64
      %19 = addf %16, %17 : f64
      %20 = addf %18, %19 : f64
      %21 = mulf %20, %cst : f64
      %22 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %23 = mulf %21, %22 : f64
      stencil.return %21, %23 : f64, f64
    }
    %11:2 = stencil.apply (%arg10 = %1 : !stencil.temp<ijk,f64>) -> (!stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 0x41584DE740000000 : f64
      %14 = divf %cst, %cst_0 : f64
      %cst_1 = constant 3.000000e+00 : f64
      %15 = divf %cst, %cst_1 : f64
      %16 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %19 = addf %16, %17 : f64
      %20 = addf %19, %18 : f64
      %21 = mulf %20, %15 : f64
      %22 = mulf %21, %14 : f64
      stencil.return %21, %22 : f64, f64
    }
    %12 = stencil.apply (%arg10 = %1 : !stencil.temp<ijk,f64>, %arg11 = %10#1 : !stencil.temp<ijk,f64>, %arg12 = %11#1 : !stencil.temp<ijk,f64>, %arg13 = %arg8 : f64, %arg14 = %arg9 : f64) -> !stencil.temp<ijk,f64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant -1.000000e+00 : f64
      %cst_1 = constant 3.000000e+00 : f64
      %cst_2 = constant 6.000000e+00 : f64
      %14 = divf %cst_0, %cst_2 : f64
      %cst_3 = constant -5.000000e-01 : f64
      %15 = divf %cst_0, %cst_1 : f64
      %16 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %19 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %20 = cmpf "ogt", %16, %cst : f64
      %21 = mulf %cst_3, %17 : f64
      %22 = loop.if %20 -> (f64) {
        %31 = stencil.access %arg10 [-2, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
        %32 = addf %21, %18 : f64
        %33 = mulf %15, %19 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %33, %34 : f64
        %36 = addf %32, %35 : f64
        %37 = mulf %36, %16 : f64
        loop.yield %37 : f64
      } else {
        %31 = stencil.access %arg10 [2, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
        %32 = mulf %15, %18 : f64
        %33 = addf %21, %19 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %32, %34 : f64
        %36 = addf %33, %35 : f64
        %37 = negf %16 : f64
        %38 = mulf %37, %36 : f64
        loop.yield %38 : f64
      }
      %23 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %24 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %25 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %26 = cmpf "ogt", %23, %cst : f64
      %27 = loop.if %26 -> (f64) {
        %31 = stencil.access %arg10 [0, -2, 0] : (!stencil.temp<ijk,f64>) -> f64
        %32 = addf %21, %24 : f64
        %33 = mulf %15, %25 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %33, %34 : f64
        %36 = addf %32, %35 : f64
        %37 = mulf %36, %23 : f64
        loop.yield %37 : f64
      } else {
        %31 = stencil.access %arg10 [0, 2, 0] : (!stencil.temp<ijk,f64>) -> f64
        %32 = mulf %15, %24 : f64
        %33 = addf %21, %25 : f64
        %34 = mulf %14, %31 : f64
        %35 = addf %32, %34 : f64
        %36 = addf %33, %35 : f64
        %37 = negf %23 : f64
        %38 = mulf %37, %36 : f64
        loop.yield %38 : f64
      }
      %28 = mulf %22, %arg13 : f64
      %29 = mulf %27, %arg14 : f64
      %30 = addf %28, %29 : f64
      stencil.return %30 : f64
    }
    %13 = stencil.apply (%arg10 = %12 : !stencil.temp<ijk,f64>, %arg11 = %10#0 : !stencil.temp<ijk,f64>, %arg12 = %5 : !stencil.temp<j,f64>) -> !stencil.temp<ijk,f64> {
      %14 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %15 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %17 = mulf %14, %14 : f64
      %18 = mulf %17, %15 : f64
      %19 = subf %16, %18 : f64
      stencil.return %19 : f64
    }
    stencil.store %9 to %arg2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
    stencil.store %13 to %arg3([0, 0, 0] : [64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
    return
  }
}
