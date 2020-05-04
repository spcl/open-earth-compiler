

module {
  func @nh_p_grad(%arg0: !stencil.field<ijk,f64>, %arg1: !stencil.field<ijk,f64>, %arg2: !stencil.field<ijk,f64>, %arg3: !stencil.field<ijk,f64>, %arg4: !stencil.field<ijk,f64>, %arg5: !stencil.field<ijk,f64>, %arg6: !stencil.field<ijk,f64>, %arg7: !stencil.field<ijk,f64>, %arg8: !stencil.field<ijk,f64>, %arg9: !stencil.field<ijk,f64>, %arg10: f64) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg6([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg7([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg8([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg9([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    %0 = stencil.load %arg0 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %1 = stencil.load %arg1 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %2 = stencil.load %arg2 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %3 = stencil.load %arg3 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %4 = stencil.load %arg4 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %5 = stencil.load %arg5 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %6 = stencil.load %arg6 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %7 = stencil.load %arg7 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %8 = stencil.apply (%arg11 = %6 : !stencil.temp<ijk,f64>) -> !stencil.temp<ijk,f64> {
      %13 = stencil.access %arg11 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %14 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %15 = subf %13, %14 : f64
      stencil.return %15 : f64
    }
    %9 = stencil.apply (%arg11 = %8 : !stencil.temp<ijk,f64>, %arg12 = %4 : !stencil.temp<ijk,f64>, %arg13 = %6 : !stencil.temp<ijk,f64>, %arg14 = %arg10 : f64) -> !stencil.temp<ijk,f64> {
      %13 = stencil.access %arg12 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %14 = stencil.access %arg12 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg13 [1, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = mulf %15, %18 : f64
      %20 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %21 = stencil.access %arg12 [1, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %24 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = mulf %22, %25 : f64
      %27 = addf %19, %26 : f64
      %28 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %29 = stencil.access %arg11 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = divf %arg14, %30 : f64
      %32 = mulf %27, %31 : f64
      stencil.return %32 : f64
    }
    %10 = stencil.apply (%arg11 = %8 : !stencil.temp<ijk,f64>, %arg12 = %4 : !stencil.temp<ijk,f64>, %arg13 = %6 : !stencil.temp<ijk,f64>, %arg14 = %arg10 : f64) -> !stencil.temp<ijk,f64> {
      %13 = stencil.access %arg12 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %14 = stencil.access %arg12 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg13 [0, 1, 1] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = mulf %15, %18 : f64
      %20 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %21 = stencil.access %arg12 [0, 1, 1] : (!stencil.temp<ijk,f64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %24 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = mulf %22, %25 : f64
      %27 = addf %19, %26 : f64
      %28 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %29 = stencil.access %arg11 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = divf %arg14, %30 : f64
      %32 = mulf %27, %31 : f64
      stencil.return %32 : f64
    }
    %11 = stencil.apply (%arg11 = %0 : !stencil.temp<ijk,f64>, %arg12 = %7 : !stencil.temp<ijk,f64>, %arg13 = %4 : !stencil.temp<ijk,f64>, %arg14 = %5 : !stencil.temp<ijk,f64>, %arg15 = %9 : !stencil.temp<ijk,f64>, %arg16 = %2 : !stencil.temp<ijk,f64>, %arg17 = %arg10 : f64) -> !stencil.temp<ijk,f64> {
      %13 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %14 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg14 [1, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = mulf %15, %18 : f64
      %20 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %21 = stencil.access %arg13 [1, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = stencil.access %arg14 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %24 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = mulf %22, %25 : f64
      %27 = addf %19, %26 : f64
      %28 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %29 = stencil.access %arg12 [1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = divf %arg17, %30 : f64
      %32 = mulf %27, %31 : f64
      %33 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %34 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %35 = addf %33, %34 : f64
      %36 = addf %32, %35 : f64
      %37 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %38 = mulf %36, %37 : f64
      stencil.return %38 : f64
    }
    %12 = stencil.apply (%arg11 = %1 : !stencil.temp<ijk,f64>, %arg12 = %7 : !stencil.temp<ijk,f64>, %arg13 = %4 : !stencil.temp<ijk,f64>, %arg14 = %5 : !stencil.temp<ijk,f64>, %arg15 = %10 : !stencil.temp<ijk,f64>, %arg16 = %3 : !stencil.temp<ijk,f64>, %arg17 = %arg10 : f64) -> !stencil.temp<ijk,f64> {
      %13 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %14 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = stencil.access %arg14 [0, 1, 1] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = mulf %15, %18 : f64
      %20 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %21 = stencil.access %arg13 [0, 1, 1] : (!stencil.temp<ijk,f64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = stencil.access %arg14 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %24 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %25 = subf %23, %24 : f64
      %26 = mulf %22, %25 : f64
      %27 = addf %19, %26 : f64
      %28 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %29 = stencil.access %arg12 [0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %30 = addf %28, %29 : f64
      %31 = divf %arg17, %30 : f64
      %32 = mulf %27, %31 : f64
      %33 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %34 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %35 = addf %33, %34 : f64
      %36 = addf %32, %35 : f64
      %37 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %38 = mulf %36, %37 : f64
      stencil.return %38 : f64
    }
    stencil.store %11 to %arg8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
    stencil.store %12 to %arg9([0, 0, 0] : [64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
    return
  }
}
