

module {
  func @p_grad_c(%arg0: !stencil.field<ijk,f64>, %arg1: !stencil.field<ijk,f64>, %arg2: !stencil.field<ijk,f64>, %arg3: !stencil.field<ijk,f64>, %arg4: !stencil.field<ijk,f64>, %arg5: !stencil.field<ijk,f64>, %arg6: !stencil.field<ijk,f64>, %arg7: !stencil.field<ijk,f64>, %arg8: !stencil.field<ijk,f64>, %arg9: f64) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg6([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg7([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    stencil.assert %arg8([-4, -4, -4] : [68, 68, 68]) : !stencil.field<ijk,f64>
    %0 = stencil.load %arg0 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %1 = stencil.load %arg1 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %2 = stencil.load %arg2 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %3 = stencil.load %arg3 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %4 = stencil.load %arg4 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %5 = stencil.load %arg5 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %6 = stencil.load %arg6 : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
    %7 = stencil.apply (%arg10 = %4 : !stencil.temp<ijk,f64>) -> !stencil.temp<ijk,f64> {
      %10 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      stencil.return %10 : f64
    }
    %8 = stencil.apply (%arg10 = %0 : !stencil.temp<ijk,f64>, %arg11 = %2 : !stencil.temp<ijk,f64>, %arg12 = %7 : !stencil.temp<ijk,f64>, %arg13 = %5 : !stencil.temp<ijk,f64>, %arg14 = %6 : !stencil.temp<ijk,f64>, %arg15 = %arg9 : f64) -> !stencil.temp<ijk,f64> {
      %10 = stencil.access %arg13 [-1, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %11 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %12 = subf %10, %11 : f64
      %13 = stencil.access %arg14 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %14 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = mulf %12, %15 : f64
      %17 = stencil.access %arg13 [-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %19 = subf %17, %18 : f64
      %20 = stencil.access %arg14 [-1, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %21 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = mulf %19, %22 : f64
      %24 = addf %16, %23 : f64
      %25 = stencil.access %arg12 [-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %26 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %27 = addf %25, %26 : f64
      %28 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %29 = mulf %28, %arg15 : f64
      %30 = divf %29, %27 : f64
      %31 = mulf %30, %24 : f64
      %32 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %33 = addf %31, %32 : f64
      stencil.return %33 : f64
    }
    %9 = stencil.apply (%arg10 = %1 : !stencil.temp<ijk,f64>, %arg11 = %3 : !stencil.temp<ijk,f64>, %arg12 = %7 : !stencil.temp<ijk,f64>, %arg13 = %5 : !stencil.temp<ijk,f64>, %arg14 = %6 : !stencil.temp<ijk,f64>, %arg15 = %arg9 : f64) -> !stencil.temp<ijk,f64> {
      %10 = stencil.access %arg13 [0, -1, 1] : (!stencil.temp<ijk,f64>) -> f64
      %11 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %12 = subf %10, %11 : f64
      %13 = stencil.access %arg14 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %14 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %15 = subf %13, %14 : f64
      %16 = mulf %12, %15 : f64
      %17 = stencil.access %arg13 [0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %19 = subf %17, %18 : f64
      %20 = stencil.access %arg14 [0, -1, 1] : (!stencil.temp<ijk,f64>) -> f64
      %21 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %22 = subf %20, %21 : f64
      %23 = mulf %19, %22 : f64
      %24 = addf %16, %23 : f64
      %25 = stencil.access %arg12 [0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %26 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %27 = addf %25, %26 : f64
      %28 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %29 = mulf %28, %arg15 : f64
      %30 = divf %29, %27 : f64
      %31 = mulf %30, %24 : f64
      %32 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %33 = addf %31, %32 : f64
      stencil.return %33 : f64
    }
    stencil.store %8 to %arg7([0, 0, 0] : [64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
    stencil.store %9 to %arg8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
    return
  }
}
