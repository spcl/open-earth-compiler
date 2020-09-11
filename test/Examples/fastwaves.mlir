

module {
  func @fastwaves(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>, %arg9: !stencil.field<?x?x?xf64>, %arg10: !stencil.field<?x?x?xf64>, %arg11: !stencil.field<?x?x?xf64>, %arg12: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.cast %arg3([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %4 = stencil.cast %arg4([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %5 = stencil.cast %arg5([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %6 = stencil.cast %arg6([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %7 = stencil.cast %arg7([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %8 = stencil.cast %arg8([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %9 = stencil.cast %arg9([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %10 = stencil.cast %arg10([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %11 = stencil.cast %arg11([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %12 = stencil.cast %arg12([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %13 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %14 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %15 = stencil.load %2 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %16 = stencil.load %3 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %17 = stencil.load %4 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %18 = stencil.load %5 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %19 = stencil.load %6 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %20 = stencil.load %7 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %21 = stencil.load %8 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %22 = stencil.load %9 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %23 = stencil.apply (%arg13 = %17 : !stencil.temp<?x?x?xf64>, %arg14 = %18 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %32 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = mulf %32, %33 : f64
      %cst = constant 1.000000e+00 : f64
      %35 = subf %cst, %32 : f64
      %36 = stencil.access %arg14 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      %37 = mulf %36, %35 : f64
      %38 = addf %34, %37 : f64
      %39 = stencil.store_result %38 : (f64) -> !stencil.result<f64>
      stencil.return %39 : !stencil.result<f64>
    }
    %24 = stencil.apply (%arg13 = %23 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %32 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = subf %32, %33 : f64
      %35 = stencil.store_result %34 : (f64) -> !stencil.result<f64>
      stencil.return %35 : !stencil.result<f64>
    }
    %25 = stencil.apply (%arg13 = %18 : !stencil.temp<?x?x?xf64>, %arg14 = %24 : !stencil.temp<?x?x?xf64>, %arg15 = %19 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %32 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = subf %32, %33 : f64
      %cst = constant 5.000000e-01 : f64
      %35 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %36 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %37 = addf %35, %36 : f64
      %38 = mulf %cst, %37 : f64
      %39 = stencil.access %arg15 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %40 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg15 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = stencil.access %arg15 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %43 = addf %39, %40 : f64
      %44 = addf %41, %42 : f64
      %45 = subf %43, %44 : f64
      %46 = subf %39, %40 : f64
      %47 = subf %41, %42 : f64
      %48 = addf %46, %47 : f64
      %49 = divf %45, %48 : f64
      %50 = mulf %38, %49 : f64
      %51 = addf %34, %50 : f64
      %52 = stencil.store_result %51 : (f64) -> !stencil.result<f64>
      stencil.return %52 : !stencil.result<f64>
    }
    %26 = stencil.apply (%arg13 = %18 : !stencil.temp<?x?x?xf64>, %arg14 = %24 : !stencil.temp<?x?x?xf64>, %arg15 = %19 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %32 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = subf %32, %33 : f64
      %cst = constant 5.000000e-01 : f64
      %35 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %36 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %37 = addf %35, %36 : f64
      %38 = mulf %cst, %37 : f64
      %39 = stencil.access %arg15 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %40 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg15 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = stencil.access %arg15 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %43 = addf %39, %40 : f64
      %44 = addf %41, %42 : f64
      %45 = subf %43, %44 : f64
      %46 = subf %39, %40 : f64
      %47 = subf %41, %42 : f64
      %48 = addf %46, %47 : f64
      %49 = divf %45, %48 : f64
      %50 = mulf %38, %49 : f64
      %51 = addf %34, %50 : f64
      %52 = stencil.store_result %51 : (f64) -> !stencil.result<f64>
      stencil.return %52 : !stencil.result<f64>
    }
    %27 = stencil.apply (%arg13 = %13 : !stencil.temp<?x?x?xf64>, %arg14 = %14 : !stencil.temp<?x?x?xf64>, %arg15 = %25 : !stencil.temp<?x?x?xf64>, %arg16 = %20 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 2.000000e+00 : f64
      %cst_0 = constant 1.000000e-02 : f64
      %32 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg16 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = addf %33, %34 : f64
      %36 = divf %cst, %35 : f64
      %37 = mulf %32, %36 : f64
      %38 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %39 = subf %38, %37 : f64
      %40 = mulf %cst_0, %39 : f64
      %41 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = addf %40, %41 : f64
      %43 = stencil.store_result %42 : (f64) -> !stencil.result<f64>
      stencil.return %43 : !stencil.result<f64>
    }
    %28 = stencil.apply (%arg13 = %15 : !stencil.temp<?x?x?xf64>, %arg14 = %16 : !stencil.temp<?x?x?xf64>, %arg15 = %26 : !stencil.temp<?x?x?xf64>, %arg16 = %20 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 2.000000e+00 : f64
      %cst_0 = constant 1.000000e-02 : f64
      %32 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg16 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %35 = addf %33, %34 : f64
      %36 = divf %cst, %35 : f64
      %37 = mulf %32, %36 : f64
      %38 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %39 = subf %38, %37 : f64
      %40 = mulf %cst_0, %39 : f64
      %41 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = addf %40, %41 : f64
      %43 = stencil.store_result %42 : (f64) -> !stencil.result<f64>
      stencil.return %43 : !stencil.result<f64>
    }
    %29 = stencil.apply (%arg13 = %17 : !stencil.temp<?x?x?xf64>, %arg14 = %27 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %32 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = addf %32, %33 : f64
      %35 = mulf %cst, %34 : f64
      %36 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %37 = stencil.access %arg14 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      %38 = mulf %35, %36 : f64
      %39 = subf %cst_0, %35 : f64
      %40 = mulf %39, %37 : f64
      %41 = addf %38, %40 : f64
      %42 = stencil.store_result %41 : (f64) -> !stencil.result<f64>
      stencil.return %42 : !stencil.result<f64>
    }
    %30 = stencil.apply (%arg13 = %17 : !stencil.temp<?x?x?xf64>, %arg14 = %28 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %32 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = addf %32, %33 : f64
      %35 = mulf %cst, %34 : f64
      %36 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %37 = stencil.access %arg14 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      %38 = mulf %35, %36 : f64
      %39 = subf %cst_0, %35 : f64
      %40 = mulf %39, %37 : f64
      %41 = addf %38, %40 : f64
      %42 = stencil.store_result %41 : (f64) -> !stencil.result<f64>
      stencil.return %42 : !stencil.result<f64>
    }
    %31 = stencil.apply (%arg13 = %27 : !stencil.temp<?x?x?xf64>, %arg14 = %29 : !stencil.temp<?x?x?xf64>, %arg15 = %28 : !stencil.temp<?x?x?xf64>, %arg16 = %30 : !stencil.temp<?x?x?xf64>, %arg17 = %21 : !stencil.temp<?x?x?xf64>, %arg18 = %22 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e-01 : f64
      %cst_0 = constant 1.000000e-01 : f64
      %cst_1 = constant 2.000000e-01 : f64
      %cst_2 = constant 3.000000e-01 : f64
      %32 = stencil.access %arg14 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %34 = subf %32, %33 : f64
      %35 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %36 = mulf %34, %35 : f64
      %37 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %38 = addf %37, %36 : f64
      %39 = mulf %cst, %38 : f64
      %40 = stencil.access %arg14 [-1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = subf %40, %41 : f64
      %43 = stencil.access %arg17 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = mulf %42, %43 : f64
      %45 = stencil.access %arg13 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %46 = subf %44, %45 : f64
      %47 = mulf %cst_0, %46 : f64
      %48 = stencil.access %arg16 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %49 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %50 = subf %48, %49 : f64
      %51 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %52 = mulf %50, %51 : f64
      %53 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %54 = addf %53, %52 : f64
      %55 = mulf %cst_1, %54 : f64
      %56 = stencil.access %arg16 [0, -1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %57 = stencil.access %arg16 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %58 = subf %56, %57 : f64
      %59 = stencil.access %arg18 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %60 = mulf %58, %59 : f64
      %61 = stencil.access %arg15 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %62 = subf %60, %61 : f64
      %63 = mulf %cst_2, %62 : f64
      %64 = addf %39, %47 : f64
      %65 = addf %64, %55 : f64
      %66 = addf %63, %65 : f64
      %67 = stencil.store_result %66 : (f64) -> !stencil.result<f64>
      stencil.return %67 : !stencil.result<f64>
    }
    stencil.store %27 to %10([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %28 to %11([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %31 to %12([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
