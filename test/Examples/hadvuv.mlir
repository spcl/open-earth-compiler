

module {
  func @hadvuv(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<0x?x0xf64>, %arg5: !stencil.field<0x?x0xf64>, %arg6: !stencil.field<0x?x0xf64>, %arg7: !stencil.field<0x?x0xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.cast %arg3([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %4 = stencil.cast %arg4([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %5 = stencil.cast %arg5([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %6 = stencil.cast %arg6([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %7 = stencil.cast %arg7([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %8 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %9 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %10 = stencil.load %4 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %11 = stencil.load %5 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %12 = stencil.load %6 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %13 = stencil.load %7 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %14:2 = stencil.apply (%arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %10 : !stencil.temp<0x?x0xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 3.000000e+00 : f64
      %22 = divf %cst, %cst_0 : f64
      %23 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg8 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = addf %23, %24 : f64
      %27 = addf %26, %25 : f64
      %28 = mulf %27, %22 : f64
      %29 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %30 = mulf %28, %29 : f64
      %31 = stencil.store_result %28 : (f64) -> !stencil.result<f64>
      %32 = stencil.store_result %30 : (f64) -> !stencil.result<f64>
      stencil.return %31, %32 : !stencil.result<f64>, !stencil.result<f64>
    }
    %15:2 = stencil.apply (%arg8 = %9 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 6.371229e+06 : f64
      %22 = divf %cst, %cst_0 : f64
      %cst_1 = constant 2.500000e-01 : f64
      %23 = stencil.access %arg8 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg8 [1, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg8 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %23, %24 : f64
      %28 = addf %25, %26 : f64
      %29 = addf %27, %28 : f64
      %30 = mulf %29, %cst_1 : f64
      %31 = mulf %30, %22 : f64
      %32 = stencil.store_result %30 : (f64) -> !stencil.result<f64>
      %33 = stencil.store_result %31 : (f64) -> !stencil.result<f64>
      stencil.return %32, %33 : !stencil.result<f64>, !stencil.result<f64>
    }
    %16 = stencil.apply (%arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %14#1 : !stencil.temp<?x?x?xf64>, %arg10 = %15#1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 4.8828125E-4 : f64
      %cst_0 = constant 7.32421875E-4 : f64
      %cst_1 = constant 0.000000e+00 : f64
      %cst_2 = constant -1.000000e+00 : f64
      %cst_3 = constant 3.000000e+00 : f64
      %cst_4 = constant 6.000000e+00 : f64
      %22 = divf %cst_2, %cst_4 : f64
      %cst_5 = constant -5.000000e-01 : f64
      %23 = divf %cst_2, %cst_3 : f64
      %24 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg8 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = cmpf "ogt", %24, %cst_1 : f64
      %29 = mulf %cst_5, %25 : f64
      %30 = stencil.access %arg8 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = addf %29, %26 : f64
      %32 = mulf %23, %27 : f64
      %33 = mulf %22, %30 : f64
      %34 = addf %32, %33 : f64
      %35 = addf %31, %34 : f64
      %36 = mulf %35, %24 : f64
      %37 = stencil.access %arg8 [2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %38 = mulf %23, %26 : f64
      %39 = addf %29, %27 : f64
      %40 = mulf %22, %37 : f64
      %41 = addf %38, %40 : f64
      %42 = addf %39, %41 : f64
      %43 = negf %24 : f64
      %44 = mulf %43, %42 : f64
      %45 = select %28, %36, %44 : f64
      %46 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = stencil.access %arg8 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %48 = stencil.access %arg8 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %49 = cmpf "ogt", %46, %cst_1 : f64
      %50 = stencil.access %arg8 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %51 = addf %29, %47 : f64
      %52 = mulf %23, %48 : f64
      %53 = mulf %22, %50 : f64
      %54 = addf %52, %53 : f64
      %55 = addf %51, %54 : f64
      %56 = mulf %55, %46 : f64
      %57 = stencil.access %arg8 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %58 = mulf %23, %47 : f64
      %59 = addf %29, %48 : f64
      %60 = mulf %22, %57 : f64
      %61 = addf %58, %60 : f64
      %62 = addf %59, %61 : f64
      %63 = negf %46 : f64
      %64 = mulf %63, %62 : f64
      %65 = select %49, %56, %64 : f64
      %66 = mulf %45, %cst : f64
      %67 = mulf %65, %cst_0 : f64
      %68 = addf %66, %67 : f64
      %69 = stencil.store_result %68 : (f64) -> !stencil.result<f64>
      stencil.return %69 : !stencil.result<f64>
    }
    %17 = stencil.apply (%arg8 = %16 : !stencil.temp<?x?x?xf64>, %arg9 = %8 : !stencil.temp<?x?x?xf64>, %arg10 = %15#0 : !stencil.temp<?x?x?xf64>, %arg11 = %12 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %22 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %25 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = mulf %22, %23 : f64
      %27 = mulf %26, %24 : f64
      %28 = addf %27, %25 : f64
      %29 = stencil.store_result %28 : (f64) -> !stencil.result<f64>
      stencil.return %29 : !stencil.result<f64>
    }
    %18:2 = stencil.apply (%arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %11 : !stencil.temp<0x?x0xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 2.500000e-01 : f64
      %22 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg8 [-1, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg8 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = addf %22, %23 : f64
      %27 = addf %24, %25 : f64
      %28 = addf %26, %27 : f64
      %29 = mulf %28, %cst : f64
      %30 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %31 = mulf %29, %30 : f64
      %32 = stencil.store_result %29 : (f64) -> !stencil.result<f64>
      %33 = stencil.store_result %31 : (f64) -> !stencil.result<f64>
      stencil.return %32, %33 : !stencil.result<f64>, !stencil.result<f64>
    }
    %19:2 = stencil.apply (%arg8 = %9 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 6.371229e+06 : f64
      %22 = divf %cst, %cst_0 : f64
      %cst_1 = constant 3.000000e+00 : f64
      %23 = divf %cst, %cst_1 : f64
      %24 = stencil.access %arg8 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg8 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %24, %25 : f64
      %28 = addf %27, %26 : f64
      %29 = mulf %28, %23 : f64
      %30 = mulf %29, %22 : f64
      %31 = stencil.store_result %29 : (f64) -> !stencil.result<f64>
      %32 = stencil.store_result %30 : (f64) -> !stencil.result<f64>
      stencil.return %31, %32 : !stencil.result<f64>, !stencil.result<f64>
    }
    %20 = stencil.apply (%arg8 = %9 : !stencil.temp<?x?x?xf64>, %arg9 = %18#1 : !stencil.temp<?x?x?xf64>, %arg10 = %19#1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 4.8828125E-4 : f64
      %cst_0 = constant 7.32421875E-4 : f64
      %cst_1 = constant 0.000000e+00 : f64
      %cst_2 = constant -1.000000e+00 : f64
      %cst_3 = constant 3.000000e+00 : f64
      %cst_4 = constant 6.000000e+00 : f64
      %22 = divf %cst_2, %cst_4 : f64
      %cst_5 = constant -5.000000e-01 : f64
      %23 = divf %cst_2, %cst_3 : f64
      %24 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = stencil.access %arg8 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = cmpf "ogt", %24, %cst_1 : f64
      %29 = mulf %cst_5, %25 : f64
      %30 = stencil.access %arg8 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = addf %29, %26 : f64
      %32 = mulf %23, %27 : f64
      %33 = mulf %22, %30 : f64
      %34 = addf %32, %33 : f64
      %35 = addf %31, %34 : f64
      %36 = mulf %35, %24 : f64
      %37 = stencil.access %arg8 [2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %38 = mulf %23, %26 : f64
      %39 = addf %29, %27 : f64
      %40 = mulf %22, %37 : f64
      %41 = addf %38, %40 : f64
      %42 = addf %39, %41 : f64
      %43 = negf %24 : f64
      %44 = mulf %43, %42 : f64
      %45 = select %28, %36, %44 : f64
      %46 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = stencil.access %arg8 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %48 = stencil.access %arg8 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %49 = cmpf "ogt", %46, %cst_1 : f64
      %50 = stencil.access %arg8 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %51 = addf %29, %47 : f64
      %52 = mulf %23, %48 : f64
      %53 = mulf %22, %50 : f64
      %54 = addf %52, %53 : f64
      %55 = addf %51, %54 : f64
      %56 = mulf %55, %46 : f64
      %57 = stencil.access %arg8 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %58 = mulf %23, %47 : f64
      %59 = addf %29, %48 : f64
      %60 = mulf %22, %57 : f64
      %61 = addf %58, %60 : f64
      %62 = addf %59, %61 : f64
      %63 = negf %46 : f64
      %64 = mulf %63, %62 : f64
      %65 = select %49, %56, %64 : f64
      %66 = mulf %45, %cst : f64
      %67 = mulf %65, %cst_0 : f64
      %68 = addf %66, %67 : f64
      %69 = stencil.store_result %68 : (f64) -> !stencil.result<f64>
      stencil.return %69 : !stencil.result<f64>
    }
    %21 = stencil.apply (%arg8 = %20 : !stencil.temp<?x?x?xf64>, %arg9 = %18#0 : !stencil.temp<?x?x?xf64>, %arg10 = %13 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %22 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %24 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = mulf %22, %22 : f64
      %26 = mulf %25, %23 : f64
      %27 = subf %24, %26 : f64
      %28 = stencil.store_result %27 : (f64) -> !stencil.result<f64>
      stencil.return %28 : !stencil.result<f64>
    }
    stencil.store %17 to %2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %21 to %3([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
