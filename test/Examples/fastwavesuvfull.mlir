

module {
  func @fastwavesuvfull(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>, %arg9: !stencil.field<?x?x?xf64>, %arg10: !stencil.field<?x?x?xf64>, %arg11: !stencil.field<?x?x?xf64>, %arg12: !stencil.field<?x?x?xf64>, %arg13: !stencil.field<?x?x?xf64>, %arg14: !stencil.field<0x?x0xf64>, %arg15: !stencil.field<?x?x0xf64>, %arg16: !stencil.field<?x?x0xf64>, %arg17: !stencil.field<?x?x0xf64>, %arg18: !stencil.field<?x?x0xf64>, %arg19: !stencil.field<?x?x0xf64>, %arg20: !stencil.field<?x?x0xf64>) attributes {stencil.program} {
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
    %13 = stencil.cast %arg13([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %14 = stencil.cast %arg14([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %15 = stencil.cast %arg15([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x0xf64>) -> !stencil.field<72x72x0xf64>
    %16 = stencil.cast %arg16([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x0xf64>) -> !stencil.field<72x72x0xf64>
    %17 = stencil.cast %arg17([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x0xf64>) -> !stencil.field<72x72x0xf64>
    %18 = stencil.cast %arg18([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x0xf64>) -> !stencil.field<72x72x0xf64>
    %19 = stencil.cast %arg19([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x0xf64>) -> !stencil.field<72x72x0xf64>
    %20 = stencil.cast %arg20([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x0xf64>) -> !stencil.field<72x72x0xf64>
    %21 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %22 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %23 = stencil.load %2 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %24 = stencil.load %3 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %25 = stencil.load %4 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %26 = stencil.load %5 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %27 = stencil.load %6 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %28 = stencil.load %7 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %29 = stencil.load %8 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %30 = stencil.load %9 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %31 = stencil.load %10 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %32 = stencil.load %11 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %33 = stencil.load %14 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %34 = stencil.load %15 : (!stencil.field<72x72x0xf64>) -> !stencil.temp<?x?x0xf64>
    %35 = stencil.load %16 : (!stencil.field<72x72x0xf64>) -> !stencil.temp<?x?x0xf64>
    %36 = stencil.load %17 : (!stencil.field<72x72x0xf64>) -> !stencil.temp<?x?x0xf64>
    %37 = stencil.load %18 : (!stencil.field<72x72x0xf64>) -> !stencil.temp<?x?x0xf64>
    %38 = stencil.load %19 : (!stencil.field<72x72x0xf64>) -> !stencil.temp<?x?x0xf64>
    %39 = stencil.load %20 : (!stencil.field<72x72x0xf64>) -> !stencil.temp<?x?x0xf64>
    %40 = stencil.apply (%arg21 = %27 : !stencil.temp<?x?x?xf64>, %arg22 = %28 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %51 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %52 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %53 = mulf %51, %52 : f64
      %cst = constant 1.000000e+00 : f64
      %54 = subf %cst, %51 : f64
      %55 = stencil.access %arg22 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      %56 = mulf %55, %54 : f64
      %57 = addf %53, %56 : f64
      %58 = stencil.store_result %57 : (f64) -> !stencil.result<f64>
      stencil.return %58 : !stencil.result<f64>
    }
    %41 = stencil.apply (%arg21 = %40 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %51 = stencil.access %arg21 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %52 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %53 = subf %51, %52 : f64
      %54 = stencil.store_result %53 : (f64) -> !stencil.result<f64>
      stencil.return %54 : !stencil.result<f64>
    }
    %42 = stencil.apply (%arg21 = %28 : !stencil.temp<?x?x?xf64>, %arg22 = %41 : !stencil.temp<?x?x?xf64>, %arg23 = %29 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %51 = stencil.access %arg21 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %52 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %53 = subf %51, %52 : f64
      %cst = constant 5.000000e-01 : f64
      %54 = stencil.access %arg22 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %55 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %56 = addf %54, %55 : f64
      %57 = mulf %cst, %56 : f64
      %58 = stencil.access %arg23 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %59 = stencil.access %arg23 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %60 = stencil.access %arg23 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %61 = stencil.access %arg23 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %62 = addf %58, %59 : f64
      %63 = addf %60, %61 : f64
      %64 = subf %62, %63 : f64
      %65 = subf %58, %59 : f64
      %66 = subf %60, %61 : f64
      %67 = addf %65, %66 : f64
      %68 = divf %64, %67 : f64
      %69 = mulf %57, %68 : f64
      %70 = addf %53, %69 : f64
      %71 = stencil.store_result %70 : (f64) -> !stencil.result<f64>
      stencil.return %71 : !stencil.result<f64>
    }
    %43 = stencil.apply (%arg21 = %28 : !stencil.temp<?x?x?xf64>, %arg22 = %41 : !stencil.temp<?x?x?xf64>, %arg23 = %29 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %51 = stencil.access %arg21 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %52 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %53 = subf %51, %52 : f64
      %cst = constant 5.000000e-01 : f64
      %54 = stencil.access %arg22 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %55 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %56 = addf %54, %55 : f64
      %57 = mulf %cst, %56 : f64
      %58 = stencil.access %arg23 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %59 = stencil.access %arg23 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %60 = stencil.access %arg23 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %61 = stencil.access %arg23 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %62 = addf %58, %59 : f64
      %63 = addf %60, %61 : f64
      %64 = subf %62, %63 : f64
      %65 = subf %58, %59 : f64
      %66 = subf %60, %61 : f64
      %67 = addf %65, %66 : f64
      %68 = divf %64, %67 : f64
      %69 = mulf %57, %68 : f64
      %70 = addf %53, %69 : f64
      %71 = stencil.store_result %70 : (f64) -> !stencil.result<f64>
      stencil.return %71 : !stencil.result<f64>
    }
    %44 = stencil.apply (%arg21 = %28 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %51 = stencil.access %arg21 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %52 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %53 = subf %51, %52 : f64
      %54 = stencil.store_result %53 : (f64) -> !stencil.result<f64>
      stencil.return %54 : !stencil.result<f64>
    }
    %45 = stencil.apply (%arg21 = %28 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %51 = stencil.access %arg21 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %52 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %53 = subf %51, %52 : f64
      %54 = stencil.store_result %53 : (f64) -> !stencil.result<f64>
      stencil.return %54 : !stencil.result<f64>
    }
    %998 = stencil.apply (%arg21 = %21 : !stencil.temp<?x?x?xf64>, %arg22 = %22 : !stencil.temp<?x?x?xf64>, %arg24 = %30 : !stencil.temp<?x?x?xf64>, %arg25 = %33 : !stencil.temp<0x?x0xf64>, %arg99 = %44 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+01 : f64
      %cst_0 = constant 2.000000e+00 : f64
      %56 = stencil.access %arg99 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %57 = stencil.access %arg24 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %58 = stencil.access %arg24 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %59 = stencil.access %arg25 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %60 = addf %57, %58 : f64
      %61 = mulf %cst_0, %59 : f64
      %62 = divf %61, %60 : f64
      %63 = mulf %56, %62 : f64
      %64 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %65 = subf %64, %63 : f64
      %66 = mulf %cst, %65 : f64
      %67 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %68 = addf %66, %67 : f64
      %54 = stencil.store_result %68 : (f64) -> !stencil.result<f64>
      stencil.return %54 : !stencil.result<f64>
    }
    %999 = stencil.apply (%arg21 = %21 : !stencil.temp<?x?x?xf64>, %arg22 = %22 : !stencil.temp<?x?x?xf64>, %arg24 = %30 : !stencil.temp<?x?x?xf64>, %arg25 = %33 : !stencil.temp<0x?x0xf64>, %arg99 = %42 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+01 : f64
      %cst_0 = constant 2.000000e+00 : f64
      %56 = stencil.access %arg99 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %57 = stencil.access %arg24 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %58 = stencil.access %arg24 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %59 = stencil.access %arg25 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %60 = addf %57, %58 : f64
      %61 = mulf %cst_0, %59 : f64
      %62 = divf %61, %60 : f64
      %63 = mulf %56, %62 : f64
      %64 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %65 = subf %64, %63 : f64
      %66 = mulf %cst, %65 : f64
      %67 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %68 = addf %66, %67 : f64
      %54 = stencil.store_result %68 : (f64) -> !stencil.result<f64>
      stencil.return %54 : !stencil.result<f64>
    }
    %46 = stencil.apply (%arg21 = %30 : !stencil.temp<?x?x?xf64>, %arg22 = %28 : !stencil.temp<?x?x?xf64>, %arg23 = %22 : !stencil.temp<?x?x?xf64>, %arg24 = %33 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %51 = stencil.access %arg21 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %52 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %53 = addf %51, %52 : f64
      %54 = mulf %53, %cst : f64
      %55 = stencil.access %arg24 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %56 = negf %55 : f64
      %57 = divf %56, %54 : f64
      %58 = stencil.access %arg22 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %59 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %60 = subf %58, %59 : f64
      %61 = mulf %57, %60 : f64
      %62 = stencil.access %arg23 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %63 = addf %61, %62 : f64
      %64 = stencil.store_result %63 : (f64) -> !stencil.result<f64>
      stencil.return %64 : !stencil.result<f64>
    }
    %47 = stencil.apply (%arg21 = %30 : !stencil.temp<?x?x?xf64>, %arg22 = %28 : !stencil.temp<?x?x?xf64>, %arg23 = %24 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 4.8828125E-4 : f64
      %cst_0 = constant 5.000000e-01 : f64
      %51 = stencil.access %arg21 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %52 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %53 = addf %51, %52 : f64
      %54 = mulf %53, %cst_0 : f64
      %55 = negf %cst : f64
      %56 = divf %55, %54 : f64
      %57 = stencil.access %arg22 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %58 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %59 = subf %57, %58 : f64
      %60 = mulf %56, %59 : f64
      %61 = stencil.access %arg23 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %62 = addf %60, %61 : f64
      %63 = stencil.store_result %62 : (f64) -> !stencil.result<f64>
      stencil.return %63 : !stencil.result<f64>
    }
    %48 = stencil.apply (%arg21 = %30 : !stencil.temp<?x?x?xf64>, %arg22 = %31 : !stencil.temp<?x?x?xf64>, %arg23 = %32 : !stencil.temp<?x?x?xf64>, %arg24 = %28 : !stencil.temp<?x?x?xf64>, %arg25 = %38 : !stencil.temp<?x?x0xf64>, %arg26 = %39 : !stencil.temp<?x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %51 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %52 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %53 = divf %51, %52 : f64
      %cst = constant 9.8066499999999994 : f64
      %54 = mulf %53, %cst : f64
      %55 = stencil.access %arg23 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %56 = stencil.access %arg24 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %57 = addf %55, %56 : f64
      %58 = stencil.access %arg25 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %59 = mulf %57, %58 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %60 = subf %cst_0, %59 : f64
      %61 = mulf %54, %60 : f64
      %62 = stencil.access %arg26 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %63 = addf %61, %62 : f64
      %64 = stencil.store_result %63 : (f64) -> !stencil.result<f64>
      stencil.return %64 : !stencil.result<f64>
    }
    %49 = stencil.apply (%arg27 = %34 : !stencil.temp<?x?x0xf64>, %arg28 = %36 : !stencil.temp<?x?x0xf64>, %arg29 = %37 : !stencil.temp<?x?x0xf64>, %arg30 = %46 : !stencil.temp<?x?x?xf64>, %arg31 = %47 : !stencil.temp<?x?x?xf64>, %arg32 = %48 : !stencil.temp<?x?x?xf64>, %arg33 = %25 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+01 : f64
      %cst_1 = constant 5.000000e-01 : f64
      %55 = stencil.access %arg29 [1, -1, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %56 = stencil.access %arg29 [1, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %57 = addf %55, %56 : f64
      %58 = mulf %cst_1, %57 : f64
      %59 = stencil.access %arg29 [0, -1, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %60 = stencil.access %arg29 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %61 = addf %59, %60 : f64
      %62 = mulf %cst_1, %61 : f64
      %63 = addf %58, %62 : f64
      %64 = mulf %cst_1, %63 : f64
      %65 = stencil.access %arg31 [1, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %66 = stencil.access %arg31 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %67 = addf %65, %66 : f64
      %68 = mulf %cst_1, %67 : f64
      %69 = stencil.access %arg31 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %70 = stencil.access %arg31 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %71 = addf %69, %70 : f64
      %72 = mulf %cst_1, %71 : f64
      %73 = addf %68, %72 : f64
      %74 = mulf %cst_1, %73 : f64
      %75 = mulf %64, %74 : f64
      %76 = stencil.access %arg28 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %77 = stencil.access %arg30 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %78 = mulf %76, %77 : f64
      %79 = stencil.access %arg32 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %80 = stencil.access %arg32 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %81 = addf %79, %80 : f64
      %82 = mulf %cst_1, %81 : f64
      %83 = subf %82, %78 : f64
      %84 = subf %83, %75 : f64
      %85 = stencil.access %arg27 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %86 = mulf %85, %76 : f64
      %87 = mulf %84, %86 : f64
      %88 = addf %87, %77 : f64
      %89 = mulf %cst, %88 : f64
      %90 = stencil.access %arg33 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %91 = addf %90, %89 : f64
      %54 = stencil.store_result %91 : (f64) -> !stencil.result<f64>
      stencil.return %54 : !stencil.result<f64>
    }
    %50 = stencil.apply (%arg26 = %35 : !stencil.temp<?x?x0xf64>, %arg27 = %36 : !stencil.temp<?x?x0xf64>, %arg28 = %37 : !stencil.temp<?x?x0xf64>, %arg29 = %46 : !stencil.temp<?x?x?xf64>, %arg30 = %47 : !stencil.temp<?x?x?xf64>, %arg31 = %48 : !stencil.temp<?x?x?xf64>, %arg32 = %26 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+01 : f64
      %cst_2 = constant 5.000000e-01 : f64
      %55 = stencil.access %arg27 [-1, 1, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %56 = stencil.access %arg27 [0, 1, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %57 = addf %55, %56 : f64
      %58 = mulf %cst_2, %57 : f64
      %59 = stencil.access %arg27 [-1, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %60 = stencil.access %arg27 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %61 = addf %59, %60 : f64
      %62 = mulf %cst_2, %61 : f64
      %63 = addf %58, %62 : f64
      %64 = mulf %cst_2, %63 : f64
      %65 = stencil.access %arg29 [-1, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %66 = stencil.access %arg29 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %67 = addf %65, %66 : f64
      %68 = mulf %cst_2, %67 : f64
      %69 = stencil.access %arg29 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %70 = stencil.access %arg29 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %71 = addf %69, %70 : f64
      %72 = mulf %cst_2, %71 : f64
      %73 = addf %68, %72 : f64
      %74 = mulf %cst_2, %73 : f64
      %75 = mulf %64, %74 : f64
      %76 = stencil.access %arg28 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %77 = stencil.access %arg30 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %78 = mulf %76, %77 : f64
      %79 = stencil.access %arg31 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %80 = stencil.access %arg31 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %81 = addf %79, %80 : f64
      %82 = mulf %cst_2, %81 : f64
      %83 = subf %82, %78 : f64
      %84 = subf %83, %75 : f64
      %85 = stencil.access %arg26 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %86 = mulf %85, %76 : f64
      %87 = mulf %84, %86 : f64
      %88 = addf %87, %77 : f64
      %89 = mulf %cst, %88 : f64
      %90 = stencil.access %arg32 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %91 = addf %90, %89 : f64
      %54 = stencil.store_result %91 : (f64) -> !stencil.result<f64>
      stencil.return %54 : !stencil.result<f64>
    }
    %1000 = stencil.apply (%arg21 = %23 : !stencil.temp<?x?x?xf64>, %arg22 = %24 : !stencil.temp<?x?x?xf64>, %arg24 = %30 : !stencil.temp<?x?x?xf64>, %arg100 = %45 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+01 : f64
      %cst_0 = constant 4.8828125E-4 : f64
      %cst_1 = constant 2.000000e+00 : f64
      %56 = stencil.access %arg100 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %57 = stencil.access %arg24 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %58 = stencil.access %arg24 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %59 = addf %57, %58 : f64
      %60 = mulf %cst_1, %cst_0 : f64
      %61 = divf %60, %59 : f64
      %62 = mulf %56, %61 : f64
      %63 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %64 = subf %63, %62 : f64
      %65 = mulf %cst, %64 : f64
      %66 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %67 = addf %65, %66 : f64
      %54 = stencil.store_result %67 : (f64) -> !stencil.result<f64>
      stencil.return %54 : !stencil.result<f64>
    }
    %1001 = stencil.apply (%arg21 = %23 : !stencil.temp<?x?x?xf64>, %arg22 = %24 : !stencil.temp<?x?x?xf64>, %arg24 = %30 : !stencil.temp<?x?x?xf64>, %arg100 = %43 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+01 : f64
      %cst_0 = constant 4.8828125E-4 : f64
      %cst_1 = constant 2.000000e+00 : f64
      %56 = stencil.access %arg100 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %57 = stencil.access %arg24 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %58 = stencil.access %arg24 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %59 = addf %57, %58 : f64
      %60 = mulf %cst_1, %cst_0 : f64
      %61 = divf %60, %59 : f64
      %62 = mulf %56, %61 : f64
      %63 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %64 = subf %63, %62 : f64
      %65 = mulf %cst, %64 : f64
      %66 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %67 = addf %65, %66 : f64
      %54 = stencil.store_result %67 : (f64) -> !stencil.result<f64>
      stencil.return %54 : !stencil.result<f64>
    }
    %99:2 = stencil.combine 2 at 11 lower = (%998, %1000 : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) upper = (%999, %1001 : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
    %1002:2 = stencil.combine 2 at 59 lower = (%99#0, %99#1 : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) upper = (%49, %50 : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
    stencil.store %1002#0 to %12([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %1002#1 to %13([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
