

module {
  func @fvtp2d(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>, %arg7: !stencil.field<?x?x?xf64>, %arg8: !stencil.field<?x?x?xf64>, %arg9: !stencil.field<?x?x?xf64>, %arg10: !stencil.field<?x?x?xf64>, %arg11: !stencil.field<?x?x?xf64>, %arg12: !stencil.field<?x?x?xf64>, %arg13: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
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
    %14 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %15 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %16 = stencil.load %2 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %17 = stencil.load %3 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %18 = stencil.load %4 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %19 = stencil.load %5 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %20 = stencil.load %6 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %21 = stencil.load %7 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %22 = stencil.apply (%arg14 = %14 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %40 = divf %cst_0, %cst_1 : f64
      %41 = divf %cst, %cst_1 : f64
      %42 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %43 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = addf %42, %43 : f64
      %45 = stencil.access %arg14 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %46 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = addf %45, %46 : f64
      %48 = mulf %40, %44 : f64
      %49 = mulf %41, %47 : f64
      %50 = addf %48, %49 : f64
      stencil.return %50 : f64
    }
    %23:4 = stencil.apply (%arg14 = %14 : !stencil.temp<?x?x?xf64>, %arg15 = %22 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %40 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = subf %40, %41 : f64
      %43 = stencil.access %arg15 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = subf %43, %41 : f64
      %45 = addf %42, %44 : f64
      %46 = mulf %42, %44 : f64
      %47 = cmpf "olt", %46, %cst : f64
      %48 = select %47, %cst_0, %cst : f64
      stencil.return %42, %44, %45, %48 : f64, f64, f64, f64
    }
    %24 = stencil.apply (%arg14 = %14 : !stencil.temp<?x?x?xf64>, %arg15 = %16 : !stencil.temp<?x?x?xf64>, %arg16 = %23#0 : !stencil.temp<?x?x?xf64>, %arg17 = %23#1 : !stencil.temp<?x?x?xf64>, %arg18 = %23#2 : !stencil.temp<?x?x?xf64>, %arg19 = %23#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %40 = stencil.access %arg19 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = cmpf "oeq", %40, %cst : f64
      %42 = select %41, %cst_0, %cst : f64
      %43 = stencil.access %arg19 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = mulf %43, %42 : f64
      %45 = addf %40, %44 : f64
      %46 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = cmpf "ogt", %46, %cst : f64
      %48 = scf.if %47 -> (f64) {
        %51 = stencil.access %arg17 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = stencil.access %arg18 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %53 = mulf %46, %52 : f64
        %54 = subf %51, %53 : f64
        %55 = subf %cst_0, %46 : f64
        %56 = mulf %55, %54 : f64
        scf.yield %56 : f64
      } else {
        %51 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %53 = mulf %46, %52 : f64
        %54 = addf %51, %53 : f64
        %55 = addf %cst_0, %46 : f64
        %56 = mulf %55, %54 : f64
        scf.yield %56 : f64
      }
      %49 = mulf %48, %45 : f64
      %50 = scf.if %47 -> (f64) {
        %51 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = addf %51, %49 : f64
        scf.yield %52 : f64
      } else {
        %51 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = addf %51, %49 : f64
        scf.yield %52 : f64
      }
      stencil.return %50 : f64
    }
    %25 = stencil.apply (%arg14 = %20 : !stencil.temp<?x?x?xf64>, %arg15 = %24 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %40 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = mulf %40, %41 : f64
      stencil.return %42 : f64
    }
    %26 = stencil.apply (%arg14 = %14 : !stencil.temp<?x?x?xf64>, %arg15 = %21 : !stencil.temp<?x?x?xf64>, %arg16 = %25 : !stencil.temp<?x?x?xf64>, %arg17 = %18 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %40 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = mulf %40, %41 : f64
      %43 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = stencil.access %arg16 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %45 = subf %43, %44 : f64
      %46 = addf %42, %45 : f64
      %47 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %48 = divf %46, %47 : f64
      stencil.return %48 : f64
    }
    %27 = stencil.apply (%arg14 = %26 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %40 = divf %cst_0, %cst_1 : f64
      %41 = divf %cst, %cst_1 : f64
      %42 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %43 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = addf %42, %43 : f64
      %45 = stencil.access %arg14 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %46 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = addf %45, %46 : f64
      %48 = mulf %40, %44 : f64
      %49 = mulf %41, %47 : f64
      %50 = addf %48, %49 : f64
      stencil.return %50 : f64
    }
    %28:4 = stencil.apply (%arg14 = %26 : !stencil.temp<?x?x?xf64>, %arg15 = %27 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %40 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = subf %40, %41 : f64
      %43 = stencil.access %arg15 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = subf %43, %41 : f64
      %45 = addf %42, %44 : f64
      %46 = mulf %42, %44 : f64
      %47 = cmpf "olt", %46, %cst : f64
      %48 = select %47, %cst_0, %cst : f64
      stencil.return %42, %44, %45, %48 : f64, f64, f64, f64
    }
    %29 = stencil.apply (%arg14 = %26 : !stencil.temp<?x?x?xf64>, %arg15 = %15 : !stencil.temp<?x?x?xf64>, %arg16 = %28#0 : !stencil.temp<?x?x?xf64>, %arg17 = %28#1 : !stencil.temp<?x?x?xf64>, %arg18 = %28#2 : !stencil.temp<?x?x?xf64>, %arg19 = %28#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %40 = stencil.access %arg19 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = cmpf "oeq", %40, %cst : f64
      %42 = select %41, %cst_0, %cst : f64
      %43 = stencil.access %arg19 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = mulf %43, %42 : f64
      %45 = addf %40, %44 : f64
      %46 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = cmpf "ogt", %46, %cst : f64
      %48 = scf.if %47 -> (f64) {
        %51 = stencil.access %arg17 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = stencil.access %arg18 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %53 = mulf %46, %52 : f64
        %54 = subf %51, %53 : f64
        %55 = subf %cst_0, %46 : f64
        %56 = mulf %55, %54 : f64
        scf.yield %56 : f64
      } else {
        %51 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %53 = mulf %46, %52 : f64
        %54 = addf %51, %53 : f64
        %55 = addf %cst_0, %46 : f64
        %56 = mulf %55, %54 : f64
        scf.yield %56 : f64
      }
      %49 = mulf %48, %45 : f64
      %50 = scf.if %47 -> (f64) {
        %51 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = addf %51, %49 : f64
        scf.yield %52 : f64
      } else {
        %51 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = addf %51, %49 : f64
        scf.yield %52 : f64
      }
      stencil.return %50 : f64
    }
    %30 = stencil.apply (%arg14 = %14 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %40 = divf %cst_0, %cst_1 : f64
      %41 = divf %cst, %cst_1 : f64
      %42 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %43 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = addf %42, %43 : f64
      %45 = stencil.access %arg14 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %46 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = addf %45, %46 : f64
      %48 = mulf %40, %44 : f64
      %49 = mulf %41, %47 : f64
      %50 = addf %48, %49 : f64
      stencil.return %50 : f64
    }
    %31:4 = stencil.apply (%arg14 = %14 : !stencil.temp<?x?x?xf64>, %arg15 = %30 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %40 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = subf %40, %41 : f64
      %43 = stencil.access %arg15 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = subf %43, %41 : f64
      %45 = addf %42, %44 : f64
      %46 = mulf %42, %44 : f64
      %47 = cmpf "olt", %46, %cst : f64
      %48 = select %47, %cst_0, %cst : f64
      stencil.return %42, %44, %45, %48 : f64, f64, f64, f64
    }
    %32 = stencil.apply (%arg14 = %14 : !stencil.temp<?x?x?xf64>, %arg15 = %15 : !stencil.temp<?x?x?xf64>, %arg16 = %31#0 : !stencil.temp<?x?x?xf64>, %arg17 = %31#1 : !stencil.temp<?x?x?xf64>, %arg18 = %31#2 : !stencil.temp<?x?x?xf64>, %arg19 = %31#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %40 = stencil.access %arg19 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = cmpf "oeq", %40, %cst : f64
      %42 = select %41, %cst_0, %cst : f64
      %43 = stencil.access %arg19 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = mulf %43, %42 : f64
      %45 = addf %40, %44 : f64
      %46 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = cmpf "ogt", %46, %cst : f64
      %48 = scf.if %47 -> (f64) {
        %51 = stencil.access %arg17 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = stencil.access %arg18 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %53 = mulf %46, %52 : f64
        %54 = subf %51, %53 : f64
        %55 = subf %cst_0, %46 : f64
        %56 = mulf %55, %54 : f64
        scf.yield %56 : f64
      } else {
        %51 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %53 = mulf %46, %52 : f64
        %54 = addf %51, %53 : f64
        %55 = addf %cst_0, %46 : f64
        %56 = mulf %55, %54 : f64
        scf.yield %56 : f64
      }
      %49 = mulf %48, %45 : f64
      %50 = scf.if %47 -> (f64) {
        %51 = stencil.access %arg14 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = addf %51, %49 : f64
        scf.yield %52 : f64
      } else {
        %51 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = addf %51, %49 : f64
        scf.yield %52 : f64
      }
      stencil.return %50 : f64
    }
    %33 = stencil.apply (%arg14 = %19 : !stencil.temp<?x?x?xf64>, %arg15 = %32 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %40 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = mulf %40, %41 : f64
      stencil.return %42 : f64
    }
    %34 = stencil.apply (%arg14 = %14 : !stencil.temp<?x?x?xf64>, %arg15 = %21 : !stencil.temp<?x?x?xf64>, %arg16 = %33 : !stencil.temp<?x?x?xf64>, %arg17 = %17 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %40 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = mulf %40, %41 : f64
      %43 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = stencil.access %arg16 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %45 = subf %43, %44 : f64
      %46 = addf %42, %45 : f64
      %47 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %48 = divf %46, %47 : f64
      stencil.return %48 : f64
    }
    %35 = stencil.apply (%arg14 = %34 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 7.000000e+00 : f64
      %cst_1 = constant 1.200000e+01 : f64
      %40 = divf %cst_0, %cst_1 : f64
      %41 = divf %cst, %cst_1 : f64
      %42 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %43 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = addf %42, %43 : f64
      %45 = stencil.access %arg14 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %46 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = addf %45, %46 : f64
      %48 = mulf %40, %44 : f64
      %49 = mulf %41, %47 : f64
      %50 = addf %48, %49 : f64
      stencil.return %50 : f64
    }
    %36:4 = stencil.apply (%arg14 = %34 : !stencil.temp<?x?x?xf64>, %arg15 = %35 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %40 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = subf %40, %41 : f64
      %43 = stencil.access %arg15 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = subf %43, %41 : f64
      %45 = addf %42, %44 : f64
      %46 = mulf %42, %44 : f64
      %47 = cmpf "olt", %46, %cst : f64
      %48 = select %47, %cst_0, %cst : f64
      stencil.return %42, %44, %45, %48 : f64, f64, f64, f64
    }
    %37 = stencil.apply (%arg14 = %34 : !stencil.temp<?x?x?xf64>, %arg15 = %16 : !stencil.temp<?x?x?xf64>, %arg16 = %36#0 : !stencil.temp<?x?x?xf64>, %arg17 = %36#1 : !stencil.temp<?x?x?xf64>, %arg18 = %36#2 : !stencil.temp<?x?x?xf64>, %arg19 = %36#3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %40 = stencil.access %arg19 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = cmpf "oeq", %40, %cst : f64
      %42 = select %41, %cst_0, %cst : f64
      %43 = stencil.access %arg19 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %44 = mulf %43, %42 : f64
      %45 = addf %40, %44 : f64
      %46 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = cmpf "ogt", %46, %cst : f64
      %48 = scf.if %47 -> (f64) {
        %51 = stencil.access %arg17 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = stencil.access %arg18 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %53 = mulf %46, %52 : f64
        %54 = subf %51, %53 : f64
        %55 = subf %cst_0, %46 : f64
        %56 = mulf %55, %54 : f64
        scf.yield %56 : f64
      } else {
        %51 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %53 = mulf %46, %52 : f64
        %54 = addf %51, %53 : f64
        %55 = addf %cst_0, %46 : f64
        %56 = mulf %55, %54 : f64
        scf.yield %56 : f64
      }
      %49 = mulf %48, %45 : f64
      %50 = scf.if %47 -> (f64) {
        %51 = stencil.access %arg14 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = addf %51, %49 : f64
        scf.yield %52 : f64
      } else {
        %51 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %52 = addf %51, %49 : f64
        scf.yield %52 : f64
      }
      stencil.return %50 : f64
    }
    %38 = stencil.apply (%arg14 = %29 : !stencil.temp<?x?x?xf64>, %arg15 = %32 : !stencil.temp<?x?x?xf64>, %arg16 = %19 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %40 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = addf %40, %41 : f64
      %43 = mulf %42, %cst : f64
      %44 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %45 = mulf %44, %43 : f64
      stencil.return %45 : f64
    }
    %39 = stencil.apply (%arg14 = %37 : !stencil.temp<?x?x?xf64>, %arg15 = %24 : !stencil.temp<?x?x?xf64>, %arg16 = %20 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 5.000000e-01 : f64
      %40 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %41 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %42 = addf %40, %41 : f64
      %43 = mulf %42, %cst : f64
      %44 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %45 = mulf %44, %43 : f64
      stencil.return %45 : f64
    }
    stencil.store %26 to %8([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %34 to %9([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %38 to %10([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %32 to %11([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %39 to %12([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %24 to %13([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
