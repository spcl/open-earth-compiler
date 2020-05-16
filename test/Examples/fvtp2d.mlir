

module {
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
}
