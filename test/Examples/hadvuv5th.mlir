

module {
  func @hadvuv5th(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<0x?x0xf64>, %arg5: !stencil.field<0x?x0xf64>, %arg6: !stencil.field<0x?x0xf64>, %arg7: !stencil.field<0x?x0xf64>, %arg8: f64, %arg9: f64) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg3([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg4([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg5([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg6([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %arg7([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.load %arg4 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %3 = stencil.load %arg5 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %4 = stencil.load %arg6 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %5 = stencil.load %arg7 : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %6:2 = stencil.apply (%arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %2 : !stencil.temp<0x?x0xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 3.000000e+00 : f64
      %14 = divf %cst, %cst_0 : f64
      %15 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %15, %16 : f64
      %19 = addf %18, %17 : f64
      %20 = mulf %19, %14 : f64
      %21 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %22 = mulf %20, %21 : f64
      stencil.return %20, %22 : f64, f64
    }
    %7:2 = stencil.apply (%arg10 = %1 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 6371.229e3 : f64
      %14 = divf %cst, %cst_0 : f64
      %cst_1 = constant 2.500000e-01 : f64
      %15 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [1, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = addf %15, %16 : f64
      %20 = addf %17, %18 : f64
      %21 = addf %19, %20 : f64
      %22 = mulf %21, %cst_1 : f64
      %23 = mulf %22, %14 : f64
      stencil.return %22, %23 : f64, f64
    }
    %8 = stencil.apply (%arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %6#1 : !stencil.temp<?x?x?xf64>, %arg12 = %7#1 : !stencil.temp<?x?x?xf64>, %arg13 = %arg8 : f64, %arg14 = %arg9 : f64) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %cst_1 = constant -1.000000e+00 : f64
      %cst_2 = constant 2.000000e+00 : f64
      %cst_3 = constant 3.000000e+00 : f64
      %cst_4 = constant 4.000000e+00 : f64
      %cst_5 = constant 2.000000e+01 : f64
      %cst_6 = constant 3.000000e+01 : f64
      %14 = divf %cst_0, %cst_6 : f64
      %15 = divf %cst_1, %cst_4 : f64
      %16 = divf %cst_1, %cst_3 : f64
      %17 = divf %cst_1, %cst_2 : f64
      %18 = divf %cst_0, %cst_5 : f64
      %19 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg10 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg10 [2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = cmpf "ogt", %19, %cst : f64
      %26 = mulf %16, %20 : f64

      %138 = stencil.access %arg10 [-3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %139 = mulf %15, %21 : f64
      %140 = addf %26, %22 : f64
      %141 = mulf %17, %23 : f64
      %142 = mulf %18, %24 : f64
      %143 = mulf %14, %138 : f64
      %144 = addf %139, %141 : f64
      %145 = addf %142, %143 : f64
      %146 = addf %140, %144 : f64
      %147 = addf %145, %146 : f64
      %148 = mulf %147, %19 : f64

      %238 = stencil.access %arg10 [3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %239 = mulf %18, %21 : f64
      %240 = mulf %17, %22 : f64
      %241 = addf %26, %23 : f64
      %242 = mulf %15, %24 : f64
      %243 = mulf %14, %238 : f64
      %244 = addf %239, %240 : f64
      %245 = addf %242, %243 : f64
      %246 = addf %241, %244 : f64
      %247 = addf %245, %246 : f64
      %248 = negf %19 : f64
      %249 = mulf %248, %247 : f64

      %27 = select %25, %148, %249 : f64

      %28 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg10 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg10 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = cmpf "ogt", %28, %cst : f64

      %338 = stencil.access %arg10 [0, -3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %339 = mulf %15, %29 : f64
      %340 = addf %26, %30 : f64
      %341 = mulf %17, %31 : f64
      %342 = mulf %18, %32 : f64
      %343 = mulf %14, %338 : f64
      %344 = addf %339, %341 : f64
      %345 = addf %342, %343 : f64
      %346 = addf %340, %344 : f64
      %347 = addf %345, %346 : f64
      %348 = mulf %347, %28 : f64

      %438 = stencil.access %arg10 [0, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %439 = mulf %18, %29 : f64
      %440 = mulf %17, %30 : f64
      %441 = addf %26, %31 : f64
      %442 = mulf %15, %32 : f64
      %443 = mulf %14, %438 : f64
      %444 = addf %439, %440 : f64
      %445 = addf %442, %443 : f64
      %446 = addf %441, %444 : f64
      %447 = addf %445, %446 : f64
      %448 = negf %28 : f64
      %449 = mulf %448, %447 : f64

      %34 = select %33, %348, %449 : f64

      %35 = mulf %27, %arg13 : f64
      %36 = mulf %34, %arg14 : f64
      %37 = addf %35, %36 : f64
      stencil.return %37 : f64
    }
    %9 = stencil.apply (%arg10 = %8 : !stencil.temp<?x?x?xf64>, %arg11 = %0 : !stencil.temp<?x?x?xf64>, %arg12 = %7#0 : !stencil.temp<?x?x?xf64>, %arg13 = %4 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = mulf %14, %15 : f64
      %19 = mulf %18, %16 : f64
      %20 = addf %19, %17 : f64
      stencil.return %20 : f64
    }
    %10:2 = stencil.apply (%arg10 = %0 : !stencil.temp<?x?x?xf64>, %arg11 = %3 : !stencil.temp<0x?x0xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 2.500000e-01 : f64
      %14 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg10 [-1, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %14, %15 : f64
      %19 = addf %16, %17 : f64
      %20 = addf %18, %19 : f64
      %21 = mulf %20, %cst : f64
      %22 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %23 = mulf %21, %22 : f64
      stencil.return %21, %23 : f64, f64
    }
    %11:2 = stencil.apply (%arg10 = %1 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = constant 1.000000e+00 : f64
      %cst_0 = constant 6371.229e3 : f64
      %14 = divf %cst, %cst_0 : f64
      %cst_1 = constant 3.000000e+00 : f64
      %15 = divf %cst, %cst_1 : f64
      %16 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = addf %16, %17 : f64
      %20 = addf %19, %18 : f64
      %21 = mulf %20, %15 : f64
      %22 = mulf %21, %14 : f64
      stencil.return %21, %22 : f64, f64
    }
    %12 = stencil.apply (%arg10 = %1 : !stencil.temp<?x?x?xf64>, %arg11 = %10#1 : !stencil.temp<?x?x?xf64>, %arg12 = %11#1 : !stencil.temp<?x?x?xf64>, %arg13 = %arg8 : f64, %arg14 = %arg9 : f64) -> !stencil.temp<?x?x?xf64> {
      %cst = constant 0.000000e+00 : f64
      %cst_0 = constant 1.000000e+00 : f64
      %cst_1 = constant -1.000000e+00 : f64
      %cst_2 = constant 2.000000e+00 : f64
      %cst_3 = constant 3.000000e+00 : f64
      %cst_4 = constant 4.000000e+00 : f64
      %cst_5 = constant 2.000000e+01 : f64
      %cst_6 = constant 3.000000e+01 : f64
      %14 = divf %cst_0, %cst_6 : f64
      %15 = divf %cst_1, %cst_4 : f64
      %16 = divf %cst_1, %cst_3 : f64
      %17 = divf %cst_1, %cst_2 : f64
      %18 = divf %cst_0, %cst_5 : f64
      %19 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg10 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = stencil.access %arg10 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg10 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg10 [2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = cmpf "ogt", %19, %cst : f64
      %26 = mulf %16, %20 : f64

      %138 = stencil.access %arg10 [-3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %139 = mulf %15, %21 : f64
      %140 = addf %26, %22 : f64
      %141 = mulf %17, %23 : f64
      %142 = mulf %18, %24 : f64
      %143 = mulf %14, %138 : f64
      %144 = addf %139, %141 : f64
      %145 = addf %142, %143 : f64
      %146 = addf %140, %144 : f64
      %147 = addf %145, %146 : f64
      %148 = mulf %147, %19 : f64

      %238 = stencil.access %arg10 [3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %239 = mulf %18, %21 : f64
      %240 = mulf %17, %22 : f64
      %241 = addf %26, %23 : f64
      %242 = mulf %15, %24 : f64
      %243 = mulf %14, %238 : f64
      %244 = addf %239, %240 : f64
      %245 = addf %242, %243 : f64
      %246 = addf %241, %244 : f64
      %247 = addf %245, %246 : f64
      %248 = negf %19 : f64
      %249 = mulf %248, %247 : f64

      %27 = select %25, %148, %249 : f64

      %28 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg10 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg10 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg10 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg10 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = cmpf "ogt", %28, %cst : f64

      %338 = stencil.access %arg10 [0, -3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %339 = addf %26, %30 : f64
      %340 = mulf %15, %29 : f64
      %341 = mulf %17, %31 : f64
      %342 = mulf %18, %32 : f64
      %343 = mulf %14, %338 : f64
      %344 = addf %340, %341 : f64
      %345 = addf %342, %343 : f64
      %346 = addf %339, %344 : f64
      %347 = addf %345, %346 : f64
      %348 = mulf %347, %28 : f64

      %438 = stencil.access %arg10 [0, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %439 = addf %26, %31 : f64
      %440 = mulf %18, %29 : f64
      %441 = mulf %17, %30 : f64
      %442 = mulf %15, %32 : f64
      %443 = mulf %14, %438 : f64
      %444 = addf %440, %441 : f64
      %445 = addf %442, %443 : f64
      %446 = addf %439, %444 : f64
      %447 = addf %445, %446 : f64
      %448 = negf %28 : f64
      %449 = mulf %448, %447 : f64

      %34 = select %33, %348, %449 : f64

      %35 = mulf %27, %arg13 : f64
      %36 = mulf %34, %arg14 : f64
      %37 = addf %35, %36 : f64
      stencil.return %37 : f64
    }
    %13 = stencil.apply (%arg10 = %12 : !stencil.temp<?x?x?xf64>, %arg11 = %10#0 : !stencil.temp<?x?x?xf64>, %arg12 = %5 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %14 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = stencil.access %arg12 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %16 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %14, %14 : f64
      %18 = mulf %17, %15 : f64
      %19 = subf %16, %18 : f64
      stencil.return %19 : f64
    }
    stencil.store %9 to %arg2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %13 to %arg3([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
