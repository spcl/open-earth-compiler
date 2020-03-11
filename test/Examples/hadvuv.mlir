
func @hadvuv(
  %uin_fd : !stencil.field<ijk,f64>,
  %vin_fd : !stencil.field<ijk,f64>,
  %uout_fd : !stencil.field<ijk,f64>,
  %vout_fd : !stencil.field<ijk,f64>,
  %acrlat0_fd : !stencil.field<j,f64>,
  %acrlat1_fd : !stencil.field<j,f64>,
  %tgrlatda0_fd : !stencil.field<j,f64>,
  %tgrlatda1_fd : !stencil.field<j,f64>,
  %eddlat : f64,
  %eddlon : f64,
  %earth_radi_recip : f64)
  attributes { stencil.program } {
  // asserts
  stencil.assert %uin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %uout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %acrlat0_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  stencil.assert %acrlat1_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  stencil.assert %tgrlatda0_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  stencil.assert %tgrlatda1_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  // loads
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %vin = stencil.load %vin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %acrlat0 = stencil.load %acrlat0_fd : (!stencil.field<j,f64>) -> !stencil.view<j,f64>
  %acrlat1 = stencil.load %acrlat1_fd : (!stencil.field<j,f64>) -> !stencil.view<j,f64>
  %tgrlatda0 = stencil.load %tgrlatda0_fd : (!stencil.field<j,f64>) -> !stencil.view<j,f64>
  %tgrlatda1 = stencil.load %tgrlatda1_fd : (!stencil.field<j,f64>) -> !stencil.view<j,f64>
  // uatupos
  %uatupos = stencil.apply %arg1 = %uin : !stencil.view<ijk,f64> {
      %cst = constant 3.333333333333333148296e-01 : f64
      %0 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = addf %0, %1 : f64
      %4 = addf %3, %2 : f64
      %5 = mulf %4, %cst : f64
      stencil.return %5 : f64
  } : !stencil.view<ijk,f64>
  // vatupos
  %vatupos = stencil.apply %arg2 = %vin : !stencil.view<ijk,f64> {
      %cst = constant 0.25 : f64
      %0 = stencil.access %arg2[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg2[1, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = stencil.access %arg2[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = addf %0, %1 : f64
      %5 = addf %2, %3 : f64
      %6 = addf %4, %5 : f64
      %7 = mulf %6, %cst : f64
      stencil.return %7 : f64
  } : !stencil.view<ijk,f64>
  // uavgu
  %uavgu = stencil.apply %arg3 = %uatupos, %arg4 = %acrlat0 : !stencil.view<ijk,f64>, !stencil.view<j,f64> {
      %0 = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg4[0, 0, 0] : (!stencil.view<j,f64>) -> f64
      %2 = mulf %0, %1 : f64
      stencil.return %2 : f64
  } : !stencil.view<ijk,f64>
  // vavgu
  %vavgu = stencil.apply %arg5 = %vatupos, %arg6 = %earth_radi_recip : !stencil.view<ijk,f64>, f64 {
      %0 = stencil.access %arg5[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = mulf %0, %arg6 : f64
      stencil.return %1 : f64
  } : !stencil.view<ijk,f64>
  // udelta
  %udelta = stencil.apply %arg7 = %uin, %arg8 = %uavgu, %arg9 = %vavgu, %arg10 = %eddlat, %arg11 = %eddlon :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64, f64 {
      %zero = constant 0.000000e+00 : f64
      %minus_one = constant -1.000000e+00 : f64
      %cst_0 = constant -1.666666666666666574148e-01 : f64
      %cst_2 = constant -0.5 : f64
      %cst_3 = constant -3.333333333333333148296e-01 : f64
      %0 = stencil.access %arg8[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg7[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg7[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = stencil.access %arg7[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = cmpf "ogt", %0, %zero : f64
      %5 = mulf %cst_2, %2 : f64
      %6 = loop.if %4 -> (f64) {
          %7 = stencil.access %arg7[-2, 0, 0] : (!stencil.view<ijk,f64>) -> f64
          %8 = mulf %cst_3, %3 : f64
          %9 = mulf %cst_0, %7 : f64
          %10 = addf %5, %1 : f64
          %11 = addf %8, %9 : f64
          %12 = addf %10, %11 : f64
          %13 = mulf %12, %0 : f64
          loop.yield %13 : f64
      } else {
          %14 = stencil.access %arg7[2, 0, 0] : (!stencil.view<ijk,f64>) -> f64
          %15 = mulf %cst_3, %1 : f64
          %16 = mulf %cst_0, %14 : f64
          %17 = addf %5, %3 : f64
          %18 = addf %15, %16 : f64
          %19 = addf %17, %18 : f64
          %20 = mulf %19, %0 : f64
          %21 = mulf %20, %minus_one : f64
          loop.yield %21 : f64
      }
      %22 = stencil.access %arg9[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %23 = stencil.access %arg7[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %24 = stencil.access %arg7[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %25 = cmpf "ogt", %22, %zero : f64
      %26 = loop.if %25 -> (f64) {
          %27 = stencil.access %arg7[0, -2, 0] : (!stencil.view<ijk,f64>) -> f64
          %28 = mulf %cst_3, %24 : f64
          %29 = mulf %cst_0, %27 : f64
          %30 = addf %5, %23 : f64
          %31 = addf %28, %29 : f64
          %32 = addf %30, %31 : f64
          %33 = mulf %32, %22 : f64
          loop.yield %33 : f64
      } else {
          %34 = stencil.access %arg7[0, 2, 0] : (!stencil.view<ijk,f64>) -> f64
          %35 = mulf %cst_3, %23 : f64
          %36 = mulf %cst_0, %34 : f64
          %37 = addf %5, %24 : f64
          %38 = addf %35, %36 : f64
          %39 = addf %37, %38 : f64
          %40 = mulf %39, %22 : f64
          %41 = mulf %40, %minus_one : f64
          loop.yield %41 : f64
      }
      %42 = mulf %6, %arg10 : f64
      %43 = mulf %26, %arg11 : f64
      %44 = addf %42, %43 : f64
      stencil.return %44 : f64
  } : !stencil.view<ijk,f64>
  // uout
  %uout = stencil.apply %arg12 = %udelta, %arg13 = %uin, %arg14 = %vatupos, %arg15 = %tgrlatda0 :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<j,f64> {
      %0 = stencil.access %arg13[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg14[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg15[0, 0, 0] : (!stencil.view<j,f64>) -> f64
      %3 = stencil.access %arg12[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = mulf %0, %1 : f64
      %5 = mulf %4, %2 : f64
      %6 = addf %5, %3 : f64
      stencil.return %6 : f64
  } : !stencil.view<ijk,f64>
  // uatvpos
  %uatvpos = stencil.apply %arg16 = %uin : !stencil.view<ijk,f64> {
      %cst = constant 0.25 : f64
      %0 = stencil.access %arg16[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg16[-1, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg16[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = stencil.access %arg16[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = addf %0, %1 : f64
      %5 = addf %2, %3 : f64
      %6 = addf %4, %5 : f64
      %7 = mulf %6, %cst : f64
      stencil.return %7 : f64
  } : !stencil.view<ijk,f64>
  // vatvpos
  %vatvpos = stencil.apply %arg17 = %vin : !stencil.view<ijk,f64> {
      %cst = constant 3.333333333333333148296e-01 : f64
      %0 = stencil.access %arg17[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg17[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg17[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = addf %0, %1 : f64
      %4 = addf %3, %2 : f64
      %5 = mulf %4, %cst : f64
      stencil.return %5 : f64
  } : !stencil.view<ijk,f64>
  // uavgv
  %uavgv = stencil.apply %arg18 = %uatvpos, %arg19 = %acrlat1 : !stencil.view<ijk,f64>, !stencil.view<j,f64> {
      %0 = stencil.access %arg18[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg19[0, 0, 0] : (!stencil.view<j,f64>) -> f64
      %2 = mulf %0, %1 : f64
      stencil.return %2 : f64
  } : !stencil.view<ijk,f64>
  // vavgv
  %vavgv = stencil.apply %arg20 = %vatvpos, %arg21 = %earth_radi_recip : !stencil.view<ijk,f64>, f64 {
      %0 = stencil.access %arg20[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = mulf %0, %arg21 : f64
      stencil.return %1 : f64
  } : !stencil.view<ijk,f64>
  // vdelta
  %vdelta = stencil.apply %arg22 = %vin, %arg23 = %uavgv, %arg24 = %vavgv, %arg25 = %eddlat, %arg26 = %eddlon :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64, f64 {
      %zero = constant 0.000000e+00 : f64
      %minus_one = constant -1.000000e+00 : f64
      %cst_0 = constant -1.666666666666666574148e-01 : f64
      %cst_2 = constant -0.5 : f64
      %cst_3 = constant -3.333333333333333148296e-01 : f64
      %0 = stencil.access %arg23[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg22[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg22[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = stencil.access %arg22[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = cmpf "ogt", %0, %zero : f64
      %5 = mulf %cst_2, %2 : f64
      %6 = loop.if %4 -> (f64) {
          %7 = stencil.access %arg22[-2, 0, 0] : (!stencil.view<ijk,f64>) -> f64
          %8 = mulf %cst_3, %3 : f64
          %9 = mulf %cst_0, %7 : f64
          %10 = addf %5, %1 : f64
          %11 = addf %8, %9 : f64
          %12 = addf %10, %11 : f64
          %13 = mulf %12, %0 : f64
          loop.yield %13 : f64
      } else {
          %14 = stencil.access %arg22[2, 0, 0] : (!stencil.view<ijk,f64>) -> f64
          %15 = mulf %cst_3, %1 : f64
          %16 = mulf %cst_0, %14 : f64
          %17 = addf %5, %3 : f64
          %18 = addf %15, %16 : f64
          %19 = addf %17, %18 : f64
          %20 = mulf %19, %0 : f64
          %21 = mulf %20, %minus_one : f64
          loop.yield %21 : f64
      }
      %22 = stencil.access %arg24[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %23 = stencil.access %arg22[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %24 = stencil.access %arg22[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %25 = cmpf "ogt", %22, %zero : f64
      %26 = loop.if %25 -> (f64) {
          %27 = stencil.access %arg22[0, -2, 0] : (!stencil.view<ijk,f64>) -> f64
          %28 = mulf %cst_3, %24 : f64
          %29 = mulf %cst_0, %27 : f64
          %30 = addf %5, %23 : f64
          %31 = addf %28, %29 : f64
          %32 = addf %30, %31 : f64
          %33 = mulf %32, %22 : f64
          loop.yield %33 : f64
      } else {
          %34 = stencil.access %arg22[0, 2, 0] : (!stencil.view<ijk,f64>) -> f64
          %35 = mulf %cst_3, %23 : f64
          %36 = mulf %cst_0, %34 : f64
          %37 = addf %5, %24 : f64
          %38 = addf %35, %36 : f64
          %39 = addf %37, %38 : f64
          %40 = mulf %39, %22 : f64
          %41 = mulf %40, %minus_one : f64
          loop.yield %41 : f64
      }
      %42 = mulf %6, %arg25 : f64
      %43 = mulf %26, %arg26 : f64
      %44 = addf %42, %43 : f64
      stencil.return %44 : f64
  } : !stencil.view<ijk,f64>
  // vout
  %vout = stencil.apply %arg27 = %vdelta, %arg28 = %uatvpos, %arg29 = %tgrlatda1 :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<j,f64> {
      %0 = stencil.access %arg28[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg29[0, 0, 0] : (!stencil.view<j,f64>) -> f64
      %2 = stencil.access %arg27[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = mulf %0, %0 : f64
      %4 = mulf %3, %1 : f64
      %5 = subf %2, %4 : f64
      stencil.return %5 : f64
  } : !stencil.view<ijk,f64>
  // store results
  stencil.store %uout to %uout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  stencil.store %vout to %vout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
