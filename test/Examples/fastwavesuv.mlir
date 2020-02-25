
func @fastwavesuv(
  %uin_fd : !stencil.field<ijk,f64>,
  %utens_fd : !stencil.field<ijk,f64>,
  %vin_fd : !stencil.field<ijk,f64>,
  %vtens_fd : !stencil.field<ijk,f64>,
  %wgtfac_fd : !stencil.field<ijk,f64>,
  %ppuv_fd : !stencil.field<ijk,f64>,
  %hhl_fd : !stencil.field<ijk,f64>,
  %rho_fd : !stencil.field<ijk,f64>,
  %uout_fd : !stencil.field<ijk,f64>,
  %vout_fd : !stencil.field<ijk,f64>,
  %fx_fd : !stencil.field<j,f64>,
  %dt : f64,
  %edadlat : f64)
  attributes { stencil.program } {
  // asserts
  stencil.assert %uin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %utens_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vtens_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %wgtfac_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %ppuv_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %hhl_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %rho_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %uout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %fx_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  // loads
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %utens = stencil.load %utens_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %vin = stencil.load %vin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %vtens = stencil.load %vtens_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %wgtfac = stencil.load %wgtfac_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %ppuv = stencil.load %ppuv_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %hhl = stencil.load %hhl_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %rho = stencil.load %rho_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %fx = stencil.load %fx_fd : (!stencil.field<j,f64>) -> !stencil.view<j,f64>
  // ppgk
  %ppgk = stencil.apply %arg1 = %wgtfac, %arg2 = %ppuv :
    !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
      %0 = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = mulf %0, %1 : f64
      %cst = constant 1.0 : f64
      %3 = subf %cst, %0 : f64
      %4 = stencil.access %arg2[0, 0, -1] : (!stencil.view<ijk,f64>) -> f64
      %5 = mulf %4, %3 : f64
      %6 = addf %2, %5 : f64
      stencil.return %6 : f64
  } : !stencil.view<ijk,f64>
  // ppgc
  %ppgc = stencil.apply %arg3 = %ppgk : !stencil.view<ijk,f64> {
      %0 = stencil.access %arg3[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      stencil.return %2 : f64
  } : !stencil.view<ijk,f64>
  // ppgu
  %ppgu = stencil.apply %arg4 = %ppuv, %arg5 = %ppgc, %arg6 = %hhl :
    !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
      // ppuv(i+1,j,k) - ppuv(i,j,k)
      %0 = stencil.access %arg4[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      // ppgc(i+1,j,k) + ppgc(i,j,k)
      %cst = constant 0.5 : f64
      %3 = stencil.access %arg5[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg5[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = addf %3, %4 : f64
      %6 = mulf %cst, %5 : f64
      // hhl ratio
      %7 = stencil.access %arg6[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %8 = stencil.access %arg6[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %9 = stencil.access %arg6[1, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %10 = stencil.access %arg6[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %11 = addf %7, %8 : f64
      %12 = addf %9, %10 : f64
      %13 = subf %11, %12 : f64
      %14 = subf %7, %8 : f64
      %15 = subf %9, %10 : f64
      %16 = addf %14, %15 : f64
      %17 = divf %13, %16 : f64
      %18 = mulf %6, %17 : f64
      %19 = addf %2, %18 : f64
      stencil.return %19 : f64
  } : !stencil.view<ijk,f64>
  // ppgv
  %ppgv = stencil.apply %arg7 = %ppuv, %arg8 = %ppgc, %arg9 = %hhl :
    !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
      // ppuv(i+1,j,k) - ppuv(i,j,k)
      %0 = stencil.access %arg7[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg7[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      // ppgc(i+1,j,k) + ppgc(i,j,k)
      %cst = constant 0.5 : f64
      %3 = stencil.access %arg8[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg8[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = addf %3, %4 : f64
      %6 = mulf %cst, %5 : f64
      // hhl ratio
      %7 = stencil.access %arg9[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %8 = stencil.access %arg9[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %9 = stencil.access %arg9[0, 1, 1] : (!stencil.view<ijk,f64>) -> f64
      %10 = stencil.access %arg9[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %11 = addf %7, %8 : f64
      %12 = addf %9, %10 : f64
      %13 = subf %11, %12 : f64
      %14 = subf %7, %8 : f64
      %15 = subf %9, %10 : f64
      %16 = addf %14, %15 : f64
      %17 = divf %13, %16 : f64
      %18 = mulf %6, %17 : f64
      %19 = addf %2, %18 : f64
      stencil.return %19 : f64
  } : !stencil.view<ijk,f64>
  // uout
  %uout = stencil.apply %arg10 = %uin, %arg11 = %utens, %arg12 = %ppgu, %arg13 = %rho, %arg14 = %fx, %arg15 = %dt :
    !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<j,f64>, f64 {
      %cst = constant 2.0 : f64
      // ppguv divided by rho
      %0 = stencil.access %arg12[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg13[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg13[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = stencil.access %arg14[0, 0, 0] : (!stencil.view<j,f64>) -> f64
      %4 = addf %1, %2 : f64
      %5 = mulf %cst, %3 : f64
      %6 = divf %5, %4 : f64
      %7 = mulf %0, %6 : f64
      // utens
      %8 = stencil.access %arg11[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %9 = subf %8, %7 : f64
      %10 = mulf %arg15, %9 : f64
      // uin
      %11 = stencil.access %arg10[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %12 = addf %10, %11 : f64
      stencil.return %12 : f64
  } : !stencil.view<ijk,f64>
  // vout
  %vout = stencil.apply %arg16 = %vin, %arg17 = %vtens, %arg18 = %ppgv, %arg19 = %rho, %arg20 = %dt, %arg21 = %edadlat :
    !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64, f64 {
      %cst = constant 2.0 : f64
      // ppguv divided by rho
      %0 = stencil.access %arg18[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg19[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg19[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = addf %1, %2 : f64
      %4 = mulf %cst, %arg21 : f64
      %5 = divf %4, %3 : f64
      %6 = mulf %0, %5 : f64
      // utens
      %7 = stencil.access %arg17[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %8 = subf %7, %6 : f64
      %9 = mulf %arg20, %8 : f64
      // uin
      %10 = stencil.access %arg16[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %11 = addf %9, %10 : f64
      stencil.return %11 : f64
  } : !stencil.view<ijk,f64>
  // store results
  stencil.store %uout to %uout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  stencil.store %vout to %vout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
