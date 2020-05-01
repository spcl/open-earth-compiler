
func @fastwaves(
  %uin_fd : !stencil.field<ijk,f64>, 
  %utens_fd : !stencil.field<ijk,f64>, 
  %vin_fd : !stencil.field<ijk,f64>, 
  %vtens_fd : !stencil.field<ijk,f64>, 
  %wgtfac_fd : !stencil.field<ijk,f64>, 
  %ppuv_fd : !stencil.field<ijk,f64>, 
  %hhl_fd : !stencil.field<ijk,f64>,
  %rho_fd : !stencil.field<ijk,f64>,
  %dzdx_fd : !stencil.field<ijk,f64>,
  %dzdy_fd : !stencil.field<ijk,f64>,
  %uout_fd : !stencil.field<ijk,f64>, 
  %vout_fd : !stencil.field<ijk,f64>,
  %div_fd : !stencil.field<ijk,f64>)
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
  stencil.assert %dzdx_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %dzdy_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %uout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %div_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  // loads
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %utens = stencil.load %utens_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %vin = stencil.load %vin_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %vtens = stencil.load %vtens_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %wgtfac = stencil.load %wgtfac_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %ppuv = stencil.load %ppuv_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %hhl = stencil.load %hhl_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %rho = stencil.load %rho_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %dzdx = stencil.load %dzdx_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %dzdy = stencil.load %dzdy_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  // ppgk
  %ppgk = stencil.apply %arg1 = %wgtfac, %arg2 = %ppuv : 
    !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {  
      %0 = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = mulf %0, %1 : f64
      %cst = constant 1.0 : f64
      %3 = subf %cst, %0 : f64
      %4 = stencil.access %arg2[0, 0, -1] : (!stencil.temp<ijk,f64>) -> f64
      %5 = mulf %4, %3 : f64
      %6 = addf %2, %5 : f64
      stencil.return %6 : f64
  } : !stencil.temp<ijk,f64>
  // ppgc
  %ppgc = stencil.apply %arg3 = %ppgk : !stencil.temp<ijk,f64> {  
      %0 = stencil.access %arg3[0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg3[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      stencil.return %2 : f64
  } : !stencil.temp<ijk,f64>
  // ppgu
  %ppgu = stencil.apply %arg4 = %ppuv, %arg5 = %ppgc, %arg6 = %hhl : 
    !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {  
      // ppuv(i+1,j,k) - ppuv(i,j,k)
      %0 = stencil.access %arg4[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg4[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      // ppgc(i+1,j,k) + ppgc(i,j,k)
      %cst = constant 0.5 : f64
      %3 = stencil.access %arg5[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %4 = stencil.access %arg5[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %5 = addf %3, %4 : f64
      %6 = mulf %cst, %5 : f64
      // hhl ratio
      %7 = stencil.access %arg6[0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %8 = stencil.access %arg6[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %9 = stencil.access %arg6[1, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %10 = stencil.access %arg6[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
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
  } : !stencil.temp<ijk,f64>
  // ppgv
  %ppgv = stencil.apply %arg7 = %ppuv, %arg8 = %ppgc, %arg9 = %hhl : 
    !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {  
      // ppuv(i+1,j,k) - ppuv(i,j,k)
      %0 = stencil.access %arg7[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg7[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      // ppgc(i+1,j,k) + ppgc(i,j,k)
      %cst = constant 0.5 : f64
      %3 = stencil.access %arg8[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %4 = stencil.access %arg8[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %5 = addf %3, %4 : f64
      %6 = mulf %cst, %5 : f64
      // hhl ratio
      %7 = stencil.access %arg9[0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %8 = stencil.access %arg9[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %9 = stencil.access %arg9[0, 1, 1] : (!stencil.temp<ijk,f64>) -> f64
      %10 = stencil.access %arg9[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
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
  } : !stencil.temp<ijk,f64>
  // uout
  %uout = stencil.apply %arg10 = %uin, %arg11 = %utens, %arg12 = %ppgu, %arg13 = %rho : 
    !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {  
      %cst = constant 2.0 : f64
      %fake = constant 0.01 : f64
      // ppguv divided by rho
      %0 = stencil.access %arg12[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg13[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = stencil.access %arg13[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %3 = addf %1, %2 : f64
      %4 = divf %cst, %3 : f64
      %5 = mulf %0, %4 : f64
      // utens
      %6 = stencil.access %arg11[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %7 = subf %6, %5 : f64
      %8 = mulf %fake, %7 : f64
      // uin
      %9 = stencil.access %arg10[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %10 = addf %8, %9 : f64
      stencil.return %10 : f64
  } : !stencil.temp<ijk,f64>
  // vout
  %vout = stencil.apply %arg14 = %vin, %arg15 = %vtens, %arg16 = %ppgv, %arg17 = %rho : 
    !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {  
      %cst = constant 2.0 : f64
      %fake = constant 0.01 : f64
      // ppguv divided by rho
      %0 = stencil.access %arg16[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg17[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = stencil.access %arg17[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %3 = addf %1, %2 : f64
      %4 = divf %cst, %3 : f64
      %5 = mulf %0, %4 : f64
      // utens
      %6 = stencil.access %arg15[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %7 = subf %6, %5 : f64
      %8 = mulf %fake, %7 : f64
      // uin
      %9 = stencil.access %arg14[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %10 = addf %8, %9 : f64
      stencil.return %10 : f64
  } : !stencil.temp<ijk,f64>
  // udc
  %udc = stencil.apply %arg18 = %wgtfac, %arg19 = %uout : 
    !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {  
      %cst0 = constant 0.5 : f64
      %cst1 = constant 1.0 : f64
      %0 = stencil.access %arg18[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg18[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = addf %0, %1 : f64
      %3 = mulf %cst0, %2 : f64
      // weighted sum
      %4 = stencil.access %arg19[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %5 = stencil.access %arg19[0, 0, -1] : (!stencil.temp<ijk,f64>) -> f64   
      %6 = mulf %3, %4 : f64
      %7 = subf %cst1, %3 : f64
      %8 = mulf %7, %5 : f64
      %9 = addf %6, %8 : f64
      stencil.return %9 : f64
  } : !stencil.temp<ijk,f64>
  // vdc
  %vdc = stencil.apply %arg20 = %wgtfac, %arg21 = %vout : 
    !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {  
      %cst0 = constant 0.5 : f64
      %cst1 = constant 1.0 : f64
      %0 = stencil.access %arg20[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg20[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = addf %0, %1 : f64
      %3 = mulf %cst0, %2 : f64
      // weighted sum
      %4 = stencil.access %arg21[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %5 = stencil.access %arg21[0, 0, -1] : (!stencil.temp<ijk,f64>) -> f64   
      %6 = mulf %3, %4 : f64
      %7 = subf %cst1, %3 : f64
      %8 = mulf %7, %5 : f64
      %9 = addf %6, %8 : f64
      stencil.return %9 : f64
  } : !stencil.temp<ijk,f64>
  // div
  %div = stencil.apply %arg22 = %uout, %arg23 = %udc, %arg24 = %vout, %arg25 = %vdc, %arg26 = %dzdx, %arg27 = %dzdy : 
    !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {  
      %fake1 = constant 0.1 : f64
      %fake2 = constant 0.1 : f64
      %fake3 = constant 0.2 : f64
      %fake4 = constant 0.3 : f64
      // uout(i,j,k) term
      %0 = stencil.access %arg23[0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg23[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      %3 = stencil.access %arg26[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %4 = mulf %2, %3 : f64
      %5 = stencil.access %arg22[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %6 = addf %5, %4 : f64
      %7 = mulf %fake1, %6 : f64
      // uout(i-1,j,k) term
      %8 = stencil.access %arg23[-1, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %9 = stencil.access %arg23[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %10 = subf %8, %9 : f64
      %11 = stencil.access %arg26[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %12 = mulf %10, %11 : f64
      %13 = stencil.access %arg22[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %14 = subf %12, %13 : f64
      %15 = mulf %fake2, %14 : f64
      // vout(i,j,k) term
      %16 = stencil.access %arg25[0, 0, 1] : (!stencil.temp<ijk,f64>) -> f64
      %17 = stencil.access %arg25[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %18 = subf %16, %17 : f64
      %19 = stencil.access %arg27[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %20 = mulf %18, %19 : f64
      %21 = stencil.access %arg24[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %22 = addf %21, %20 : f64
      %23 = mulf %fake3, %22 : f64
      // vout(i,j-1,k) term
      %24 = stencil.access %arg25[0, -1, 1] : (!stencil.temp<ijk,f64>) -> f64
      %25 = stencil.access %arg25[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %26 = subf %24, %25 : f64
      %27 = stencil.access %arg27[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %28 = mulf %26, %27 : f64
      %29 = stencil.access %arg24[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %30 = subf %28, %29 : f64
      %31 = mulf %fake4, %30 : f64
      // sum everything
      %33 = addf %7, %15 : f64
      %34 = addf %33, %23 : f64
      %35 = addf %31, %34 : f64
      stencil.return %35 : f64
  } : !stencil.temp<ijk,f64>
  // store results
  stencil.store %uout to %uout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  stencil.store %vout to %vout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  stencil.store %div to %div_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  return
}