
func @hdiffsa(%uin_fd : !stencil.field<ijk,f64>, %mask_fd : !stencil.field<ijk,f64>, %uout_fd : !stencil.field<ijk,f64>, %crlato_fd : !stencil.field<i,f64>, %crlatu_fd : !stencil.field<i,f64>)
  attributes { stencil.program } {
  stencil.assert %uin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %mask_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %uout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %crlato_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<i,f64>
  stencil.assert %crlatu_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<i,f64>
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %mask = stencil.load %mask_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %crlato = stencil.load %crlato_fd : (!stencil.field<i,f64>) -> !stencil.view<i,f64>
  %crlatu = stencil.load %crlatu_fd : (!stencil.field<i,f64>) -> !stencil.view<i,f64>
  // lap
  %lap = stencil.apply %arg1 = %uin, %arg2 = %crlato, %arg3 = %crlatu: !stencil.view<ijk,f64>, !stencil.view<i,f64>, !stencil.view<i,f64> {
      %0 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = addf %0, %1 : f64
      %cst = constant -2.0 : f64
      %4 = mulf %2, %cst : f64
      %5 = addf %3, %4 : f64
      %6 = stencil.access %arg1[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %7 = stencil.access %arg1[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %8 = stencil.access %arg2[0, 0, 0] : (!stencil.view<i,f64>) -> f64
      %9 = stencil.access %arg3[0, 0, 0] : (!stencil.view<i,f64>) -> f64
      %10 = subf %6, %2 : f64
      %11 = subf %7, %2 : f64
      %12 = mulf %10, %8 : f64
      %13 = mulf %11, %9 : f64
      %14 = addf %12, %5 : f64
      %15 = addf %14, %13 : f64
      stencil.return %15 : f64
	} : !stencil.view<ijk,f64>
  // flx
  %flx = stencil.apply %arg4 = %uin, %arg5 = %lap : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
      %0 = stencil.access %arg5[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg5[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      %3 = stencil.access %arg4[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = subf %3, %4 : f64
      %6 = mulf %2, %5 : f64
      %c0 = constant 0.0 : f64
      %7 = cmpf "ogt", %6, %c0 : f64
      %8 = select %7, %c0, %2 : f64
      stencil.return %8 : f64
	} : !stencil.view<ijk,f64>
  // fly
  %fly = stencil.apply %arg6 = %uin, %arg7 = %lap, %arg8 = %crlato : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<i,f64> {
      %0 = stencil.access %arg7[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg7[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      %3 = stencil.access %arg8[0, 0, 0] : (!stencil.view<i,f64>) -> f64
      %4 = mulf %2, %3 : f64
      %5 = stencil.access %arg6[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %6 = stencil.access %arg6[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %7 = subf %5, %6 : f64
      %8 = mulf %4, %7 : f64
      %c0 = constant 0.0 : f64
      %9 = cmpf "ogt", %8, %c0 : f64
      %10 = select %9, %c0, %4 : f64
      stencil.return %10 : f64
	} : !stencil.view<ijk,f64>
  // out
  %out = stencil.apply %arg9 = %uin, %arg10 = %flx, %arg11 = %fly, %arg12 = %mask : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
      %0 = stencil.access %arg10[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg10[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      %3 = stencil.access %arg11[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg11[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = subf %3, %4 : f64
      %6 = addf %2, %5 : f64
      %7 = stencil.access %arg12[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %8 = mulf %7, %6 : f64
      %9 = stencil.access %arg9[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %10 = addf %8, %9 : f64
      stencil.return %10 : f64
	} : !stencil.view<ijk,f64>
	stencil.store %out to %uout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
