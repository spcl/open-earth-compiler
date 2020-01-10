
func @hdiff(%uin_fd : !stencil.field<ijk,f64>, %mask_fd : !stencil.field<ijk,f64>, %uout_fd : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	stencil.assert %uin_fd ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %mask_fd ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %uout_fd ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %mask = stencil.load %mask_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  // lap
  %lap = stencil.apply %arg1 = %uin : !stencil.view<ijk,f64> {  
      %0 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg1[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = stencil.access %arg1[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = addf %0, %1 : f64
      %6 = addf %2, %3 : f64
      %7 = addf %5, %6 : f64
      %cst = constant -4.000000e+00 : f64
      %8 = mulf %4, %cst : f64
      %9 = addf %8, %7 : f64
      stencil.return %9 : f64
	} : !stencil.view<ijk,f64>
  // flx
  %flx = stencil.apply %arg2 = %uin, %arg3 = %lap : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
      %0 = stencil.access %arg3[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      %3 = stencil.access %arg2[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = subf %3, %4 : f64
      %6 = mulf %2, %5 : f64
      %c0 = constant 0.0 : f64
      %7 = cmpf "ogt", %6, %c0 : f64
      %8 = select %7, %2, %c0 : f64
      stencil.return %8 : f64
	} : !stencil.view<ijk,f64>
  // fly
  %fly = stencil.apply %arg4 = %uin, %arg5 = %lap : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
      %0 = stencil.access %arg4[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      %3 = stencil.access %arg5[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg5[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = subf %3, %4 : f64
      %6 = mulf %2, %5 : f64
      %c0 = constant 0.0 : f64
      %7 = cmpf "ogt", %6, %c0 : f64
      %8 = select %7, %2, %c0 : f64
      stencil.return %8 : f64
	} : !stencil.view<ijk,f64>
  // out
  %out = stencil.apply %arg6 = %uin, %arg7 = %flx, %arg8 = %fly, %arg9 = %mask : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
      %0 = stencil.access %arg7[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg7[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = subf %0, %1 : f64
      %3 = stencil.access %arg8[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg8[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = subf %3, %4 : f64
      %6 = addf %2, %5 : f64
      %7 = stencil.access %arg9[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %8 = mulf %7, %6 : f64
      %9 = stencil.access %arg6[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %10 = addf %8, %9 : f64
      stencil.return %10 : f64
	} : !stencil.view<ijk,f64>
	stencil.store %out to %uout_fd ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
