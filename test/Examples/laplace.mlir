
func @laplace(%uin_fd : !stencil.field<ijk,f64>, %lap_fd : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	stencil.assert %uin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %lap_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  // lap
  %lap = stencil.apply %arg1 = %uin : !stencil.view<ijk,f64> {  
      %0 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %2 = stencil.access %arg1[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = stencil.access %arg1[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = addf %3, %4 : f64
      %6 = addf %5, %2 : f64 
      %7 = addf %6, %1 : f64 
      %cst = constant -4.0 : f64
      %8 = mulf %4, %cst : f64
      %9 = addf %8, %7 : f64
      stencil.return %9 : f64
 	} : !stencil.view<ijk,f64>
  stencil.store %lap to %lap_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
