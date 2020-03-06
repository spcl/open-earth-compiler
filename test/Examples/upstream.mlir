
func @laplace(%uin_fd : !stencil.field<ijk,f64>, %lap_fd : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	stencil.assert %uin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %lap_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  // lap
  %lap = stencil.apply %arg1 = %uin : !stencil.view<ijk,f64> {  
      %cst = constant 0.000000e+00 : f64
      %0 = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %1 = cmpf "ogt", %0, %cst : f64
      %2 = loop.if %1 -> (f64) {
        %3 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
        %4 = subf %3, %0 : f64
        loop.yield %4 : f64
      } else {
        %5 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
        %6 = subf %0, %5 : f64
        loop.yield %6 : f64
      }
      stencil.return %2 : f64
 	} : !stencil.view<ijk,f64>
  stencil.store %lap to %lap_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
