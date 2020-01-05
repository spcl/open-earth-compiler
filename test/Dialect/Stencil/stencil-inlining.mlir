// RUN: oec-opt %s 

func @hdiff(%uin : !stencil.field<ijk,f64>, %mask : !stencil.field<ijk,f64>, %uout : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	stencil.assert %uin ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %uout ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %mask ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  %0 = stencil.load %uin : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %1 = stencil.load %mask : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %2 = stencil.apply %arg1 = %0 : !stencil.view<ijk,f64> {  
      %3 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %5 = stencil.access %arg1[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %6 = stencil.access %arg1[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %7 = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %8 = addf %3, %4 : f64
      %9 = addf %5, %6 : f64
      %10 = addf %8, %9 : f64
      %cst = constant -4.000000e+00 : f64
      %11 = mulf %7, %cst : f64
      %12 = addf %11, %10 : f64
      stencil.return %12 : f64
	} : !stencil.view<ijk,f64>
  %13 = stencil.apply %arg2 = %1, %arg3 = %2 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
      %14 = stencil.access %arg3[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %15 = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %16 = subf %14, %15 : f64
      %17 = stencil.access %arg2[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %18 = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %19 = subf %17, %18 : f64
      %20 = mulf %16, %19 : f64
      %c0 = constant 0.0 : f64
      %21 = cmpf "ogt", %20, %c0 : f64
      %22 = select %21, %16, %c0 : f64
      stencil.return %22 : f64
	} : !stencil.view<ijk,f64>
	stencil.store %13 to %uout ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}

