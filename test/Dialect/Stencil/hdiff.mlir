// RUN: oec-opt %s 

func @lap_stencil(%uin : !stencil.field<ijk,f64>, %mask : !stencil.field<ijk,f64>, %uout : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	stencil.assert %uin ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %mask ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %uout ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
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
  %13 = stencil.apply %arg2 = %0, %arg3 = %2 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
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
  %23 = stencil.apply %arg4 = %0, %arg5 = %2 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
      %24 = stencil.access %arg5[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %25 = stencil.access %arg5[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %26 = subf %24, %25 : f64
      %27 = stencil.access %arg4[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %28 = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %29 = subf %27, %28 : f64
      %30 = mulf %26, %29 : f64
      %c0 = constant 0.0 : f64
      %31 = cmpf "ogt", %30, %c0 : f64
      %32 = select %31, %26, %c0 : f64
      stencil.return %32 : f64
	} : !stencil.view<ijk,f64>
  %33 = stencil.apply %arg6 = %0, %arg7 = %13, %arg8 = %23, %arg9 = %1 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
      %34 = stencil.access %arg7[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %35 = stencil.access %arg7[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %36 = subf %34, %35 : f64
      %37 = stencil.access %arg8[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %38 = stencil.access %arg8[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %39 = subf %37, %38 : f64
      %40 = addf %36, %39 : f64
      %41 = stencil.access %arg9[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %42 = mulf %41, %40 : f64
      %43 = stencil.access %arg6[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %44 = addf %42, %43 : f64
      stencil.return %44 : f64
	} : !stencil.view<ijk,f64>
	stencil.store %33 to %uout ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
