// RUN: oec-opt %s --stencil-shape-inference | oec-opt | FileCheck %s

func @lap_stencil(%in: !stencil.field<ijk,f64>, %out: !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	%0 = stencil.load %in : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
	%1 = stencil.apply %in1 = %0 : !stencil.view<ijk,f64> {  
      %10 = stencil.access %in1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %11 = stencil.access %in1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %12 = stencil.access %in1[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %13 = stencil.access %in1[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %14 = stencil.access %in1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %15 = addf %10, %11 : f64
      %16 = addf %12, %13 : f64
      %17 = addf %15, %16 : f64
      %cst = constant -4.000000e+00 : f64
      %18 = mulf %14, %cst : f64
      %19 = addf %18, %17 : f64
      stencil.return %19 : f64
	} : !stencil.view<ijk,f64>
	%2 = stencil.apply %in2 = %1 : !stencil.view<ijk,f64> {  
      %20 = stencil.access %in2[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %21 = stencil.access %in2[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %22 = stencil.access %in2[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %23 = stencil.access %in2[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %24 = stencil.access %in2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %25 = addf %20, %21 : f64
      %26 = addf %22, %23 : f64
      %27 = addf %25, %26 : f64
      %cst = constant -4.000000e+00 : f64
      %28 = mulf %24, %cst : f64
      %29 = addf %28, %27 : f64
      stencil.return %29 : f64
	} : !stencil.view<ijk,f64>
	stencil.store %2 to %out([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
	return
}

// CHECK-LABEL: func @lap_stencil(%{{.*}}: !stencil.field<ijk,f64>, %{{.*}}: !stencil.field<ijk,f64>) attributes {stencil.program} {
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} ([-2, -2, 0]:[66, 66, 60]) : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>

//       CHECK: } to ([-1, -1, 0]:[65, 65, 60]) : !stencil.view<ijk,f64>
//       CHECK: } to ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64>

