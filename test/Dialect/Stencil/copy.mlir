// RUN: oec-opt %s 

func @copy(%in : !stencil.field<ijk,f64>, %out : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	stencil.assert %in ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %out ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  %0 = stencil.load %in : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %1 = stencil.apply %arg1 = %0 : !stencil.view<ijk,f64> {  
      %2 = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      stencil.return %2 : f64
	} : !stencil.view<ijk,f64>
	stencil.store %1 to %out ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
