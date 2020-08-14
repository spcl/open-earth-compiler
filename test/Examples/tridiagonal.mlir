
module {
  func @tridiagonal(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2 = stencil.apply seq(dim=2, range=0 to 64, dir=-1) (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %k = stencil.index 2 [0,0,0] : index
      %clast = constant 63 : index
      %top = cmpi "eq", %k, %clast : index
      %res = scf.if %top -> (f64) {
        %3 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        scf.yield %3 : f64
      } else {
        %4 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %5 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %6 = stencil.depend 0 [0, 0, 1] : f64
        %7 = mulf %5, %6 : f64
        %8 = addf %7, %4 : f64
        scf.yield %8 : f64
      }
      stencil.return %res : f64
    }
    stencil.store %2 to %arg2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
