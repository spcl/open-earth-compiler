

module {
  func @tridiagonal(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    stencil.assert %arg0([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg1([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %arg2([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    %0 = stencil.load %arg0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %1 = stencil.load %arg1 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %2:2 = stencil.apply seq(dim = 2, range = 0 to 60, dir = 1) (%arg3 = %0 : !stencil.temp<?x?x?xf64>, %arg4 = %1 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %4 = stencil.index 2 [0, 0, 0] : index
      %c0 = constant 0 : index
      %5 = cmpi "eq", %4, %c0 : index
      %cst = constant 3.000000e+00 : f64
      %cst_0 = constant -1.000000e+00 : f64
      %6:2 = scf.if %5 -> (f64, f64) {
        %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %8 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %9 = divf %7, %cst : f64
        %10 = divf %8, %cst : f64
        scf.yield %9, %10 : f64, f64
      } else {
        %7 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %8 = stencil.depend 1 [0, 0, -1] : f64
        %9 = mulf %8, %cst_0 : f64
        %10 = subf %cst, %9 : f64
        %11 = divf %7, %10 : f64
        %12 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %13 = stencil.depend 0 [0, 0, -1] : f64
        %14 = mulf %cst_0, %13 : f64
        %15 = subf %12, %14 : f64
        %16 = mulf %8, %cst_0 : f64
        %17 = subf %cst, %16 : f64
        %18 = divf %15, %17 : f64
        scf.yield %18, %11 : f64, f64
      }
      stencil.return %6#0, %6#1 : f64, f64
    }
    %3 = stencil.apply seq(dim = 2, range = 0 to 60, dir = -1) (%arg3 = %2#0 : !stencil.temp<?x?x?xf64>, %arg4 = %2#1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %4 = stencil.index 2 [0, 0, 0] : index
      %c59 = constant 59 : index
      %5 = cmpi "eq", %4, %c59 : index
      %6 = scf.if %5 -> (f64) {
        %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        scf.yield %7 : f64
      } else {
        %7 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %8 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %9 = stencil.depend 0 [0, 0, 1] : f64
        %10 = mulf %8, %9 : f64
        %11 = addf %10, %7 : f64
        scf.yield %11 : f64
      }
      stencil.return %6 : f64
    }
    stencil.store %3 to %arg2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
