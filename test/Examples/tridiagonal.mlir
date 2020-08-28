

module {
  func @tridiagonal(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %4 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %5:2 = stencil.apply seq(dim = 2, range = 0 to 64, dir = 1) (%arg3 = %3 : !stencil.temp<?x?x?xf64>, %arg4 = %4 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %7 = stencil.index 2 [0, 0, 0] : index
      %c0 = constant 0 : index
      %8 = cmpi "eq", %7, %c0 : index
      %cst = constant 3.000000e+00 : f64
      %cst_0 = constant -1.000000e+00 : f64
      %9:2 = scf.if %8 -> (f64, f64) {
        %10 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %11 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %12 = divf %10, %cst : f64
        %13 = divf %11, %cst : f64
        scf.yield %12, %13 : f64, f64
      } else {
        %10 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %11 = stencil.depend 1 [0, 0, -1] : f64
        %12 = mulf %11, %cst_0 : f64
        %13 = subf %cst, %12 : f64
        %14 = divf %10, %13 : f64
        %15 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %16 = stencil.depend 0 [0, 0, -1] : f64
        %17 = mulf %cst_0, %16 : f64
        %18 = subf %15, %17 : f64
        %19 = mulf %11, %cst_0 : f64
        %20 = subf %cst, %19 : f64
        %21 = divf %18, %20 : f64
        scf.yield %21, %14 : f64, f64
      }
      stencil.return %9#0, %9#1 : f64, f64
    }
    %6 = stencil.apply seq(dim = 2, range = 0 to 64, dir = -1) (%arg3 = %5#0 : !stencil.temp<?x?x?xf64>, %arg4 = %5#1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %7 = stencil.index 2 [0, 0, 0] : index
      %c59 = constant 59 : index
      %8 = cmpi "eq", %7, %c59 : index
      %9 = scf.if %8 -> (f64) {
        %10 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        scf.yield %10 : f64
      } else {
        %10 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %11 = stencil.access %arg4 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %12 = stencil.depend 0 [0, 0, 1] : f64
        %13 = mulf %11, %12 : f64
        %14 = addf %13, %10 : f64
        scf.yield %14 : f64
      }
      stencil.return %9 : f64
    }
    stencil.store %6 to %2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
