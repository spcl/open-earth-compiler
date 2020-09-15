

module {
  func @copy(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %4 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %5 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
      stencil.return %5 : !stencil.result<f64>
    }
    stencil.store %3 to %1([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
