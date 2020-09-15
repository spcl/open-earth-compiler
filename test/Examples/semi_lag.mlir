module {
  func @semi_lag(
    %in: !stencil.field<?x?x?xf64>,
    %out: !stencil.field<?x?x?xf64>,
    %adeltat: f64)
  attributes {stencil.program} {
    %0 = stencil.cast %in([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %out([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>, %arg3 = %adeltat : f64) -> !stencil.temp<?x?x?xf64> {

      %one_index = constant 1 : index
      %one_int = constant 1 : i64
      %one_float = constant 1.0 : f64
      %zero_float = constant 0.0 : f64

      %i_idx = stencil.index 0 [0, 0, 0] : index
      %j_idx = stencil.index 1 [0, 0, 0] : index
      %k_idx = stencil.index 2 [0, 0, 0] : index

      %smallerzero = cmpf "olt", %arg3, %zero_float : f64
      %offset_0 = fptosi %arg3 : f64 to i64
      %offset_m1 = subi %offset_0, %one_int : i64
      %offset = select %smallerzero, %offset_m1, %offset_0 : i64
      %offset_float = sitofp %offset : i64 to f64
      %sigma = subf %arg3, %offset_float : f64
      %offset_index = index_cast %offset : i64 to index
      %shifted_i_idx = addi %i_idx, %offset_index : index
      %8 = stencil.dyn_access %arg2(%shifted_i_idx, %j_idx, %k_idx) in [-4, -4, -4] : [4, 4, 4] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = addi %shifted_i_idx, %one_index : index
      %10 = stencil.dyn_access %arg2(%9, %j_idx, %k_idx) in [-4, -4, -4] : [4, 4, 4] : (!stencil.temp<?x?x?xf64>) -> f64
      %onemsigma = subf %one_float, %sigma : f64
      %12 = mulf %onemsigma, %8 : f64
      %13 = mulf %sigma, %10 : f64
      %14 = addf %12, %13 : f64
      %15 = stencil.store_result %14 : (f64) -> !stencil.result<f64>
      stencil.return %15 : !stencil.result<f64>
    }
    stencil.store %3 to %1([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
