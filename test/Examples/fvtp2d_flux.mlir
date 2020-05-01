func @fvtp2d_flux(
  %cry_fd  : !stencil.field<ijk,f64>,
  %xfx_fd  : !stencil.field<ijk,f64>,
  %yfx_fd  : !stencil.field<ijk,f64>,
  %q_j_fd  : !stencil.field<ijk,f64>,
  %fx1i_fd  : !stencil.field<ijk,f64>,
  %fx2_fd  : !stencil.field<ijk,f64>,
  %fy2_fd  : !stencil.field<ijk,f64>,

  %fx1_fd  : !stencil.field<ijk,f64>,
  %fy1_fd  : !stencil.field<ijk,f64>)

  attributes { stencil.program } {
  // asserts
  stencil.assert %cry_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %xfx_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %yfx_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>

  stencil.assert %q_j_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %fx1i_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %fx2_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %fy2_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>

  stencil.assert %fx1_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %fy1_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>

  // loads
  %cry  = stencil.load %cry_fd  : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %xfx  = stencil.load %xfx_fd  : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %yfx  = stencil.load %yfx_fd  : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>

  %q_j  = stencil.load %q_j_fd  : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %fx1i = stencil.load %fx1i_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %fx2  = stencil.load %fx2_fd  : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %fy2  = stencil.load %fy2_fd  : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>

  // fy1
  %al_fy1 = stencil.apply %arg1 = %q_j : !stencil.temp<ijk,f64> {
      %one = constant 1.0 : f64
      %seven = constant 7.0 : f64
      %twelve = constant 12.0 : f64
      %p1 = divf %seven, %twelve : f64
      %p2 = divf %one, %twelve : f64

      %q_j_jm1 = stencil.access %arg1[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %q_j_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %q_j_jm1pcen = addf %q_j_jm1, %q_j_cen : f64

      %q_j_jm2 = stencil.access %arg1[0, -2, 0] : (!stencil.temp<ijk,f64>) -> f64
      %q_j_jp1 = stencil.access %arg1[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64

      %q_j_jm2pjp1 = addf %q_j_jm2, %q_j_jp1 : f64

      %left = mulf %p1, %q_j_jm1pcen : f64
      %right = mulf %p2, %q_j_jm2pjp1 : f64

      %al_cen = addf %left, %right : f64

      stencil.return %al_cen : f64
  } : !stencil.temp<ijk,f64>


  %almq_fy1, %br_fy1, %b0_fy1, %smt5_fy1 = stencil.apply %arg1 = %q_j, %arg2 = %al_fy1 : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %zero = constant 0.0 : f64
      %one = constant 1.0 : f64

      %al_cen = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %q_j_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %almqfy1 = subf %al_cen, %q_j_cen : f64

      %al_jp1 = stencil.access %arg2[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64

      %br = subf %al_jp1, %q_j_cen : f64

      %b0 = addf %almqfy1, %br : f64

      %tmp = mulf %almqfy1, %br : f64

      %smaller_zero = cmpf "olt", %tmp, %zero : f64

      %smt5 = select %smaller_zero, %one, %zero : f64

      stencil.return %almqfy1, %br, %b0, %smt5 : f64, f64, f64, f64
  } : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>


  %fy1_tmp = stencil.apply %arg1 = %q_j, %arg2 = %cry, %arg3 = %almq_fy1, %arg4 = %br_fy1, %arg5 = %b0_fy1, %arg6 = %smt5_fy1:
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %zero = constant 0.0 : f64
      %one = constant 1.0 : f64

      %smt5_jm1 = stencil.access %arg6[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64

      %equal_zero = cmpf "oeq", %smt5_jm1, %zero : f64
      %optional_smt5 = select %equal_zero, %one, %zero : f64

      %smt5_cen = stencil.access %arg6[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %smt5_prod = mulf %smt5_cen, %optional_smt5 : f64

      %smt5tmp = addf %smt5_jm1, %smt5_prod : f64

      %cry_cen = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %crygzero = cmpf "ogt", %cry_cen, %zero : f64

      %crytmp = loop.if %crygzero -> (f64) {
          %br_jm1 = stencil.access %arg4[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
          %b0_jm1 = stencil.access %arg5[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64

          %cryb0 = mulf %cry_cen, %b0_jm1 : f64
          %right_tmp = subf %br_jm1, %cryb0 : f64

          %left_tmp = subf %one, %cry_cen : f64

          %crypos_res = mulf %left_tmp, %right_tmp : f64
          loop.yield %crypos_res : f64
      } else {
          %almq_cen = stencil.access %arg3[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
          %b0_cen = stencil.access %arg5[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

          %cryb0 = mulf %cry_cen, %b0_cen : f64
          %right_tmp = addf %almq_cen, %cryb0 : f64

          %left_tmp = addf %one, %cry_cen : f64

          %cryneg_res = mulf %left_tmp, %right_tmp : f64

          loop.yield %cryneg_res : f64
      }

      %tmp = mulf %crytmp, %smt5tmp : f64

      %fy1_cen = loop.if %crygzero -> (f64) {
          %q_j_jm1 = stencil.access %arg1[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64

          %crypos_res = addf %q_j_jm1, %tmp : f64
          loop.yield %crypos_res : f64
      } else {
          %q_j_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

          %cryneg_res = addf %q_j_cen, %tmp : f64
          loop.yield %cryneg_res : f64
      }

      stencil.return %fy1_cen : f64
  } : !stencil.temp<ijk,f64>


  // fx1
  %fx1 = stencil.apply %arg1 = %fx1i, %arg2 = %fx2, %arg3 = %xfx :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %half = constant 0.5 : f64

      %fx1_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %fx2_cen = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %fx_sum = addf %fx1_cen, %fx2_cen : f64

      %half_sum = mulf %fx_sum, %half : f64

      %xfx_cen = stencil.access %arg3[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %result = mulf %xfx_cen, %half_sum : f64

      stencil.return %result : f64
  } : !stencil.temp<ijk,f64>


  // fy1
  %fy1 = stencil.apply %arg1 = %fy1_tmp, %arg2 = %fy2, %arg3 = %yfx :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %half = constant 0.5 : f64

      %fy1_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %fy2_cen = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %fx_sum = addf %fy1_cen, %fy2_cen : f64

      %half_sum = mulf %fx_sum, %half : f64

      %yfx_cen = stencil.access %arg3[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %result = mulf %yfx_cen, %half_sum : f64

      stencil.return %result : f64
  } : !stencil.temp<ijk,f64>


  // store results
  stencil.store %fx1 to %fx1_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  stencil.store %fy1 to %fy1_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  return
}
