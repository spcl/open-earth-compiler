func @fvtp2d_qi(
  %q_fd    : !stencil.field<ijk,f64>,
  %cry_fd  : !stencil.field<ijk,f64>,
  %ra_y_fd : !stencil.field<ijk,f64>,
  %yfx_fd  : !stencil.field<ijk,f64>,
  %area_fd : !stencil.field<ijk,f64>,

  %q_i_fd  : !stencil.field<ijk,f64>,
  %fy2_fd  : !stencil.field<ijk,f64>)

  attributes { stencil.program } {
  // asserts
  stencil.assert %q_fd    ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %cry_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %ra_y_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %yfx_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %area_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>

  stencil.assert %q_i_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %fy2_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>

  // loads
  %q    = stencil.load %q_fd    : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %cry  = stencil.load %cry_fd  : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %ra_y = stencil.load %ra_y_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %yfx  = stencil.load %yfx_fd  : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %area = stencil.load %area_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>

  // fy2
  %al_fy2 = stencil.apply %arg1 = %q : !stencil.view<ijk,f64> {
      %one = constant 1.0 : f64
      %seven = constant 7.0 : f64
      %twelve = constant 12.0 : f64
      %p1 = divf %seven, %twelve : f64
      %p2 = divf %one, %twelve : f64

      %q_jm1 = stencil.access %arg1[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %q_cen = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %q_jm1pcen = addf %q_jm1, %q_cen : f64

      %q_jm2 = stencil.access %arg1[0, -2, 0] : (!stencil.view<ijk,f64>) -> f64
      %q_jp1 = stencil.access %arg1[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64

      %q_jm2pjp1 = addf %q_jm2, %q_jp1 : f64

      %left = mulf %p1, %q_jm1pcen : f64
      %right = mulf %p2, %q_jm2pjp1 : f64

      %al_cen = addf %left, %right : f64

      stencil.return %al_cen : f64
  } : !stencil.view<ijk,f64>


  %almq_fy2, %br_fy2, %b0_fy2, %smt5_fy2 = stencil.apply %arg1 = %q, %arg2 = %al_fy2 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
      %zero = constant 0.0 : f64
      %one = constant 1.0 : f64

      %al_cen = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %q_cen = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %almqfy2 = subf %al_cen, %q_cen : f64

      %al_jp1 = stencil.access %arg2[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64

      %br = subf %al_jp1, %q_cen : f64

      %b0 = addf %almqfy2, %br : f64

      %tmp = mulf %almqfy2, %br : f64

      %smaller_zero = cmpf "olt", %tmp, %zero : f64

      %smt5 = select %smaller_zero, %one, %zero : f64

      stencil.return %almqfy2, %br, %b0, %smt5 : f64, f64, f64, f64
  } : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>


  %fy2 = stencil.apply %arg1 = %q, %arg2 = %cry, %arg3 = %almq_fy2, %arg4 = %br_fy2, %arg5 = %b0_fy2, %arg6 = %smt5_fy2:
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
      %zero = constant 0.0 : f64
      %one = constant 1.0 : f64

      %smt5_jm1 = stencil.access %arg6[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64

      %equal_zero = cmpf "oeq", %smt5_jm1, %zero : f64
      %optional_smt5 = select %equal_zero, %one, %zero : f64

      %smt5_cen = stencil.access %arg6[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %smt5_prod = mulf %smt5_cen, %optional_smt5 : f64

      %smt5tmp = addf %smt5_jm1, %smt5_prod : f64

      %cry_cen = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %crygzero = cmpf "ogt", %cry_cen, %zero : f64

      %crytmp = loop.if %crygzero -> (f64) {
          %br_jm1 = stencil.access %arg4[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
          %b0_jm1 = stencil.access %arg5[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64

          %cryb0 = mulf %cry_cen, %b0_jm1 : f64
          %right_tmp = subf %br_jm1, %cryb0 : f64

          %left_tmp = subf %one, %cry_cen : f64

          %crypos_res = mulf %left_tmp, %right_tmp : f64
          loop.yield %crypos_res : f64
      } else {
          %almq_cen = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
          %b0_cen = stencil.access %arg5[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

          %cryb0 = mulf %cry_cen, %b0_cen : f64
          %right_tmp = addf %almq_cen, %cryb0 : f64

          %left_tmp = addf %one, %cry_cen : f64

          %cryneg_res = mulf %left_tmp, %right_tmp : f64

          loop.yield %cryneg_res : f64
      }

      %tmp = mulf %crytmp, %smt5tmp : f64

      %fy2_cen = loop.if %crygzero -> (f64) {
          %q_jm1 = stencil.access %arg1[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64

          %crypos_res = addf %q_jm1, %tmp : f64
          loop.yield %crypos_res : f64
      } else {
          %q_cen = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

          %cryneg_res = addf %q_cen, %tmp : f64
          loop.yield %cryneg_res : f64
      }

      stencil.return %fy2_cen : f64
  } : !stencil.view<ijk,f64>


  // q_i
  %fyy = stencil.apply %arg1 = %yfx, %arg2 = %fy2 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
      %yfx_cen = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %fy2_cen = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %fyy_cen = mulf %yfx_cen, %fy2_cen : f64

      stencil.return %fyy_cen : f64
  } : !stencil.view<ijk,f64>


  %q_i = stencil.apply %arg1 = %q, %arg2 = %area, %arg3 = %fyy, %arg4 = %ra_y :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
      %q_cen = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %area_cen = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %qarea = mulf %q_cen, %area_cen : f64

      %fyy_cen = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %fyy_jp1 = stencil.access %arg3[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64

      %fyy_cenmjp1 = subf %fyy_cen, %fyy_jp1 : f64

      %tmp = addf %qarea, %fyy_cenmjp1 : f64

      %ray_cen = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %q_i_cen = divf %tmp, %ray_cen : f64

      stencil.return %q_i_cen : f64
  } : !stencil.view<ijk,f64>

  // store results
  stencil.store %fy2 to %fy2_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  stencil.store %q_i to %q_i_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
