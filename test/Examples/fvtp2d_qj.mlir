func @fvtp2d_qj(
  %q_fd    : !stencil.field<ijk,f64>,
  %crx_fd  : !stencil.field<ijk,f64>,
  %ra_x_fd : !stencil.field<ijk,f64>,
  %xfx_fd  : !stencil.field<ijk,f64>,
  %area_fd : !stencil.field<ijk,f64>,
  %q_i_fd  : !stencil.field<ijk,f64>,

  %q_j_fd  : !stencil.field<ijk,f64>,
  %fx1_fd  : !stencil.field<ijk,f64>,
  %fx2_fd  : !stencil.field<ijk,f64>)

  attributes { stencil.program } {
  // asserts
  stencil.assert %q_fd    ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %crx_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %ra_x_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %xfx_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %area_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %q_i_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>

  stencil.assert %q_j_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %fx1_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %fx2_fd  ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>

  // loads
  %q    = stencil.load %q_fd    : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %crx  = stencil.load %crx_fd  : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %ra_x = stencil.load %ra_x_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %xfx  = stencil.load %xfx_fd  : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %area = stencil.load %area_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %q_i  = stencil.load %q_i_fd  : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>

  // fx1
  %al_fx1 = stencil.apply %arg1 = %q_i : !stencil.temp<ijk,f64> {
      %one = constant 1.0 : f64
      %seven = constant 7.0 : f64
      %twelve = constant 12.0 : f64
      %p1 = divf %seven, %twelve : f64
      %p2 = divf %one, %twelve : f64

      %q_i_im1 = stencil.access %arg1[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %q_i_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %q_i_im1pcen = addf %q_i_im1, %q_i_cen : f64

      %q_i_im2 = stencil.access %arg1[-2, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %q_i_ip1 = stencil.access %arg1[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %q_i_im2pip1 = addf %q_i_im2, %q_i_ip1 : f64

      %left = mulf %p1, %q_i_im1pcen : f64
      %right = mulf %p2, %q_i_im2pip1 : f64

      %al_cen = addf %left, %right : f64

      stencil.return %al_cen : f64
  } : !stencil.temp<ijk,f64>


  %almq_fx1, %br_fx1, %b0_fx1, %smt5_fx1 = stencil.apply %arg1 = %q_i, %arg2 = %al_fx1 : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %zero = constant 0.0 : f64
      %one = constant 1.0 : f64

      %al_cen = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %q_i_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %almqfx1 = subf %al_cen, %q_i_cen : f64

      %al_ip1 = stencil.access %arg2[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %br = subf %al_ip1, %q_i_cen : f64

      %b0 = addf %almqfx1, %br : f64

      %tmp = mulf %almqfx1, %br : f64

      %smaller_zero = cmpf "olt", %tmp, %zero : f64

      %smt5 = select %smaller_zero, %one, %zero : f64

      stencil.return %almqfx1, %br, %b0, %smt5 : f64, f64, f64, f64
  } : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>


  %fx1 = stencil.apply %arg1 = %q_i, %arg2 = %crx, %arg3 = %almq_fx1, %arg4 = %br_fx1, %arg5 = %b0_fx1, %arg6 = %smt5_fx1:
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %zero = constant 0.0 : f64
      %one = constant 1.0 : f64

      %smt5_im1 = stencil.access %arg6[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %equal_zero = cmpf "oeq", %smt5_im1, %zero : f64
      %optional_smt5 = select %equal_zero, %one, %zero : f64

      %smt5_cen = stencil.access %arg6[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %smt5_prod = mulf %smt5_cen, %optional_smt5 : f64

      %smt5tmp = addf %smt5_im1, %smt5_prod : f64

      %crx_cen = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %crxgzero = cmpf "ogt", %crx_cen, %zero : f64

      %crxtmp = loop.if %crxgzero -> (f64) {
          %br_im1 = stencil.access %arg4[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
          %b0_im1 = stencil.access %arg5[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

          %crxb0 = mulf %crx_cen, %b0_im1 : f64
          %right_tmp = subf %br_im1, %crxb0 : f64

          %left_tmp = subf %one, %crx_cen : f64

          %crxpos_res = mulf %left_tmp, %right_tmp : f64
          loop.yield %crxpos_res : f64
      } else {
          %almq_cen = stencil.access %arg3[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
          %b0_cen = stencil.access %arg5[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

          %crxb0 = mulf %crx_cen, %b0_cen : f64
          %right_tmp = addf %almq_cen, %crxb0 : f64

          %left_tmp = addf %one, %crx_cen : f64

          %crxneg_res = mulf %left_tmp, %right_tmp : f64

          loop.yield %crxneg_res : f64
      }

      %tmp = mulf %crxtmp, %smt5tmp : f64

      %fx1_cen = loop.if %crxgzero -> (f64) {
          %q_i_im1 = stencil.access %arg1[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

          %crxpos_res = addf %q_i_im1, %tmp : f64
          loop.yield %crxpos_res : f64
      } else {
          %q_i_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

          %crxneg_res = addf %q_i_cen, %tmp : f64
          loop.yield %crxneg_res : f64
      }

      stencil.return %fx1_cen : f64
  } : !stencil.temp<ijk,f64>


  // fx2
  %al_fx2 = stencil.apply %arg1 = %q : !stencil.temp<ijk,f64> {
      %one = constant 1.0 : f64
      %seven = constant 7.0 : f64
      %twelve = constant 12.0 : f64
      %p1 = divf %seven, %twelve : f64
      %p2 = divf %one, %twelve : f64

      %q_im1 = stencil.access %arg1[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %q_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %q_im1pcen = addf %q_im1, %q_cen : f64

      %q_im2 = stencil.access %arg1[-2, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %q_ip1 = stencil.access %arg1[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %q_im2pip1 = addf %q_im2, %q_ip1 : f64

      %left = mulf %p1, %q_im1pcen : f64
      %right = mulf %p2, %q_im2pip1 : f64

      %al_cen = addf %left, %right : f64

      stencil.return %al_cen : f64
  } : !stencil.temp<ijk,f64>


  %almq_fx2, %br_fx2, %b0_fx2, %smt5_fx2 = stencil.apply %arg1 = %q, %arg2 = %al_fx2 : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %zero = constant 0.0 : f64
      %one = constant 1.0 : f64

      %al_cen = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %q_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %almqfx2 = subf %al_cen, %q_cen : f64

      %al_ip1 = stencil.access %arg2[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %br = subf %al_ip1, %q_cen : f64

      %b0 = addf %almqfx2, %br : f64

      %tmp = mulf %almqfx2, %br : f64

      %smaller_zero = cmpf "olt", %tmp, %zero : f64

      %smt5 = select %smaller_zero, %one, %zero : f64

      stencil.return %almqfx2, %br, %b0, %smt5 : f64, f64, f64, f64
  } : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>


  %fx2 = stencil.apply %arg1 = %q, %arg2 = %crx, %arg3 = %almq_fx2, %arg4 = %br_fx2, %arg5 = %b0_fx2, %arg6 = %smt5_fx2:
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %zero = constant 0.0 : f64
      %one = constant 1.0 : f64

      %smt5_im1 = stencil.access %arg6[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %equal_zero = cmpf "oeq", %smt5_im1, %zero : f64
      %optional_smt5 = select %equal_zero, %one, %zero : f64

      %smt5_cen = stencil.access %arg6[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %smt5_prod = mulf %smt5_cen, %optional_smt5 : f64

      %smt5tmp = addf %smt5_im1, %smt5_prod : f64

      %crx_cen = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %crxgzero = cmpf "ogt", %crx_cen, %zero : f64

      %crxtmp = loop.if %crxgzero -> (f64) {
          %br_im1 = stencil.access %arg4[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
          %b0_im1 = stencil.access %arg5[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

          %crxb0 = mulf %crx_cen, %b0_im1 : f64
          %right_tmp = subf %br_im1, %crxb0 : f64

          %left_tmp = subf %one, %crx_cen : f64

          %crxpos_res = mulf %left_tmp, %right_tmp : f64
          loop.yield %crxpos_res : f64
      } else {
          %almq_cen = stencil.access %arg3[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
          %b0_cen = stencil.access %arg5[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

          %crxb0 = mulf %crx_cen, %b0_cen : f64
          %right_tmp = addf %almq_cen, %crxb0 : f64

          %left_tmp = addf %one, %crx_cen : f64

          %crxneg_res = mulf %left_tmp, %right_tmp : f64

          loop.yield %crxneg_res : f64
      }

      %tmp = mulf %crxtmp, %smt5tmp : f64

      %fx2_cen = loop.if %crxgzero -> (f64) {
          %q_im1 = stencil.access %arg1[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

          %crxpos_res = addf %q_im1, %tmp : f64
          loop.yield %crxpos_res : f64
      } else {
          %q_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

          %crxneg_res = addf %q_cen, %tmp : f64
          loop.yield %crxneg_res : f64
      }

      stencil.return %fx2_cen : f64
  } : !stencil.temp<ijk,f64>


  // q_j
  %fxx = stencil.apply %arg1 = %xfx, %arg2 = %fx2 : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %xfx_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %fx2_cen = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %fxx_cen = mulf %xfx_cen, %fx2_cen : f64

      stencil.return %fxx_cen : f64
  } : !stencil.temp<ijk,f64>


  %q_j = stencil.apply %arg1 = %q, %arg2 = %area, %arg3 = %fxx, %arg4 = %ra_x :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64> {
      %q_cen = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %area_cen = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %qarea = mulf %q_cen, %area_cen : f64

      %fxx_cen = stencil.access %arg3[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %fxx_ip1 = stencil.access %arg3[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %fxx_cenmip1 = subf %fxx_cen, %fxx_ip1 : f64

      %tmp = addf %qarea, %fxx_cenmip1 : f64

      %rax_cen = stencil.access %arg4[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64

      %q_j_cen = divf %tmp, %rax_cen : f64

      stencil.return %q_j_cen : f64
  } : !stencil.temp<ijk,f64>

  // store results
  stencil.store %q_j to %q_j_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  stencil.store %fx1 to %fx1_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  stencil.store %fx2 to %fx2_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  return
}
