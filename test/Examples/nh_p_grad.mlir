
func @nh_p_grad(
  %uin_fd : !stencil.field<ijk,f64>,
  %vin_fd : !stencil.field<ijk,f64>,
  %rdx_fd : !stencil.field<ijk,f64>,
  %rdy_fd : !stencil.field<ijk,f64>,
  %gz_fd : !stencil.field<ijk,f64>,
  %pp_fd : !stencil.field<ijk,f64>,
  %pk3_fd : !stencil.field<ijk,f64>,
  %wk1_fd : !stencil.field<ijk,f64>,
  %uout_fd : !stencil.field<ijk,f64>,
  %vout_fd : !stencil.field<ijk,f64>,
  %dt : f64)
  attributes { stencil.program } {
  stencil.assert %uin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %rdx_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %rdy_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %gz_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %pp_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %pk3_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %wk1_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %uout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %vin = stencil.load %vin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %rdx = stencil.load %rdx_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %rdy = stencil.load %rdy_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %gz = stencil.load %gz_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %pp = stencil.load %pp_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %pk3 = stencil.load %pk3_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %wk1 = stencil.load %wk1_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  // wk
  %wk = stencil.apply %arg1 = %pk3 : !stencil.view<ijk,f64> {
      %pk3_kp1 = stencil.access %arg1[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %pk3_center = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %pk3_kp1mcenter = subf %pk3_kp1, %pk3_center : f64
      stencil.return %pk3_kp1mcenter : f64
	} : !stencil.view<ijk,f64>
  // du
  %du = stencil.apply %arg1 = %wk, %arg2 = %gz, %arg3 = %pk3, %arg4 = %dt :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64 {

      %gz_kp1 = stencil.access %arg2[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_ip1 = stencil.access %arg2[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_kp1mip1 = subf %gz_kp1, %gz_ip1 : f64

      %pk3_ip1kp1 = stencil.access %arg3[1, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %pk3_center = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %pk3_ip1kp1mcenter = subf %pk3_ip1kp1, %pk3_center : f64

      %prod_0 = mulf %gz_kp1mip1, %pk3_ip1kp1mcenter : f64

      %gz_center = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_ip1kp1 = stencil.access %arg2[1, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_centermip1kp1 = subf %gz_center, %gz_ip1kp1 : f64

      %pk3_kp1 = stencil.access %arg3[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %pk3_ip1 = stencil.access %arg3[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %pk3_kp1mip1 = subf %pk3_kp1, %pk3_ip1 : f64

      %prod_1 = mulf %gz_centermip1kp1, %pk3_kp1mip1 : f64

      %sum = addf %prod_0, %prod_1 : f64

      %wk_center = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %wk_ip1 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %wk_centerpip1 = addf %wk_center, %wk_ip1 : f64

      %wkdt = divf %arg4, %wk_centerpip1 : f64

      %du_center = mulf %sum, %wkdt : f64

      stencil.return %du_center : f64
	} : !stencil.view<ijk,f64>
  // dv
  %dv = stencil.apply %arg1 = %wk, %arg2 = %gz, %arg3 = %pk3, %arg4 = %dt :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64 {

      %gz_kp1 = stencil.access %arg2[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_jp1 = stencil.access %arg2[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_kp1mjp1 = subf %gz_kp1, %gz_jp1 : f64

      %pk3_jp1kp1 = stencil.access %arg3[0, 1, 1] : (!stencil.view<ijk,f64>) -> f64
      %pk3_center = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %pk3_jp1kp1mcenter = subf %pk3_jp1kp1, %pk3_center : f64

      %prod_0 = mulf %gz_kp1mjp1, %pk3_jp1kp1mcenter : f64

      %gz_center = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_jp1kp1 = stencil.access %arg2[0, 1, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_centermjp1kp1 = subf %gz_center, %gz_jp1kp1 : f64

      %pk3_kp1 = stencil.access %arg3[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %pk3_jp1 = stencil.access %arg3[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %pk3_kp1mjp1 = subf %pk3_kp1, %pk3_jp1 : f64

      %prod_1 = mulf %gz_centermjp1kp1, %pk3_kp1mjp1 : f64

      %sum = addf %prod_0, %prod_1 : f64

      %wk_center = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %wk_jp1 = stencil.access %arg1[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64

      %wk_centerpjp1 = addf %wk_center, %wk_jp1 : f64

      %wkdt = divf %arg4, %wk_centerpjp1 : f64

      %dv_center = mulf %sum, %wkdt : f64

      stencil.return %dv_center : f64
	} : !stencil.view<ijk,f64>
  // uout
  %uout = stencil.apply %arg0 = %uin ,%arg1 = %wk1, %arg2 = %gz, %arg3 = %pp, %arg4 = %du, %arg5 = %rdx, %arg6 = %dt :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64 {

      %gz_kp1 = stencil.access %arg2[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_ip1 = stencil.access %arg2[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_kp1mip1 = subf %gz_kp1, %gz_ip1 : f64

      %pp_ip1kp1 = stencil.access %arg3[1, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %pp_center = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %pp_ip1kp1mcenter = subf %pp_ip1kp1, %pp_center : f64

      %prod_0 = mulf %gz_kp1mip1, %pp_ip1kp1mcenter : f64

      %gz_center = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_ip1kp1 = stencil.access %arg2[1, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_centermip1kp1 = subf %gz_center, %gz_ip1kp1 : f64

      %pp_kp1 = stencil.access %arg3[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %pp_ip1 = stencil.access %arg3[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %pp_kp1mip1 = subf %pp_kp1, %pp_ip1 : f64

      %prod_1 = mulf %gz_centermip1kp1, %pp_kp1mip1 : f64

      %sum = addf %prod_0, %prod_1 : f64

      %wk1_center = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %wk1_ip1 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %wk1_centerpip1 = addf %wk1_center, %wk1_ip1 : f64

      %wk1dt = divf %arg6, %wk1_centerpip1 : f64

      %wk1dtsum = mulf %sum, %wk1dt : f64

      %uin_center = stencil.access %arg0[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %du_center = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %u_sum = addf %uin_center, %du_center : f64

      %full_sum = addf %wk1dtsum, %u_sum : f64

      %rdx_center = stencil.access %arg5[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %uout_center = mulf %full_sum, %rdx_center : f64

      stencil.return %uout_center : f64
	} : !stencil.view<ijk,f64>
  // vout
  %vout = stencil.apply %arg0 = %vin ,%arg1 = %wk1, %arg2 = %gz, %arg3 = %pp, %arg4 = %dv, %arg5 = %rdy, %arg6 = %dt :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64 {

      %gz_kp1 = stencil.access %arg2[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_jp1 = stencil.access %arg2[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_kp1mjp1 = subf %gz_kp1, %gz_jp1 : f64

      %pp_jp1kp1 = stencil.access %arg3[0, 1, 1] : (!stencil.view<ijk,f64>) -> f64
      %pp_center = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %pp_jp1kp1mcenter = subf %pp_jp1kp1, %pp_center : f64

      %prod_0 = mulf %gz_kp1mjp1, %pp_jp1kp1mcenter : f64

      %gz_center = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_jp1kp1 = stencil.access %arg2[0, 1, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_centermjp1kp1 = subf %gz_center, %gz_jp1kp1 : f64

      %pp_kp1 = stencil.access %arg3[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %pp_jp1 = stencil.access %arg3[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %pp_kp1mjp1 = subf %pp_kp1, %pp_jp1 : f64

      %prod_1 = mulf %gz_centermjp1kp1, %pp_kp1mjp1 : f64

      %sum = addf %prod_0, %prod_1 : f64

      %wk1_center = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %wk1_jp1 = stencil.access %arg1[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64

      %wk1_centerpjp1 = addf %wk1_center, %wk1_jp1 : f64

      %wk1dt = divf %arg6, %wk1_centerpjp1 : f64

      %wk1dtsum = mulf %sum, %wk1dt : f64

      %vin_center = stencil.access %arg0[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %dv_center = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %v_sum = addf %vin_center, %dv_center : f64

      %full_sum = addf %wk1dtsum, %v_sum : f64

      %rdy_center = stencil.access %arg5[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %vout_center = mulf %full_sum, %rdy_center : f64

      stencil.return %vout_center : f64
	} : !stencil.view<ijk,f64>
	stencil.store %uout to %uout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
	stencil.store %vout to %vout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
