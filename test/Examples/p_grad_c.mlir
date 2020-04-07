
func @p_grad_c(
  %uin_fd : !stencil.field<ijk,f64>,
  %vin_fd : !stencil.field<ijk,f64>,
  %rdxc_fd : !stencil.field<ijk,f64>,
  %rdyc_fd : !stencil.field<ijk,f64>,
  %delpc_fd : !stencil.field<ijk,f64>,
  %gz_fd : !stencil.field<ijk,f64>,
  %pkc_fd : !stencil.field<ijk,f64>,
  %uout_fd : !stencil.field<ijk,f64>,
  %vout_fd : !stencil.field<ijk,f64>,
  %dt2 : f64)
  attributes { stencil.program } {
  stencil.assert %uin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %rdxc_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %rdyc_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %delpc_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %gz_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %pkc_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %uout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %vin = stencil.load %vin_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %rdxc = stencil.load %rdxc_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %rdyc = stencil.load %rdyc_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %delpc = stencil.load %delpc_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %gz  = stencil.load %gz_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %pkc = stencil.load %pkc_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  // wk
  %wk = stencil.apply %arg1 = %delpc : !stencil.view<ijk,f64> {
      %delpc_center = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      stencil.return %delpc_center : f64
	} : !stencil.view<ijk,f64>
  // uout
  %uout = stencil.apply %arg0 = %uin, %arg1 = %rdxc, %arg2 = %wk, %arg3 = %gz, %arg4 = %pkc, %arg5 = %dt2 :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64 {

      %gz_im1kp1 = stencil.access %arg3[-1, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_center = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_im1kp1mcenter = subf %gz_im1kp1, %gz_center : f64

      %pkc_kp1 = stencil.access %arg4[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %pkc_im1 = stencil.access %arg4[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %pkc_kp1mim1 = subf %pkc_kp1, %pkc_im1 : f64

      %prod_0 = mulf %gz_im1kp1mcenter, %pkc_kp1mim1 : f64

      %gz_im1 = stencil.access %arg3[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_kp1 = stencil.access %arg3[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_im1mkp1 = subf %gz_im1, %gz_kp1 : f64

      %pkc_im1kp1 = stencil.access %arg4[-1, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %pkc_center = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %pkc_im1kp1mcenter = subf %pkc_im1kp1, %pkc_center : f64

      %prod_1 = mulf %gz_im1mkp1, %pkc_im1kp1mcenter : f64

      %sum = addf %prod_0, %prod_1 : f64

      %wk_im1 = stencil.access %arg2[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %wk_center = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %wk_im1pcenter = addf %wk_im1, %wk_center : f64

      %rdxc_center = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %rdxcdt2 = mulf %rdxc_center, %arg5 : f64

      %prefac = divf %rdxcdt2, %wk_im1pcenter : f64

      %delta = mulf %prefac, %sum : f64

      %uin_center = stencil.access %arg0[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %result = addf %delta, %uin_center : f64

      stencil.return %result : f64
	} : !stencil.view<ijk,f64>
  // vout
  %vout = stencil.apply %arg0 = %vin, %arg1 = %rdyc, %arg2 = %wk, %arg3 = %gz, %arg4 = %pkc, %arg5 = %dt2 :
  !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64 {

      %gz_jm1kp1 = stencil.access %arg3[0, -1, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_center = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_jm1kp1mcenter = subf %gz_jm1kp1, %gz_center : f64

      %pkc_kp1 = stencil.access %arg4[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %pkc_jm1 = stencil.access %arg4[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %pkc_kp1mjm1 = subf %pkc_kp1, %pkc_jm1 : f64

      %prod_0 = mulf %gz_jm1kp1mcenter, %pkc_kp1mjm1 : f64

      %gz_jm1 = stencil.access %arg3[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %gz_kp1 = stencil.access %arg3[0, 0, 1] : (!stencil.view<ijk,f64>) -> f64
      %gz_jm1mkp1 = subf %gz_jm1, %gz_kp1 : f64

      %pkc_jm1kp1 = stencil.access %arg4[0, -1, 1] : (!stencil.view<ijk,f64>) -> f64
      %pkc_center = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %pkc_jm1kp1mcenter = subf %pkc_jm1kp1, %pkc_center : f64

      %prod_1 = mulf %gz_jm1mkp1, %pkc_jm1kp1mcenter : f64

      %sum = addf %prod_0, %prod_1 : f64

      %wk_jm1 = stencil.access %arg2[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %wk_center = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %wk_jm1pcenter = addf %wk_jm1, %wk_center : f64

      %rdyc_center = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %rdycdt2 = mulf %rdyc_center, %arg5 : f64

      %prefac = divf %rdycdt2, %wk_jm1pcenter : f64

      %delta = mulf %prefac, %sum : f64

      %uin_center = stencil.access %arg0[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %result = addf %delta, %uin_center : f64

      stencil.return %result : f64
	} : !stencil.view<ijk,f64>
	stencil.store %uout to %uout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
	stencil.store %vout to %vout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
