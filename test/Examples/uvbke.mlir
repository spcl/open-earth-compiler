
func @uvbke(
  %uc_fd : !stencil.field<ijk,f64>,
  %vc_fd : !stencil.field<ijk,f64>,
  %cosa_fd : !stencil.field<ijk,f64>,
  %rsina_fd : !stencil.field<ijk,f64>,
  %ub_fd : !stencil.field<ijk,f64>,
  %vb_fd : !stencil.field<ijk,f64>,
  %dt5 : f64)
  attributes { stencil.program } {
  // asserts
  stencil.assert %uc_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vc_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %cosa_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %rsina_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %ub_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vb_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  // loads
  %uc = stencil.load %uc_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %vc = stencil.load %vc_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %cosa = stencil.load %cosa_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %rsina = stencil.load %rsina_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>

  // ub
  %ub = stencil.apply %arg1 = %uc, %arg2 = %vc, %arg3 = %cosa, %arg4 = %rsina, %arg5 = %dt5 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64 {

      %vc_im1 = stencil.access %arg2[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %vc_center = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %vc_im1pcenter = addf %vc_im1, %vc_center : f64

      %cosa_center = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %vccosa = mulf %vc_im1pcenter, %cosa_center : f64

      %uc_jm1 = stencil.access %arg1[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %uc_center = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %uc_jm1pcenter = addf %uc_jm1, %uc_center : f64

      %ucvccosa = subf %uc_jm1pcenter, %vccosa : f64

      %ucvccosadt5 = mulf %arg5, %ucvccosa : f64

      %rsina_center = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %ub_center = mulf %rsina_center, %ucvccosadt5 : f64

      stencil.return %ub_center : f64
  } : !stencil.view<ijk,f64>

  // vb
  %vb = stencil.apply %arg1 = %uc, %arg2 = %vc, %arg3 = %cosa, %arg4 = %rsina, %arg5 = %dt5 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, f64 {

      %uc_jm1 = stencil.access %arg1[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %uc_center = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %uc_jm1pcenter = addf %uc_jm1, %uc_center : f64

      %cosa_center = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %uccosa = mulf %uc_jm1pcenter, %cosa_center : f64

      %vc_im1 = stencil.access %arg2[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %vc_center = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %vc_im1pcenter = addf %vc_im1, %vc_center : f64

      %ucvccosa = subf %vc_im1pcenter, %uccosa : f64

      %ucvccosadt5 = mulf %arg5, %ucvccosa : f64

      %rsina_center = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

      %vb_center = mulf %rsina_center, %ucvccosadt5 : f64

      stencil.return %vb_center : f64
  } : !stencil.view<ijk,f64>

  // store results
  stencil.store %ub to %ub_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  stencil.store %vb to %vb_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}
