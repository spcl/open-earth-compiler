func @hdiffsmag(
  %uin_fd : !stencil.field<ijk,f64>,
  %vin_fd : !stencil.field<ijk,f64>,
  %mask_fd : !stencil.field<ijk,f64>,
  %uout_fd : !stencil.field<ijk,f64>,
  %vout_fd : !stencil.field<ijk,f64>,
  %crlavo_fd : !stencil.field<j,f64>,
  %crlavu_fd : !stencil.field<j,f64>,
  %crlato_fd : !stencil.field<j,f64>,
  %crlatu_fd : !stencil.field<j,f64>,
  %acrlat0_fd : !stencil.field<j,f64>,
  %eddlat : f64,
  %eddlon : f64,
  %tau_smag : f64,
  %weight_smag : f64)
  attributes { stencil.program } {
  // asserts
  stencil.assert %uin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %mask_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %uout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %crlavo_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  stencil.assert %crlavu_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  stencil.assert %crlato_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  stencil.assert %crlatu_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  stencil.assert %acrlat0_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  // loads
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %vin = stencil.load %vin_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %mask = stencil.load %mask_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %crlavo = stencil.load %crlavo_fd : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
  %crlavu = stencil.load %crlavu_fd : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
  %crlato = stencil.load %crlato_fd : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
  %crlatu = stencil.load %crlatu_fd : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
  %acrlat0 = stencil.load %acrlat0_fd : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>

  // T_sqr_s
  %T_sqr_s = stencil.apply %arg1 = %uin, %arg2 = %vin, %arg3 = %acrlat0, %arg4 = %eddlat, %arg5 = %eddlon :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<j,f64>, f64, f64 {

      %one = constant 1.0 : f64
      %earth_radius = constant 6371.229e3 : f64
      %earth_radius_recip = divf %one, %earth_radius : f64

      %acrlat = stencil.access %arg3[0, 0, 0] : (!stencil.temp<j,f64>) -> f64

      %frac_1_dx = mulf %acrlat, %arg5 : f64
      %frac_1_dy = mulf %arg4, %earth_radius_recip: f64

      %v_jminus1 = stencil.access %arg2[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %v_center = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %v_-10 = subf %v_jminus1, %v_center : f64
      %T_sv = mulf %v_-10, %frac_1_dy : f64

      %u_iminus1 = stencil.access %arg1[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_center = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_-10 = subf %u_iminus1, %u_center : f64
      %T_su = mulf %u_-10, %frac_1_dx : f64

      %T_s = subf %T_sv, %T_su : f64
      %T_sqr_s = mulf %T_s, %T_s : f64

      stencil.return %T_sqr_s : f64

	} : !stencil.temp<ijk,f64>

  // S_sqr_uv
  %S_sqr_uv = stencil.apply %arg1 = %uin, %arg2 = %vin, %arg3 = %acrlat0, %arg4 = %eddlat, %arg5 = %eddlon :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<j,f64>, f64, f64 {

      %one = constant 1.0 : f64
      %earth_radius = constant 6371.229e3 : f64
      %earth_radius_recip = divf %one, %earth_radius : f64

      %acrlat = stencil.access %arg3[0, 0, 0] : (!stencil.temp<j,f64>) -> f64

      %frac_1_dx = mulf %acrlat, %arg5 : f64
      %frac_1_dy = mulf %arg4, %earth_radius_recip : f64

      %v_iplus1 = stencil.access %arg2[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %v_center = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %v_10 = subf %v_iplus1, %v_center : f64
      %S_v = mulf %v_10, %frac_1_dx : f64

      %u_jplus1 = stencil.access %arg1[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_center = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_10 = subf %u_jplus1, %u_center : f64
      %S_u = mulf %u_10, %frac_1_dy : f64

      %S_uv = addf %S_u, %S_v : f64
      %S_sqr_uv = mulf %S_uv, %S_uv : f64

      stencil.return %S_sqr_uv : f64

	} : !stencil.temp<ijk,f64>

  // lapu
  %lapu = stencil.apply %arg1 = %uin, %arg2 = %crlato, %arg3 = %crlatu: !stencil.temp<ijk,f64>, !stencil.temp<j,f64>, !stencil.temp<j,f64> {
      %0 = stencil.access %arg1[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg1[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %3 = addf %0, %1 : f64
      %cst = constant -2.0 : f64
      %4 = mulf %2, %cst : f64
      %5 = addf %3, %4 : f64
      %6 = stencil.access %arg1[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %7 = stencil.access %arg1[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %8 = stencil.access %arg2[0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %9 = stencil.access %arg3[0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %10 = subf %6, %2 : f64
      %11 = subf %7, %2 : f64
      %12 = mulf %10, %8 : f64
      %13 = mulf %11, %9 : f64
      %14 = addf %12, %5 : f64
      %15 = addf %14, %13 : f64
      stencil.return %15 : f64
	} : !stencil.temp<ijk,f64>

  // lapv
  %lapv = stencil.apply %arg1 = %vin, %arg2 = %crlavo, %arg3 = %crlavu: !stencil.temp<ijk,f64>, !stencil.temp<j,f64>, !stencil.temp<j,f64> {
      %0 = stencil.access %arg1[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg1[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %3 = addf %0, %1 : f64
      %cst = constant -2.0 : f64
      %4 = mulf %2, %cst : f64
      %5 = addf %3, %4 : f64
      %6 = stencil.access %arg1[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %7 = stencil.access %arg1[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %8 = stencil.access %arg2[0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %9 = stencil.access %arg3[0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %10 = subf %6, %2 : f64
      %11 = subf %7, %2 : f64
      %12 = mulf %10, %8 : f64
      %13 = mulf %11, %9 : f64
      %14 = addf %12, %5 : f64
      %15 = addf %14, %13 : f64
      stencil.return %15 : f64
	} : !stencil.temp<ijk,f64>

  // u_out
  %u_out = stencil.apply %arg1 = %uin, %arg2 = %T_sqr_s, %arg3 = %S_sqr_uv, %arg4 = %lapu, %arg5 = %mask, %arg6 = %weight_smag, %arg7 = %tau_smag :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, f64, f64 {
      %hdmaskvel = stencil.access %arg5[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %hdweight = mulf %arg6, %hdmaskvel : f64

      %half = constant 0.5 : f64
      %zero = constant 0.0 : f64

      %T_iplus1 = stencil.access %arg2[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %T_center = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %T_10 = addf %T_iplus1, %T_center : f64
      %T_avg = mulf %T_10, %half : f64

      %S_jminus1 = stencil.access %arg3[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %S_center = stencil.access %arg3[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %S_-10 = addf %S_jminus1, %S_center : f64
      %S_avg = mulf %S_-10, %half : f64

      %ST = addf %T_avg, %S_avg : f64
      %sqrt_ST = sqrt %ST : f64
      %smag_u_tmp = mulf %sqrt_ST, %arg7 : f64
      %smag_u = subf %smag_u_tmp, %hdweight : f64
      
      %gzero = cmpf "ogt", %smag_u, %zero : f64
      %smag_u_g0 = select %gzero, %smag_u, %zero : f64
      %lhalf = cmpf "olt", %smag_u_g0, %half : f64
      %smag_u_final = select %lhalf, %smag_u_g0, %half : f64

      %lap = stencil.access %arg4[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_in = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %smag_lap = mulf %smag_u_final, %lap : f64
      %u_out = addf %smag_lap, %u_in : f64

      stencil.return %u_out : f64
	} : !stencil.temp<ijk,f64>

  // v_out
  %v_out = stencil.apply %arg1 = %vin, %arg2 = %T_sqr_s, %arg3 = %S_sqr_uv, %arg4 = %lapv, %arg5 = %mask, %arg6 = %weight_smag, %arg7 = %tau_smag :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, f64, f64 {
      %hdmaskvel = stencil.access %arg5[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %hdweight = mulf %arg6, %hdmaskvel : f64

      %half = constant 0.5 : f64
      %zero = constant 0.0 : f64

      %T_jplus1 = stencil.access %arg2[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %T_center = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %T_10 = addf %T_jplus1, %T_center : f64
      %T_avg = mulf %T_10, %half : f64

      %S_iminus1 = stencil.access %arg3[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %S_center = stencil.access %arg3[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %S_-10 = addf %S_iminus1, %S_center : f64
      %S_avg = mulf %S_-10, %half : f64

      %ST = addf %T_avg, %S_avg : f64
      %sqrt_ST = sqrt %ST : f64
      %smag_v_tmp = mulf %sqrt_ST, %arg7 : f64
      %smag_v = subf %smag_v_tmp, %hdweight : f64
      
      %gzero = cmpf "ogt", %smag_v, %zero : f64
      %smag_v_g0 = select %gzero, %smag_v, %zero : f64
      %lhalf = cmpf "olt", %smag_v_g0, %half : f64
      %smag_v_final = select %lhalf, %smag_v_g0, %half : f64

      %lap = stencil.access %arg4[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %v_in = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %smag_lap = mulf %smag_v_final, %lap : f64
      %v_out = addf %smag_lap, %v_in : f64

      stencil.return %v_out : f64
	} : !stencil.temp<ijk,f64>

	stencil.store %u_out to %uout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
	stencil.store %v_out to %vout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  return
}
