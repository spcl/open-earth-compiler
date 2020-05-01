
func @hadvuv5th(
  %uin_fd : !stencil.field<ijk,f64>,
  %vin_fd : !stencil.field<ijk,f64>,
  %uout_fd : !stencil.field<ijk,f64>,
  %vout_fd : !stencil.field<ijk,f64>,
  %acrlat0_fd : !stencil.field<j,f64>,
  %acrlat1_fd : !stencil.field<j,f64>,
  %tgrlatda0_fd : !stencil.field<j,f64>,
  %tgrlatda1_fd : !stencil.field<j,f64>,
  %eddlat : f64,
  %eddlon : f64)
  attributes { stencil.program } {
  // asserts
  stencil.assert %uin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vin_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %uout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %vout_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %acrlat0_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  stencil.assert %acrlat1_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  stencil.assert %tgrlatda0_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  stencil.assert %tgrlatda1_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<j,f64>
  // loads
  %uin = stencil.load %uin_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %vin = stencil.load %vin_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  %acrlat0 = stencil.load %acrlat0_fd : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
  %acrlat1 = stencil.load %acrlat1_fd : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
  %tgrlatda0 = stencil.load %tgrlatda0_fd : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>
  %tgrlatda1 = stencil.load %tgrlatda1_fd : (!stencil.field<j,f64>) -> !stencil.temp<j,f64>

  // uatupos, uavgu
  %uatupos, %uavgu = stencil.apply %arg1 = %uin, %arg4 = %acrlat0 : !stencil.temp<ijk,f64>, !stencil.temp<j,f64> {
      %one = constant 1.0 : f64
      %three = constant 3.0 : f64
      %cst = divf %one, %three : f64
      %0 = stencil.access %arg1[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg1[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = stencil.access %arg1[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %3 = addf %0, %1 : f64
      %4 = addf %3, %2 : f64
      %5 = mulf %4, %cst : f64
      %6 = stencil.access %arg4[0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %7 = mulf %5, %6 : f64
      stencil.return %5, %7 : f64, f64
  } : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>
  // vatupos, vavgu
  %vatupos, %vavgu = stencil.apply %arg2 = %vin : !stencil.temp<ijk,f64> {
      %one = constant 1.0 : f64
      %earth_radius = constant 6371.229e3 : f64
      %earth_radius_recip = divf %one, %earth_radius : f64

      %cst = constant 0.25 : f64
      %0 = stencil.access %arg2[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg2[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = stencil.access %arg2[1, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %3 = stencil.access %arg2[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %4 = addf %0, %1 : f64
      %5 = addf %2, %3 : f64
      %6 = addf %4, %5 : f64
      %7 = mulf %6, %cst : f64
      %8 = mulf %7, %earth_radius_recip: f64
      stencil.return %7, %8 : f64, f64
  } : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>
  // udelta
  %udelta = stencil.apply %arg7 = %uin, %arg8 = %uavgu, %arg9 = %vavgu, %arg10 = %eddlat, %arg11 = %eddlon :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, f64, f64 {
      %zero = constant 0.0 : f64
      %one = constant 1.0 : f64
      %minus_one = constant -1.0 : f64
      %two = constant 2.0 : f64
      %three = constant 3.0 : f64
      %four = constant 4.0 : f64
      %twenty = constant 20.0 : f64
      %thirty = constant 30.0 : f64
      %by30 = divf %one, %thirty  : f64
      %-by4 = divf %minus_one, %four  : f64
      %-by3 = divf %minus_one, %three : f64
      %-by2 = divf %minus_one, %two : f64
      %by20 = divf %one, %twenty : f64
      %uavg = stencil.access %arg8[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_center = stencil.access %arg7[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_iminus2 = stencil.access %arg7[-2, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_iminus1 = stencil.access %arg7[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_iplus1 = stencil.access %arg7[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_iplus2 = stencil.access %arg7[2, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %uavg_g0 = cmpf "ogt", %uavg, %zero : f64
      %prod_0 = mulf %-by3, %u_center : f64
      %idir_res = loop.if %uavg_g0 -> (f64) {
          %u_iminus3 = stencil.access %arg7[-3, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
          %prod_-2 = mulf %-by4, %u_iminus2 : f64
          %sum_0-1 = addf %prod_0, %u_iminus1 : f64
          %prod_1 = mulf %-by2, %u_iplus1 : f64
          %prod_2 = mulf %by20, %u_iplus2 : f64
          %prod_-3 = mulf %by30, %u_iminus3 : f64
          %sum_-21 = addf %prod_-2, %prod_1 : f64
          %sum_2-3 = addf %prod_2, %prod_-3 : f64
          %sum_-2-101 = addf %sum_0-1, %sum_-21 : f64
          %sum_full = addf %sum_2-3, %sum_-2-101 : f64
          %ipos_res = mulf %sum_full, %uavg : f64
          loop.yield %ipos_res : f64
      } else {
          %u_iplus3 = stencil.access %arg7[3, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
          %prod_-2 = mulf %by20, %u_iminus2 : f64
          %prod_-1 = mulf %-by2, %u_iminus1 : f64
          %sum_01 = addf %prod_0, %u_iplus1 : f64
          %prod_2 = mulf %-by4, %u_iplus2 : f64
          %prod_3 = mulf %by30, %u_iplus3 : f64
          %sum_-2-1 = addf %prod_-2, %prod_-1 : f64
          %sum_23 = addf %prod_2, %prod_3 : f64
          %sum_-2-101 = addf %sum_01, %sum_-2-1 : f64
          %sum_full = addf %sum_23, %sum_-2-101 : f64
          %-uavg = negf %uavg : f64
          %ineg_res = mulf %-uavg, %sum_full : f64
          loop.yield %ineg_res : f64
      }
      %vavg = stencil.access %arg9[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_jminus2 = stencil.access %arg7[0, -2, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_jminus1 = stencil.access %arg7[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_jplus1 = stencil.access %arg7[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_jplus2 = stencil.access %arg7[0, 2, 0] : (!stencil.temp<ijk,f64>) -> f64
      %vavg_g0 = cmpf "ogt", %vavg, %zero : f64
      %jdir_res = loop.if %vavg_g0 -> (f64) {
          %u_jminus3 = stencil.access %arg7[0, -3, 0] : (!stencil.temp<ijk,f64>) -> f64
          %prod_-2 = mulf %-by4, %u_jminus2 : f64
          %sum_0-1 = addf %prod_0, %u_jminus1 : f64
          %prod_1 = mulf %-by2, %u_jplus1 : f64
          %prod_2 = mulf %by20, %u_jplus2 : f64
          %prod_-3 = mulf %by30, %u_jminus3 : f64
          %sum_-21 = addf %prod_-2, %prod_1 : f64
          %sum_2-3 = addf %prod_2, %prod_-3 : f64
          %sum_-2-101 = addf %sum_0-1, %sum_-21 : f64
          %sum_full = addf %sum_2-3, %sum_-2-101 : f64
          %jpos_res = mulf %sum_full, %vavg : f64
          loop.yield %jpos_res : f64
      } else {
          %u_jplus3 = stencil.access %arg7[0, 3, 0] : (!stencil.temp<ijk,f64>) -> f64
          %prod_-2 = mulf %by20, %u_jminus2 : f64
          %prod_-1 = mulf %-by2, %u_jminus1 : f64
          %sum_01 = addf %prod_0, %u_jplus1 : f64
          %prod_2 = mulf %-by4, %u_jplus2 : f64
          %prod_3 = mulf %by30, %u_jplus3 : f64
          %sum_-2-1 = addf %prod_-2, %prod_-1 : f64
          %sum_23 = addf %prod_2, %prod_3 : f64
          %sum_-2-101 = addf %sum_01, %sum_-2-1 : f64
          %sum_full = addf %sum_23, %sum_-2-101 : f64
          %-vavg = negf %vavg : f64
          %jneg_res = mulf %-vavg, %sum_full : f64
          loop.yield %jneg_res : f64
      }
      %tmp_i = mulf %idir_res, %arg10 : f64
      %tmp_j = mulf %jdir_res, %arg11 : f64
      %res = addf %tmp_i, %tmp_j : f64
      stencil.return %res : f64
  } : !stencil.temp<ijk,f64>
  // uout
  %uout = stencil.apply %arg12 = %udelta, %arg13 = %uin, %arg14 = %vatupos, %arg15 = %tgrlatda0 :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<j,f64> {
      %0 = stencil.access %arg13[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg14[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = stencil.access %arg15[0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %3 = stencil.access %arg12[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %4 = mulf %0, %1 : f64
      %5 = mulf %4, %2 : f64
      %6 = addf %5, %3 : f64
      stencil.return %6 : f64
  } : !stencil.temp<ijk,f64>
  // uatvpos, uavgv
  %uatvpos, %uavgv = stencil.apply %arg16 = %uin, %arg19 = %acrlat1 : !stencil.temp<ijk,f64>, !stencil.temp<j,f64> {
      %cst = constant 0.25 : f64
      %0 = stencil.access %arg16[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg16[-1, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = stencil.access %arg16[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %3 = stencil.access %arg16[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %4 = addf %0, %1 : f64
      %5 = addf %2, %3 : f64
      %6 = addf %4, %5 : f64
      %7 = mulf %6, %cst : f64
      %8 = stencil.access %arg19[0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %9 = mulf %7, %8 : f64
      stencil.return %7, %9 : f64, f64
  } : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>
  // vatvpos, vavgv
  %vatvpos, %vavgv = stencil.apply %arg17 = %vin : !stencil.temp<ijk,f64> {
      %one = constant 1.0 : f64
      %earth_radius = constant 6371.229e3 : f64
      %earth_radius_recip = divf %one, %earth_radius : f64
      %three = constant 3.0 : f64
      %cst = divf %one, %three : f64
      %0 = stencil.access %arg17[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg17[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %2 = stencil.access %arg17[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %3 = addf %0, %1 : f64
      %4 = addf %3, %2 : f64
      %5 = mulf %4, %cst : f64
      %6 = mulf %5, %earth_radius_recip : f64
      stencil.return %5, %6 : f64, f64
  } : !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>
  // vdelta
  %vdelta = stencil.apply %arg22 = %vin, %arg23 = %uavgv, %arg24 = %vavgv, %arg25 = %eddlat, %arg26 = %eddlon :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, f64, f64 {
      %zero = constant 0.0 : f64
      %one = constant 1.0 : f64
      %minus_one = constant -1.0 : f64
      %two = constant 2.0 : f64
      %three = constant 3.0 : f64
      %four = constant 4.0 : f64
      %twenty = constant 20.0 : f64
      %thirty = constant 30.0 : f64
      %by30 = divf %one, %thirty  : f64
      %-by4 = divf %minus_one, %four  : f64
      %-by3 = divf %minus_one, %three : f64
      %-by2 = divf %minus_one, %two : f64
      %by20 = divf %one, %twenty : f64
      %uavg = stencil.access %arg23[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_center = stencil.access %arg22[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_iminus2 = stencil.access %arg22[-2, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_iminus1 = stencil.access %arg22[-1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_iplus1 = stencil.access %arg22[1, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_iplus2 = stencil.access %arg22[2, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %uavg_g0 = cmpf "ogt", %uavg, %zero : f64
      %prod_0 = mulf %-by3, %u_center : f64
      %idir_res = loop.if %uavg_g0 -> (f64) {
          %u_iminus3 = stencil.access %arg22[-3, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
          %prod_-2 = mulf %-by4, %u_iminus2 : f64
          %sum_0-1 = addf %prod_0, %u_iminus1 : f64
          %prod_1 = mulf %-by2, %u_iplus1 : f64
          %prod_2 = mulf %by20, %u_iplus2 : f64
          %prod_-3 = mulf %by30, %u_iminus3 : f64
          %sum_-21 = addf %prod_-2, %prod_1 : f64
          %sum_2-3 = addf %prod_2, %prod_-3 : f64
          %sum_-2-101 = addf %sum_0-1, %sum_-21 : f64
          %sum_full = addf %sum_2-3, %sum_-2-101 : f64
          %ipos_res = mulf %sum_full, %uavg : f64
          loop.yield %ipos_res : f64
      } else {
          %u_iplus3 = stencil.access %arg22[3, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
          %prod_-2 = mulf %by20, %u_iminus2 : f64
          %prod_-1 = mulf %-by2, %u_iminus1 : f64
          %sum_01 = addf %prod_0, %u_iplus1 : f64
          %prod_2 = mulf %-by4, %u_iplus2 : f64
          %prod_3 = mulf %by30, %u_iplus3 : f64
          %sum_-2-1 = addf %prod_-2, %prod_-1 : f64
          %sum_23 = addf %prod_2, %prod_3 : f64
          %sum_-2-101 = addf %sum_01, %sum_-2-1 : f64
          %sum_full = addf %sum_23, %sum_-2-101 : f64
          %-uavg = negf %uavg : f64
          %ineg_res = mulf %-uavg, %sum_full : f64
          loop.yield %ineg_res : f64
      }
      %vavg = stencil.access %arg24[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_jminus2 = stencil.access %arg22[0, -2, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_jminus1 = stencil.access %arg22[0, -1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_jplus1 = stencil.access %arg22[0, 1, 0] : (!stencil.temp<ijk,f64>) -> f64
      %u_jplus2 = stencil.access %arg22[0, 2, 0] : (!stencil.temp<ijk,f64>) -> f64
      %vavg_g0 = cmpf "ogt", %vavg, %zero : f64
      %jdir_res = loop.if %vavg_g0 -> (f64) {
          %u_jminus3 = stencil.access %arg22[0, -3, 0] : (!stencil.temp<ijk,f64>) -> f64
          %sum_0-1 = addf %prod_0, %u_jminus1 : f64
          %prod_-2 = mulf %-by4, %u_jminus2 : f64
          %prod_1 = mulf %-by2, %u_jplus1 : f64
          %prod_2 = mulf %by20, %u_jplus2 : f64
          %prod_-3 = mulf %by30, %u_jminus3 : f64
          %sum_-21 = addf %prod_-2, %prod_1 : f64
          %sum_2-3 = addf %prod_2, %prod_-3 : f64
          %sum_-2-101 = addf %sum_0-1, %sum_-21 : f64
          %sum_full = addf %sum_2-3, %sum_-2-101 : f64
          %jpos_res = mulf %sum_full, %vavg : f64
          loop.yield %jpos_res : f64
      } else {
          %u_jplus3 = stencil.access %arg22[0, 3, 0] : (!stencil.temp<ijk,f64>) -> f64
          %sum_01 = addf %prod_0, %u_jplus1 : f64
          %prod_-2 = mulf %by20, %u_jminus2 : f64
          %prod_-1 = mulf %-by2, %u_jminus1 : f64
          %prod_2 = mulf %-by4, %u_jplus2 : f64
          %prod_3 = mulf %by30, %u_jplus3 : f64
          %sum_-2-1 = addf %prod_-2, %prod_-1 : f64
          %sum_23 = addf %prod_2, %prod_3 : f64
          %sum_-2-101 = addf %sum_01, %sum_-2-1 : f64
          %sum_full = addf %sum_23, %sum_-2-101 : f64
          %-vavg = negf %vavg : f64
          %jneg_res = mulf %-vavg, %sum_full : f64
          loop.yield %jneg_res : f64
      }
      %tmp_i = mulf %idir_res, %arg25 : f64
      %tmp_j = mulf %jdir_res, %arg26 : f64
      %res = addf %tmp_i, %tmp_j : f64
      stencil.return %res : f64
  } : !stencil.temp<ijk,f64>
  // vout
  %vout = stencil.apply %arg27 = %vdelta, %arg28 = %uatvpos, %arg29 = %tgrlatda1 :
  !stencil.temp<ijk,f64>, !stencil.temp<ijk,f64>, !stencil.temp<j,f64> {
      %0 = stencil.access %arg28[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %1 = stencil.access %arg29[0, 0, 0] : (!stencil.temp<j,f64>) -> f64
      %2 = stencil.access %arg27[0, 0, 0] : (!stencil.temp<ijk,f64>) -> f64
      %3 = mulf %0, %0 : f64
      %4 = mulf %3, %1 : f64
      %5 = subf %2, %4 : f64
      stencil.return %5 : f64
  } : !stencil.temp<ijk,f64>
  // store results
  stencil.store %uout to %uout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  stencil.store %vout to %vout_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  return
}
