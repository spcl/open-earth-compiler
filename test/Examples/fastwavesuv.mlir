module {
  func @fastwavesuv(
    %uin_fd: !stencil.field<?x?x?xf64>,
    %utens_fd: !stencil.field<?x?x?xf64>,
    %vin_fd: !stencil.field<?x?x?xf64>,
    %vtens_fd: !stencil.field<?x?x?xf64>,
    %wgtfac_fd: !stencil.field<?x?x?xf64>,
    %ppuv_fd: !stencil.field<?x?x?xf64>,
    %hhl_fd: !stencil.field<?x?x?xf64>,
    %rho_fd: !stencil.field<?x?x?xf64>,
    %uout_fd: !stencil.field<?x?x?xf64>,
    %vout_fd: !stencil.field<?x?x?xf64>,
    %fx_fd: !stencil.field<0x?x0xf64>)
    attributes {stencil.program} {
    stencil.assert %uin_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %utens_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %vin_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %vtens_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %wgtfac_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %ppuv_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %hhl_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %rho_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %uout_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %vout_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %fx_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    %uin = stencil.load %uin_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %utens = stencil.load %utens_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %vin = stencil.load %vin_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %vtens = stencil.load %vtens_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %wgtfac = stencil.load %wgtfac_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %ppuv = stencil.load %ppuv_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %hhl = stencil.load %hhl_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %rho = stencil.load %rho_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %fx = stencil.load %fx_fd : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>

    %ppgk = stencil.apply (%arg13 = %wgtfac : !stencil.temp<?x?x?xf64>, %arg14 = %ppuv : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %15, %16 : f64
      %cst = constant 1.000000e+00 : f64
      %18 = subf %cst, %15 : f64
      %19 = stencil.access %arg14 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = mulf %19, %18 : f64
      %21 = addf %17, %20 : f64
      stencil.return %21 : f64
    }

    %ppgc = stencil.apply (%arg13 = %ppgk : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = subf %15, %16 : f64
      stencil.return %17 : f64
    }

    %ppgu = stencil.apply (%arg13 = %ppuv : !stencil.temp<?x?x?xf64>, %arg14 = %ppgc : !stencil.temp<?x?x?xf64>, %arg15 = %hhl : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = subf %15, %16 : f64
      %cst = constant 5.000000e-01 : f64
      %18 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = addf %18, %19 : f64
      %21 = mulf %cst, %20 : f64
      %22 = stencil.access %arg15 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg15 [1, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg15 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = addf %22, %23 : f64
      %27 = addf %24, %25 : f64
      %28 = subf %26, %27 : f64
      %29 = subf %22, %23 : f64
      %30 = subf %24, %25 : f64
      %31 = addf %29, %30 : f64
      %32 = divf %28, %31 : f64
      %33 = mulf %21, %32 : f64
      %34 = addf %17, %33 : f64
      stencil.return %34 : f64
    }
    %ppgv = stencil.apply (%arg13 = %ppuv : !stencil.temp<?x?x?xf64>, %arg14 = %ppgc : !stencil.temp<?x?x?xf64>, %arg15 = %hhl : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = subf %15, %16 : f64
      %cst = constant 5.000000e-01 : f64
      %18 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = addf %18, %19 : f64
      %21 = mulf %cst, %20 : f64
      %22 = stencil.access %arg15 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg15 [0, 1, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg15 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = addf %22, %23 : f64
      %27 = addf %24, %25 : f64
      %28 = subf %26, %27 : f64
      %29 = subf %22, %23 : f64
      %30 = subf %24, %25 : f64
      %31 = addf %29, %30 : f64
      %32 = divf %28, %31 : f64
      %33 = mulf %21, %32 : f64
      %34 = addf %17, %33 : f64
      stencil.return %34 : f64
    }
    %uout = stencil.apply (%arg13 = %uin : !stencil.temp<?x?x?xf64>, %arg14 = %utens : !stencil.temp<?x?x?xf64>, %arg15 = %ppgu : !stencil.temp<?x?x?xf64>, %arg16 = %rho : !stencil.temp<?x?x?xf64>, %arg17 = %fx : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %dt = constant 10.0 : f64
      %cst = constant 2.000000e+00 : f64
      %15 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg16 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %19 = addf %16, %17 : f64
      %20 = mulf %cst, %18 : f64
      %21 = divf %20, %19 : f64
      %22 = mulf %15, %21 : f64
      %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = subf %23, %22 : f64
      %25 = mulf %dt, %24 : f64
      %26 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %25, %26 : f64
      stencil.return %27 : f64
    }
    %vout = stencil.apply (%arg13 = %vin : !stencil.temp<?x?x?xf64>, %arg14 = %vtens : !stencil.temp<?x?x?xf64>, %arg15 = %ppgv : !stencil.temp<?x?x?xf64>, %arg16 = %rho : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %dt = constant 10.0 : f64
      %edadlat = constant 0.00048828125 : f64
      %cst = constant 2.000000e+00 : f64
      %15 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg16 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %18 = addf %16, %17 : f64
      %19 = mulf %cst, %edadlat : f64
      %20 = divf %19, %18 : f64
      %21 = mulf %15, %20 : f64
      %22 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = subf %22, %21 : f64
      %24 = mulf %dt, %23 : f64
      %25 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = addf %24, %25 : f64
      stencil.return %26 : f64
    }
    stencil.store %uout to %uout_fd([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %vout to %vout_fd([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
