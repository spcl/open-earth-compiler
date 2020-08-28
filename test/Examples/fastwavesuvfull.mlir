module {
  func @fastwavesuvfull(
    %uin_fd: !stencil.field<?x?x?xf64>,
    %utens_fd: !stencil.field<?x?x?xf64>,
    %vin_fd: !stencil.field<?x?x?xf64>,
    %vtens_fd: !stencil.field<?x?x?xf64>,
    %upos_fd: !stencil.field<?x?x?xf64>,
    %vpos_fd: !stencil.field<?x?x?xf64>,
    %wgtfac_fd: !stencil.field<?x?x?xf64>,
    %ppuv_fd: !stencil.field<?x?x?xf64>,
    %hhl_fd: !stencil.field<?x?x?xf64>,
    %rho_fd: !stencil.field<?x?x?xf64>,
    %rho0_fd: !stencil.field<?x?x?xf64>,
    %p0_fd: !stencil.field<?x?x?xf64>,
    %uout_fd: !stencil.field<?x?x?xf64>,
    %vout_fd: !stencil.field<?x?x?xf64>,
    %fx_fd: !stencil.field<0x?x0xf64>,
    %xlhsx_fd: !stencil.field<?x?x0xf64>,
    %xlhsy_fd: !stencil.field<?x?x0xf64>,
    %xdzdx_fd: !stencil.field<?x?x0xf64>,
    %xdzdy_fd: !stencil.field<?x?x0xf64>,
    %cwp_fd: !stencil.field<?x?x0xf64>,
    %wbbctens_fd: !stencil.field<?x?x0xf64>)
    attributes {stencil.program} {
    stencil.assert %uin_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %utens_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %vin_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %vtens_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %upos_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %vpos_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %wgtfac_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %ppuv_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %hhl_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %rho_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %rho0_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %p0_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %uout_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %vout_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %fx_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<0x?x0xf64>
    stencil.assert %xlhsx_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x0xf64>
    stencil.assert %xlhsy_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x0xf64>
    stencil.assert %xdzdx_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x0xf64>
    stencil.assert %xdzdy_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x0xf64>
    stencil.assert %cwp_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x0xf64>
    stencil.assert %wbbctens_fd([-4, -4, -4] : [68, 68, 68]) : !stencil.field<?x?x0xf64>
    %uin = stencil.load %uin_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %utens = stencil.load %utens_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %vin = stencil.load %vin_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %vtens = stencil.load %vtens_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %upos = stencil.load %upos_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %vpos = stencil.load %vpos_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %wgtfac = stencil.load %wgtfac_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %ppuv = stencil.load %ppuv_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %hhl = stencil.load %hhl_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %rho = stencil.load %rho_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %rho0 = stencil.load %rho0_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %p0 = stencil.load %p0_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %fx = stencil.load %fx_fd : (!stencil.field<0x?x0xf64>) -> !stencil.temp<0x?x0xf64>
    %xlhsx = stencil.load %xlhsx_fd : (!stencil.field<?x?x0xf64>) -> !stencil.temp<?x?x0xf64>
    %xlhsy = stencil.load %xlhsy_fd : (!stencil.field<?x?x0xf64>) -> !stencil.temp<?x?x0xf64>
    %xdzdx = stencil.load %xdzdx_fd : (!stencil.field<?x?x0xf64>) -> !stencil.temp<?x?x0xf64>
    %xdzdy = stencil.load %xdzdy_fd : (!stencil.field<?x?x0xf64>) -> !stencil.temp<?x?x0xf64>
    %cwp = stencil.load %cwp_fd : (!stencil.field<?x?x0xf64>) -> !stencil.temp<?x?x0xf64>
    %wbbctens = stencil.load %wbbctens_fd : (!stencil.field<?x?x0xf64>) -> !stencil.temp<?x?x0xf64>

    %ppgk = stencil.apply (%arg13 = %wgtfac : !stencil.temp<?x?x?xf64>, %arg14 = %ppuv : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = mulf %15, %16 : f64
      %one = constant 1.000000e+00 : f64
      %18 = subf %one, %15 : f64
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

    %ppgu_terrain = stencil.apply (%arg13 = %ppuv : !stencil.temp<?x?x?xf64>, %arg14 = %ppgc : !stencil.temp<?x?x?xf64>, %arg15 = %hhl : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = subf %15, %16 : f64
      %half = constant 5.000000e-01 : f64
      %18 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = addf %18, %19 : f64
      %21 = mulf %half, %20 : f64
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

    %ppgv_terrain = stencil.apply (%arg13 = %ppuv : !stencil.temp<?x?x?xf64>, %arg14 = %ppgc : !stencil.temp<?x?x?xf64>, %arg15 = %hhl : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = subf %15, %16 : f64
      %half = constant 5.000000e-01 : f64
      %18 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %19 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %20 = addf %18, %19 : f64
      %21 = mulf %half, %20 : f64
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

    %ppgu_free = stencil.apply (%arg50 = %ppuv : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg50 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg50 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = subf %15, %16 : f64
      stencil.return %17 : f64
    }

    %ppgv_free = stencil.apply (%arg50 = %ppuv : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %15 = stencil.access %arg50 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg50 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = subf %15, %16 : f64
      stencil.return %17 : f64
    }

    %xrhsx = stencil.apply (%arg13 = %rho : !stencil.temp<?x?x?xf64>, %arg14 = %ppuv : !stencil.temp<?x?x?xf64>, %arg15 = %utens : !stencil.temp<?x?x?xf64>, %arg16 = %fx : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %half = constant 0.500000e+00 : f64
      %15 = stencil.access %arg13 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = addf %15, %16 : f64
      %18 = mulf %17, %half : f64
      %19 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %20 = negf %19 : f64
      %21 = divf %20, %18 : f64
      %22 = stencil.access %arg14 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = subf %22, %23 : f64
      %25 = mulf %21, %24 : f64
      %26 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %25, %26 : f64
      stencil.return %27 : f64
    }

    %xrhsy = stencil.apply (%arg13 = %rho : !stencil.temp<?x?x?xf64>, %arg14 = %ppuv : !stencil.temp<?x?x?xf64>, %arg15 = %vtens : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %edadlat = constant 0.00048828125 : f64
      %half = constant 0.500000e+00 : f64
      %15 = stencil.access %arg13 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %16 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %17 = addf %15, %16 : f64
      %18 = mulf %17, %half : f64
      %20 = negf %edadlat : f64
      %21 = divf %20, %18 : f64
      %22 = stencil.access %arg14 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = subf %22, %23 : f64
      %25 = mulf %21, %24 : f64
      %26 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = addf %25, %26 : f64
      stencil.return %27 : f64
    }

    %xrhsz = stencil.apply (%arg13 = %rho : !stencil.temp<?x?x?xf64>, %arg14 = %rho0 : !stencil.temp<?x?x?xf64>, %arg15 = %p0 : !stencil.temp<?x?x?xf64>, %arg16 = %ppuv : !stencil.temp<?x?x?xf64>, %arg17 = %cwp : !stencil.temp<?x?x0xf64>, %arg18 = %wbbctens : !stencil.temp<?x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %20 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %21 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %22 = divf %20, %21 : f64
      %gravity = constant 9.806650e+00 : f64
      %23 = mulf %22, %gravity : f64
      %24 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = addf %24, %25 : f64
      %27 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %28 = mulf %26, %27 : f64
      %one = constant 1.000000e+00 : f64
      %29 = subf %one, %28 : f64
      %30 = mulf %23, %29 : f64
      %31 = stencil.access %arg18 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
      %32 = addf %30, %31 : f64
      stencil.return %32 : f64
    }

    %uout = stencil.apply (%arg13 = %uin : !stencil.temp<?x?x?xf64>, %arg14 = %utens : !stencil.temp<?x?x?xf64>, %arg15 = %ppgu_terrain : !stencil.temp<?x?x?xf64>, %arg16 = %rho : !stencil.temp<?x?x?xf64>, %arg17 = %fx : !stencil.temp<0x?x0xf64>, %arg19 = %ppgu_free : !stencil.temp<?x?x?xf64>, %arg20 = %xlhsx : !stencil.temp<?x?x0xf64>, %arg22 = %xdzdx : !stencil.temp<?x?x0xf64>, %arg23 = %xdzdy : !stencil.temp<?x?x0xf64>, %arg24 = %xrhsx : !stencil.temp<?x?x?xf64>, %arg25 = %xrhsy : !stencil.temp<?x?x?xf64>, %arg26 = %xrhsz : !stencil.temp<?x?x?xf64>, %arg27 = %upos : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {

      %dt = constant 10.0 : f64
      %two = constant 2.000000e+00 : f64
      %half = constant 0.500000e+00 : f64
      %zero = constant 0.000000e+00 : f64
      %cFlatLimit = constant 11 : index
      %ckMax = constant 59 : index

      %k = stencil.index 2 [0, 0, 0] : index
      %max = cmpi "eq", %k, %ckMax : index

      %res = scf.if %max -> (f64) {
        %101 = stencil.access %arg23 [1, -1, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %102 = stencil.access %arg23 [1,  0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %103 = addf %101, %102 : f64
        %104 = mulf %half, %103 : f64
        %105 = stencil.access %arg23 [0, -1, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %106 = stencil.access %arg23 [0,  0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %107 = addf %105, %106 : f64
        %108 = mulf %half, %107 : f64
        %109 = addf %104, %108 : f64
        %110 = mulf %half, %109 : f64

        %111 = stencil.access %arg25 [1, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %112 = stencil.access %arg25 [1,  0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %113 = addf %111, %112 : f64
        %114 = mulf %half, %113 : f64
        %115 = stencil.access %arg25 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %116 = stencil.access %arg25 [0,  0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %117 = addf %115, %116 : f64
        %118 = mulf %half, %117 : f64
        %119 = addf %114, %118 : f64
        %120 = mulf %half, %119 : f64

        %121 = mulf %110, %120 : f64

        %122 = stencil.access %arg22 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %123 = stencil.access %arg24 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %124 = mulf %122, %123 : f64

        %125 = stencil.access %arg26 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %126 = stencil.access %arg26 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %127 = addf %125, %126 : f64
        %128 = mulf %half, %127 : f64

        %129 = subf %128, %124 : f64
        %130 = subf %129, %121 : f64

        %131 = stencil.access %arg20 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %132 = mulf %131, %122 : f64
        %133 = mulf %130, %132 : f64
        %134 = addf %133, %123 : f64

        %135 = mulf %dt, %134 : f64
        %136 = stencil.access %arg27 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %137 = addf %136, %135 : f64

        scf.yield %137 : f64
      } else {
        %above = cmpi "slt", %k, %cFlatLimit : index

        %15 = scf.if %above -> (f64) {
          %30 = stencil.access %arg19 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
          scf.yield %30 : f64
        } else {
          %31 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
          scf.yield %31 : f64
        }

        %16 = stencil.access %arg16 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %17 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %18 = stencil.access %arg17 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
        %19 = addf %16, %17 : f64
        %20 = mulf %two, %18 : f64
        %21 = divf %20, %19 : f64
        %22 = mulf %15, %21 : f64
        %23 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %24 = subf %23, %22 : f64
        %25 = mulf %dt, %24 : f64
        %26 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %27 = addf %25, %26 : f64

        scf.yield %27 : f64
      }

      stencil.return %res : f64
    }

    %vout = stencil.apply (%arg13 = %vin : !stencil.temp<?x?x?xf64>, %arg14 = %vtens : !stencil.temp<?x?x?xf64>, %arg15 = %ppgv_terrain : !stencil.temp<?x?x?xf64>, %arg16 = %rho : !stencil.temp<?x?x?xf64>, %arg19 = %ppgv_free : !stencil.temp<?x?x?xf64>, %arg21 = %xlhsy : !stencil.temp<?x?x0xf64>, %arg22 = %xdzdx : !stencil.temp<?x?x0xf64>, %arg23 = %xdzdy : !stencil.temp<?x?x0xf64>, %arg24 = %xrhsx : !stencil.temp<?x?x?xf64>, %arg25 = %xrhsy : !stencil.temp<?x?x?xf64>, %arg26 = %xrhsz : !stencil.temp<?x?x?xf64>, %arg27 = %vpos : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {

      %dt = constant 10.0 : f64
      %edadlat = constant 0.00048828125 : f64

      %two = constant 2.000000e+00 : f64
      %half = constant 0.500000e+00 : f64
      %zero = constant 0.000000e+00 : f64
      %cFlatLimit = constant 11 : index
      %ckMax = constant 59 : index

      %k = stencil.index 2 [0, 0, 0] : index
      %max = cmpi "eq", %k, %ckMax : index

      %res = scf.if %max -> (f64) {
        %101 = stencil.access %arg22 [-1, 1, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %102 = stencil.access %arg22 [0,  1, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %103 = addf %101, %102 : f64
        %104 = mulf %half, %103 : f64
        %105 = stencil.access %arg22 [-1, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %106 = stencil.access %arg22 [0,  0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %107 = addf %105, %106 : f64
        %108 = mulf %half, %107 : f64
        %109 = addf %104, %108 : f64
        %110 = mulf %half, %109 : f64

        %111 = stencil.access %arg24 [-1, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %112 = stencil.access %arg24 [0,  1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %113 = addf %111, %112 : f64
        %114 = mulf %half, %113 : f64
        %115 = stencil.access %arg24 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %116 = stencil.access %arg24 [0,  0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %117 = addf %115, %116 : f64
        %118 = mulf %half, %117 : f64
        %119 = addf %114, %118 : f64
        %120 = mulf %half, %119 : f64

        %121 = mulf %110, %120 : f64

        %122 = stencil.access %arg23 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %123 = stencil.access %arg25 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %124 = mulf %122, %123 : f64

        %125 = stencil.access %arg26 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %126 = stencil.access %arg26 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %127 = addf %125, %126 : f64
        %128 = mulf %half, %127 : f64

        %129 = subf %128, %124 : f64
        %130 = subf %129, %121 : f64

        %131 = stencil.access %arg21 [0, 0, 0] : (!stencil.temp<?x?x0xf64>) -> f64
        %132 = mulf %131, %122 : f64
        %133 = mulf %130, %132 : f64
        %134 = addf %133, %123 : f64

        %135 = mulf %dt, %134 : f64
        %136 = stencil.access %arg27 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %137 = addf %136, %135 : f64

        scf.yield %137 : f64
      } else {
        %above = cmpi "slt", %k, %cFlatLimit : index

        %15 = scf.if %above -> (f64) {
          %30 = stencil.access %arg19 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
          scf.yield %30 : f64
        } else {
          %31 = stencil.access %arg15 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
          scf.yield %31 : f64
        }

        %16 = stencil.access %arg16 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %17 = stencil.access %arg16 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %18 = addf %16, %17 : f64
        %19 = mulf %two, %edadlat : f64
        %20 = divf %19, %18 : f64
        %21 = mulf %15, %20 : f64
        %22 = stencil.access %arg14 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %23 = subf %22, %21 : f64
        %24 = mulf %dt, %23 : f64
        %25 = stencil.access %arg13 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
        %26 = addf %24, %25 : f64

        scf.yield %26 : f64
      }

      stencil.return %res : f64
    }

    stencil.store %uout to %uout_fd([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    stencil.store %vout to %vout_fd([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
  }
}
