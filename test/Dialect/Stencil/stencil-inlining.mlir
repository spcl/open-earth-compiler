// RUN: oec-opt %s --stencil-inlining | oec-opt | FileCheck %s

func @simple(%in : !stencil.field<ijk,f64>, %out : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	stencil.assert %in ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %out ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  %0 = stencil.load %in : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %1 = stencil.apply %arg1 = %0 : !stencil.view<ijk,f64> {  
      %2 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = addf %2, %3 : f64
      stencil.return %4 : f64
	} : !stencil.view<ijk,f64>
  %5 = stencil.apply %arg2 = %0, %arg3 = %1 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
      %6 = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %7 = stencil.access %arg3[1, 2, 3] : (!stencil.view<ijk,f64>) -> f64
      %8 = addf %6, %7 : f64
      stencil.return %8 : f64
	} : !stencil.view<ijk,f64>
	stencil.store %5 to %out ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}

// CHECK-LABEL: func @simple(%{{.*}}: !stencil.field<ijk,f64>, %{{.*}}: !stencil.field<ijk,f64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}} ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
//  CHECK-NEXT: stencil.assert %{{.*}} ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
//  CHECK-NEXT: %{{.*}} = stencil.apply %{{.*}} = %{{.*}} : !stencil.view<ijk,f64> {
//       CHECK: %{{.*}} = stencil.access %{{.*}}[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
//       CHECK: %{{.*}} = stencil.access %{{.*}}[0, 2, 3] : (!stencil.view<ijk,f64>) -> f64
//       CHECK: %{{.*}} = stencil.access %{{.*}}[2, 2, 3] : (!stencil.view<ijk,f64>) -> f64

func @multiple_edges(%in1 : !stencil.field<ijk,f64>, %in2 : !stencil.field<ijk,f64>, %out : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	stencil.assert %in1 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
	stencil.assert %in2 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %out ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  %0 = stencil.load %in1 : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %1,%2 = stencil.apply %arg1 = %0 : !stencil.view<ijk,f64> {  
      %3 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      stencil.return %3, %4 : f64, f64
	} : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>
  %5 = stencil.load %in2 : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %6 = stencil.apply %arg2 = %0, %arg3 = %1, %arg4 = %2, %arg5 = %5 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
      %7 = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %8 = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %9 = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %10 = stencil.access %arg5[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %11 = addf %7, %8 : f64
      %12 = addf %9, %10 : f64
      %13 = addf %11, %12 : f64
      stencil.return %13 : f64
	} : !stencil.view<ijk,f64>
	stencil.store %6 to %out ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}

// CHECK-LABEL: func @multiple_edges(%{{.*}}: !stencil.field<ijk,f64>, %{{.*}}: !stencil.field<ijk,f64>, %{{.*}}: !stencil.field<ijk,f64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}} ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
//  CHECK-NEXT: stencil.assert %{{.*}} ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
//  CHECK-NEXT: stencil.assert %{{.*}} ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
//  CHECK-NEXT: %{{.*}} = stencil.apply %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}} : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
//       CHECK: %{{.*}} = stencil.access %{{.*}}[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
//       CHECK: %{{.*}} = stencil.access %{{.*}}[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
//       CHECK: %{{.*}} = stencil.access %{{.*}}[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
//       CHECK: %{{.*}} = stencil.access %{{.*}}[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64

func @avoid_redundant(%in : !stencil.field<ijk,f64>, %out : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	stencil.assert %in ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  stencil.assert %out ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  %0 = stencil.load %in : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %1 = stencil.apply %arg1 = %0 : !stencil.view<ijk,f64> {  
      %2 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %3 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %4 = addf %2, %3 : f64
      stencil.return %4 : f64
	} : !stencil.view<ijk,f64>
  %5 = stencil.apply %arg2 = %1 : !stencil.view<ijk,f64> {  
      %6 = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %7 = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %8 = addf %6, %7 : f64
      stencil.return %8 : f64
	} : !stencil.view<ijk,f64>
	stencil.store %5 to %out ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
}

// CHECK-LABEL: func @avoid_redundant(%{{.*}}: !stencil.field<ijk,f64>, %{{.*}}: !stencil.field<ijk,f64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}} ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
//  CHECK-NEXT: stencil.assert %{{.*}} ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
//  CHECK-NEXT: %{{.*}} = stencil.apply %{{.*}} = %{{.*}} : !stencil.view<ijk,f64> {
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
//  CHECK-NEXT: %{{.*}} = addf %{{.*}}, %{{.*}} : f64
//  CHECK-NEXT: %{{.*}} = addf %{{.*}}, %{{.*}} : f64
//  CHECK-NEXT: stencil.return %{{.*}} : f64

   func @redirect(%in : !stencil.field<ijk,f64>, %out1 : !stencil.field<ijk,f64>, %out2 : !stencil.field<ijk,f64>)
     attributes { stencil.program } {
   	stencil.assert %in ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
     stencil.assert %out1 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
     stencil.assert %out2 ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
     %0 = stencil.load %in : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
     %1 = stencil.apply %arg1 = %0 : !stencil.view<ijk,f64> {  
         %2 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
         %3 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
         %4 = addf %2, %3 : f64
         stencil.return %4 : f64
   	} : !stencil.view<ijk,f64>
     %5 = stencil.apply %arg2 = %0, %arg3 = %1 : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
         %6 = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
         %7 = stencil.access %arg3[1, 2, 3] : (!stencil.view<ijk,f64>) -> f64
         %8 = addf %6, %7 : f64
         stencil.return %8 : f64
   	} : !stencil.view<ijk,f64>
// 	stencil.store %1 to %out1 ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  	stencil.store %5 to %out2 ([0, 0, 0]:[64, 64, 60]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
    return
  }

// CHECK-LABEL: func @redirect(%{{.*}}: !stencil.field<ijk,f64>, %{{.*}}: !stencil.field<ijk,f64>, %{{.*}}: !stencil.field<ijk,f64>) attributes {stencil.program}
//  CHECK-NEXT: stencil.assert %{{.*}} ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
//  CHECK-NEXT: stencil.assert %{{.*}} ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
//  CHECK-NEXT: stencil.assert %{{.*}} ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
//  CHECK-NEXT: %{{.*}} = stencil.apply %{{.*}} = %{{.*}} : !stencil.view<ijk,f64> {
//       CHECK: %{{.*}} = stencil.access %{{.*}}[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
//       CHECK: %{{.*}} = stencil.access %{{.*}}[0, 2, 3] : (!stencil.view<ijk,f64>) -> f64
//       CHECK: %{{.*}} = stencil.access %{{.*}}[2, 2, 3] : (!stencil.view<ijk,f64>) -> f64