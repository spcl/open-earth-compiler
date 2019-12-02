// RUN: mlir-opt %s --sstencil-shape-inference | mlir-opt | FileCheck %s

func @lap(%in : !sstencil.view<?x?x?xf64>) -> f64
  attributes { sstencil.function } {
	%0 = sstencil.access %in[-1, 0, 0] : !sstencil.view<?x?x?xf64>
	%1 = sstencil.access %in[ 1, 0, 0] : !sstencil.view<?x?x?xf64>
	%2 = sstencil.access %in[ 0, 1, 0] : !sstencil.view<?x?x?xf64>
	%3 = sstencil.access %in[ 0,-1, 0] : !sstencil.view<?x?x?xf64>
	%4 = sstencil.access %in[ 0, 0, 0] : !sstencil.view<?x?x?xf64>
	%5 = addf %0, %1 : f64
	%6 = addf %2, %3 : f64
	%7 = addf %5, %6 : f64
	%8 = constant -4.0 : f64
	%9 = mulf %4, %8 : f64
	%10 = addf %9, %7 : f64
	return %10 : f64
}

// CHECK-LABEL: func @lap(%{{.*}}: !sstencil.view<?x?x?xf64>) -> f64
//  CHECK-NEXT: attributes {sstencil.function} {
//  CHECK-NEXT: %{{.*}} = sstencil.access %{{.*}}[-1, 0, 0] : !sstencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sstencil.access %{{.*}}[1, 0, 0] : !sstencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sstencil.access %{{.*}}[0, 1, 0] : !sstencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sstencil.access %{{.*}}[0, -1, 0] : !sstencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sstencil.access %{{.*}}[0, 0, 0] : !sstencil.view<?x?x?xf64>

func @lap_stencil(%in: !sstencil.field<?x?x?xf64>, %out: !sstencil.field<12x12x16xf64>)
  attributes { sstencil.program } {
	%0 = "sstencil.load"(%in) : (!sstencil.field<?x?x?xf64>) -> !sstencil.view<?x?x?xf64>
	%1 = "sstencil.apply"(%0) { callee = @lap } : (!sstencil.view<?x?x?xf64>) -> !sstencil.view<?x?x?xf64>
	%2 = "sstencil.apply"(%1) { callee = @lap } : (!sstencil.view<?x?x?xf64>) -> !sstencil.view<?x?x?xf64>
	"sstencil.store"(%out, %2) : (!sstencil.field<12x12x16xf64>, !sstencil.view<?x?x?xf64>) -> ()
	return
}

// CHECK-LABEL: func @lap_stencil(%{{.*}}: !sstencil.field<?x?x?xf64>, %{{.*}}: !sstencil.field<12x12x16xf64>)
//  CHECK-NEXT: attributes {sstencil.program}
//  CHECK-NEXT: %{{.*}} = sstencil.load %{{.*}} : (!sstencil.field<?x?x?xf64>) -> !sstencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sstencil.apply @lap(%{{.*}}) : (!sstencil.view<?x?x?xf64>) -> !sstencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sstencil.apply @lap(%{{.*}}) : (!sstencil.view<?x?x?xf64>) -> !sstencil.view<?x?x?xf64>
//  CHECK-NEXT: sstencil.store %{{.*}}, %{{.*}} : !sstencil.field<12x12x16xf64>, !sstencil.view<?x?x?xf64>