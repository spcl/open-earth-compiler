// RUN: oec-opt %s | oec-opt | FileCheck %s

func @lap(%in : !stencil.view<?x?x?xf64>) -> f64
  attributes { stencil.function } {
	%0 = "stencil.access"(%in) {offset = [-1, 0, 0]} : (!stencil.view<?x?x?xf64>) -> f64
	%1 = "stencil.access"(%in) {offset = [ 1, 0, 0]} : (!stencil.view<?x?x?xf64>) -> f64
	%2 = "stencil.access"(%in) {offset = [ 0, 1, 0]} : (!stencil.view<?x?x?xf64>) -> f64
	%3 = "stencil.access"(%in) {offset = [ 0,-1, 0]} : (!stencil.view<?x?x?xf64>) -> f64
	%4 = "stencil.access"(%in) {offset = [ 0, 0, 0]} : (!stencil.view<?x?x?xf64>) -> f64
	%5 = addf %0, %1 : f64
	%6 = addf %2, %3 : f64
	%7 = addf %5, %6 : f64
	%8 = constant -4.0 : f64
	%9 = mulf %4, %8 : f64
	%10 = addf %9, %7 : f64
	return %10 : f64
}

// CHECK-LABEL: func @lap(%{{.*}}: !stencil.view<?x?x?xf64>) -> f64
//  CHECK-NEXT: attributes {stencil.function} {
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[-1, 0, 0] : !stencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[1, 0, 0] : !stencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[0, 1, 0] : !stencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[0, -1, 0] : !stencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[0, 0, 0] : !stencil.view<?x?x?xf64>

func @lap_stencil(%in: !stencil.field<?x?x?xf64>, %out: !stencil.field<12x12x16xf64>)
  attributes { stencil.program } {
	%0 = "stencil.load"(%in) : (!stencil.field<?x?x?xf64>) -> !stencil.view<?x?x?xf64>
	%1 = "stencil.apply"(%0) { callee = @lap } : (!stencil.view<?x?x?xf64>) -> !stencil.view<?x?x?xf64>
	%2 = "stencil.apply"(%1) { callee = @lap } : (!stencil.view<?x?x?xf64>) -> !stencil.view<?x?x?xf64>
	"stencil.store"(%2, %out) : (!stencil.view<?x?x?xf64>, !stencil.field<12x12x16xf64>) -> ()
	return
}

// CHECK-LABEL: func @lap_stencil(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<12x12x16xf64>)
//  CHECK-NEXT: attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<?x?x?xf64>) -> !stencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply @lap(%{{.*}}) : (!stencil.view<?x?x?xf64>) -> !stencil.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = stencil.apply @lap(%{{.*}}) : (!stencil.view<?x?x?xf64>) -> !stencil.view<?x?x?xf64>
//  CHECK-NEXT: stencil.store %{{.*}} to %{{.*}} : !stencil.view<?x?x?xf64>, !stencil.field<12x12x16xf64>
