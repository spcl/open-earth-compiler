// RUN: oec-opt %s | oec-opt | FileCheck %s

func @lap(%in : !stencil.view<IJK,f64>) -> f64
  attributes { stencil.function } {
	%0 = "stencil.access"(%in) {offset = [-1, 0, 0]} : (!stencil.view<IJK,f64>) -> f32
	%1 = "stencil.access"(%in) {offset = [ 1, 0, 0]} : (!stencil.view<IJK,f64>) -> f32
	%2 = "stencil.access"(%in) {offset = [ 0, 1, 0]} : (!stencil.view<IJK,f64>) -> f32
	%3 = "stencil.access"(%in) {offset = [ 0,-1, 0]} : (!stencil.view<IJK,f64>) -> f32
	%4 = "stencil.access"(%in) {offset = [ 0, 0, 0]} : (!stencil.view<IJK,f64>) -> f32
	%5 = addf %0, %1 : f32
	%6 = addf %2, %3 : f32
	%7 = addf %5, %6 : f32
	%8 = constant -4.0 : f32
	%9 = mulf %4, %8 : f32
	%10 = addf %9, %7 : f32
	return %10 : f32
}

// CHECK-LABEL: func @lap(%{{.*}}: !stencil.view<IJK,f64>) -> f64
//  CHECK-NEXT: attributes {stencil.function} {
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[-1, 0, 0] : !stencil.view<IJK,f64>
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[1, 0, 0] : !stencil.view<IJK,f64>
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[0, 1, 0] : !stencil.view<IJK,f64>
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[0, -1, 0] : !stencil.view<IJK,f64>
//  CHECK-NEXT: %{{.*}} = stencil.access %{{.*}}[0, 0, 0] : !stencil.view<IJK,f64>

func @lap_stencil(%in: !stencil.field<IJK,f64>, %out: !stencil.field<IJK,f64>)
  attributes { stencil.program } {
	%0 = "stencil.load"(%in) : (!stencil.field<IJK,f64>) -> !stencil.view<IJK,f64>
	%1 = "stencil.apply"(%0) { callee = @lap } : (!stencil.view<IJK,f64>) -> !stencil.view<IJK,f64>
	%2 = "stencil.apply"(%1) { callee = @lap } : (!stencil.view<IJK,f64>) -> !stencil.view<IJK,f64>
	"stencil.store"(%2, %out) : (!stencil.view<IJK,f64>, !stencil.field<IJK,f64>) -> ()
	return
}

// CHECK-LABEL: func @lap_stencil(%{{.*}}: !stencil.field<IJK,f64>, %{{.*}}: !stencil.field<IJK,f64>)
//  CHECK-NEXT: attributes {stencil.program}
//  CHECK-NEXT: %{{.*}} = stencil.load %{{.*}} : (!stencil.field<IJK,f64>) -> !stencil.view<IJK,f64>
//  CHECK-NEXT: %{{.*}} = stencil.apply @lap(%{{.*}}) : (!stencil.view<IJK,f64>) -> !stencil.view<IJK,f64>
//  CHECK-NEXT: %{{.*}} = stencil.apply @lap(%{{.*}}) : (!stencil.view<IJK,f64>) -> !stencil.view<IJK,f64>
//  CHECK-NEXT: stencil.store %{{.*}} to %{{.*}} : !stencil.view<IJK,f64>, !stencil.field<IJK,f64>
