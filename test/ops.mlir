// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func @lap(%in : !sten.view<?x?x?xf64>) -> f64
  attributes { sten.function } {
    %0 = "sten.access"(%in) {offset = [-1, 0, 0]} : (!sten.view<?x?x?xf64>) -> f64
    %1 = "sten.access"(%in) {offset = [ 1, 0, 0]} : (!sten.view<?x?x?xf64>) -> f64
    %2 = "sten.access"(%in) {offset = [ 0, 1, 0]} : (!sten.view<?x?x?xf64>) -> f64
    %3 = "sten.access"(%in) {offset = [ 0,-1, 0]} : (!sten.view<?x?x?xf64>) -> f64
    %4 = "sten.access"(%in) {offset = [ 0, 0, 0]} : (!sten.view<?x?x?xf64>) -> f64
//	%0 = sten.access %in[-1, 0, 0] : !sten.view<?x?x?xf64>
//	%1 = sten.access %in[ 1, 0, 0] : !sten.view<?x?x?xf64>
//	%2 = sten.access %in[ 0, 1, 0] : !sten.view<?x?x?xf64>
//	%3 = sten.access %in[ 0,-1, 0] : !sten.view<?x?x?xf64>
//	%4 = sten.access %in[ 0, 0, 0] : !sten.view<?x?x?xf64>
	%5 = addf %0, %1 : f64
	%6 = addf %2, %3 : f64
	%7 = addf %5, %6 : f64
	%8 = constant -4.0 : f64
	%9 = mulf %4, %8 : f64
	%10 = addf %9, %7 : f64
	return %10 : f64
}

// CHECK-LABEL: func @lap(%{{.*}}: !sten.view<?x?x?xf64>) -> f64
//  CHECK-NEXT: attributes {sten.function} {
//  CHECK-NEXT: %{{.*}} = sten.access %{{.*}}[-1, 0, 0] : !sten.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sten.access %{{.*}}[1, 0, 0] : !sten.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sten.access %{{.*}}[0, 1, 0] : !sten.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sten.access %{{.*}}[0, -1, 0] : !sten.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sten.access %{{.*}}[0, 0, 0] : !sten.view<?x?x?xf64>

func @lap_stencil(%in: !sten.field<12x13x?xf64>, %out: !sten.field<12x12x16xf64>)
  attributes { sten.program } {
	%0 = "sten.load"(%in) : (!sten.field<12x13x?xf64>) -> !sten.view<?x?x?xf64>
//	%0 = sten.load %in : !sten.field<?x?x?xf64>
	%1 = "sten.apply"(%0) { callee = @lap } : (!sten.view<?x?x?xf64>) -> !sten.view<?x?x?xf64>
//	%1 = sten.apply<lap> %0 : !sten.view<?x?x?xf64>
	%2 = "sten.apply"(%1) { callee = @lap } : (!sten.view<?x?x?xf64>) -> !sten.view<?x?x?xf64>
	"sten.store"(%out, %2) : (!sten.field<12x12x16xf64>, !sten.view<?x?x?xf64>) -> ()
//  sten.store %out, %2 : !sten.field<12x12x16xf64>, !sten.view<?x?x?xf64>
	return
}


//func @lap_stencil(%in: !sten.field<?x?x?xf64>, %out: !sten.field<12x12x16xf64>)
//  attributes { sten.program } {
//	%0 = sten.load %in : (!sten.field<?x?x?xf64>) -> !sten.view<?x?x?xf64>
//	%1 = sten.apply @lap(%0) : (!sten.view<?x?x?xf64>) -> !sten.view<?x?x?xf64>
//	%2 = sten.apply @lap(%1) : (!sten.view<?x?x?xf64>) -> !sten.view<?x?x?xf64>
//	sten.store %out, %2 : !sten.field<12x12x16xf64>, !sten.view<?x?x?xf64>
//	return
//}


// CHECK-LABEL: func @lap_stencil(%{{.*}}: !sten.field<12x13x?xf64>, %{{.*}}: !sten.field<12x12x16xf64>)
//  CHECK-NEXT: attributes {sten.program}
//  CHECK-NEXT: sten.load %{{.*}} : (!sten.field<12x13x?xf64>) -> !sten.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sten.apply @lap(%{{.*}}) : (!sten.view<?x?x?xf64>) -> !sten.view<?x?x?xf64>
//  CHECK-NEXT: %{{.*}} = sten.apply @lap(%{{.*}}) : (!sten.view<?x?x?xf64>) -> !sten.view<?x?x?xf64>
//  CHECK-NEXT: sten.store %{{.*}}, %{{.*}} : !sten.field<12x12x16xf64>, !sten.view<?x?x?xf64>

//func @laplap(%in : !sstencil.view<?x?x?xf64>) -> f64
//  attributes { sstencil.function } {
//    %0 = sstencil.call @lap(%in)[-1, 0, 0] : !sstencil.view<?x?x?xf64>
//    %1 = sstencil.call @lap(%in)[ 1, 0, 0] : !sstencil.view<?x?x?xf64>
//    %2 = sstencil.call @lap(%in)[ 0, 1, 0] : !sstencil.view<?x?x?xf64>
//    %3 = sstencil.call @lap(%in)[ 0,-1, 0] : !sstencil.view<?x?x?xf64>
//    %4 = sstencil.call @lap(%in)[ 0, 0, 0] : !sstencil.view<?x?x?xf64>
//    %5 = addf %0, %1 : f64
//    %6 = addf %2, %3 : f64
//    %7 = addf %5, %6 : f64
//    %8 = constant -4.0 : f64
//    %9 = mulf %4, %8 : f64
//    %10 = addf %9, %7 : f64
//    return %10 : f64
//}

func @foo(%in : !sten.view<?x?x?xf64>) -> f64
  attributes { sten.function } {
//    %0 = sten.call @lap(%in)[-1, 0, 0] : !sten.view<?x?x?xf64>
    %0 = "sten.call"(%in) { callee = @lap, offset = [-1, 0, 0] } : (!sten.view<?x?x?xf64>) -> f64
    %1 = addf %0, %0 : f64
    return %1 : f64
}

// CHECK-LABEL: func @foo(%{{.*}}: !sten.view<?x?x?xf64>) -> f64
//  CHECK-NEXT: attributes {sten.function} {
//  CHECK-NEXT: sten.call @lap(%{{.*}})[-1, 0, 0] : (!sten.view<?x?x?xf64>) -> f64
