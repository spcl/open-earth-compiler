// RUN: mlir-opt %s | FileCheck %s

func @foo(%arg0: !sten.field<?x?x?xf64>) {
	return
}
// CHECK-LABEL: func @foo(%{{.*}}: !sten.field<?x?x?xf64>) {

func @bar(%arg0: !sten.view<?x?x?xf64>) {
	return
}
// CHECK-LABEL: func @bar(%{{.*}}: !sten.view<?x?x?xf64>) {
