// RUN: oec-opt %s | FileCheck %s

func @foo0(%arg0: !stencil.field<?x?x?xf64>) {
	return
}
// CHECK-LABEL: func @foo0(%{{.*}}: !stencil.field<?x?x?xf64>) {

func @foo1(%arg0: !stencil.field<4096x12x37xf64>) {
	return
}
// CHECK-LABEL: func @foo1(%{{.*}}: !stencil.field<4096x12x37xf64>) {


func @bar0(%arg0: !stencil.view<?x?x?xf64>) {
	return
}
// CHECK-LABEL: func @bar0(%{{.*}}: !stencil.view<?x?x?xf64>) {

func @bar1(%arg0: !stencil.view<?x42x?xf64>) {
	return
}
// CHECK-LABEL: func @bar1(%{{.*}}: !stencil.view<?x42x?xf64>) {
