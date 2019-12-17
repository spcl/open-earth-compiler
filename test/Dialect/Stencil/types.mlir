// RUN: oec-opt %s | FileCheck %s

func @foo0(%arg0: !stencil.field<IJK,f32>) {
	return
}
// CHECK-LABEL: func @foo0(%{{.*}}: !stencil.field<IJK,f32>) {

func @foo1(%arg0: !stencil.field<IJ,f64>) {
	return
}
// CHECK-LABEL: func @foo1(%{{.*}}: !stencil.field<IJ,f64>) {


func @bar0(%arg0: !stencil.view<IJK,f32>) {
	return
}
// CHECK-LABEL: func @bar0(%{{.*}}: !stencil.view<IJK,f32>) {

func @bar1(%arg0: !stencil.view<K,f64>) {
	return
}
// CHECK-LABEL: func @bar1(%{{.*}}: !stencil.view<K,f64>) {
