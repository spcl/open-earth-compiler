// RUN: oec-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @field(%{{.*}}: !stencil.field<?x?x?xf32>) {
func @field(%arg0: !stencil.field<?x?x?xf32>) {
	return
}

// -----

// CHECK-LABEL: func @ldfield(%{{.*}}: !stencil.field<?x0x?xf64>) {
func @ldfield(%arg0: !stencil.field<?x0x?xf64>) {
	return
}

// -----

// CHECK-LABEL: func @temp(%{{.*}}: !stencil.temp<1x2x3xf32>) {
func @temp(%arg0: !stencil.temp<1x2x3xf32>) {
	return
}

// -----

// CHECK-LABEL: func @ldtemp(%{{.*}}: !stencil.temp<0x0x3xf64>) {
func @ldtemp(%arg0: !stencil.temp<0x0x3xf64>) {
	return
}

