// RUN: oec-opt %s -split-input-file --stencil-index-optimization | FileCheck %s

// CHECK-LABEL: @address_computation
llvm.func @address_computation(%arg0: !llvm<"double*">) attributes {gpu.kernel} {
  // CHECK: {{%.*}} = llvm.mlir.constant(64 : i32) : !llvm.i32
  %0 = llvm.mlir.constant(64 : index) : !llvm.i64
  %1 = nvvm.read.ptx.sreg.ctaid.y : !llvm.i32
  %2 = llvm.sext %1 : !llvm.i32 to !llvm.i64
  // CHECK: {{%.*}} = llvm.mul {{%.*}}, {{%.*}} : !llvm.i32
  %3 = llvm.mul %0, %2 : !llvm.i64
  // CHECK: {{%.*}} = llvm.add {{%.*}}, {{%.*}} : !llvm.i32
  %4 = llvm.add %3, %0 : !llvm.i64
  // CHECK: {{%.*}} = llvm.sext {{%.*}} : !llvm.i32 to !llvm.i64
  %5 = llvm.getelementptr %arg0[%4] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
  %6 = llvm.mlir.constant(1.0 : f64) : !llvm.double
  llvm.store %6, %5 : !llvm<"double*">
  llvm.return
}

// -----

// CHECK-LABEL: @comparison_op
llvm.func @comparison_op(%arg0: !llvm<"double*">) attributes {gpu.kernel} {
  // CHECK: {{%.*}} = llvm.mlir.constant(64 : i32) : !llvm.i32
  %0 = llvm.mlir.constant(64 : index) : !llvm.i64
  %1 = nvvm.read.ptx.sreg.ctaid.y : !llvm.i32
  %2 = llvm.sext %1 : !llvm.i32 to !llvm.i64
  // CHECK: {{%.*}} = llvm.icmp "slt" {{%.*}}, {{%.*}} : !llvm.i32
  %3 = llvm.icmp "slt" %0, %2 : !llvm.i64
  llvm.cond_br %3, ^bb1, ^bb2
^bb1:       // pred: ^bb0
  %5 = llvm.getelementptr %arg0[%0] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
  %6 = llvm.mlir.constant(1.0 : f64) : !llvm.double
  llvm.store %6, %5 : !llvm<"double*">
  llvm.br ^bb2
^bb2:       
  llvm.return
}
