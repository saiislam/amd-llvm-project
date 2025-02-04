; This run is to demonstrate what MIR SSA looks like.
; RUN: llc -max-registers-for-gc-values=4 -stop-after finalize-isel < %s | FileCheck --check-prefix=CHECK-VREG %s
; This run is to demonstrate register allocator work.
; RUN: llc -max-registers-for-gc-values=4 -stop-after virtregrewriter < %s | FileCheck --check-prefix=CHECK-PREG %s
; This run is to demonstrate resulting assembly/stackmaps.
; NOTE: When D81647 is landed this run line will need to be adjusted!
; RUN: llc -max-registers-for-gc-values=4 < %s | FileCheck --check-prefix=CHECK-ASM %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare i1 @return_i1()
declare void @func()
declare void @consume(i32 addrspace(1)*)
declare void @consume2(i32 addrspace(1)*, i32 addrspace(1)*)
declare void @consume5(i32 addrspace(1)*, i32 addrspace(1)*, i32 addrspace(1)*, i32 addrspace(1)*, i32 addrspace(1)*)
declare void @use1(i32 addrspace(1)*, i8 addrspace(1)*)

; test most simple relocate
define i1 @test_relocate(i32 addrspace(1)* %a) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_relocate
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    %1:gr64 = STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, %0, %0(tied-def 0), csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def $al
; CHECK-VREG:    %2:gr8 = COPY $al
; CHECK-VREG:    $rdi = COPY %1
; CHECK-VREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_relocate
; CHECK-PREG:    renamable $rbx = COPY $rdi
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, killed renamable $rbx, renamable $rbx(tied-def 0), csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def $al
; CHECK-PREG:    renamable $bpl = COPY killed $al
; CHECK-PREG:    $rdi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

; CHECK-ASM-LABEL: test_relocate:
; CHECK-ASM:       # %bb.0:
; CHECK-ASM-NEXT:	pushq	%rbp
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	pushq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 24
; CHECK-ASM-NEXT:	pushq	%rax
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 32
; CHECK-ASM-NEXT:	.cfi_offset %rbx, -24
; CHECK-ASM-NEXT:	.cfi_offset %rbp, -16
; CHECK-ASM-NEXT:	movq	%rdi, %rbx
; CHECK-ASM-NEXT:	callq	return_i1
; CHECK-ASM-NEXT:  .Ltmp0:
; CHECK-ASM-NEXT:	movl	%eax, %ebp
; CHECK-ASM-NEXT:	movq	%rbx, %rdi
; CHECK-ASM-NEXT:	callq	consume
; CHECK-ASM-NEXT:	movl	%ebp, %eax
; CHECK-ASM-NEXT:	addq	$8, %rsp
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 24
; CHECK-ASM-NEXT:	popq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	popq	%rbp
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 8
; CHECK-ASM-NEXT:	retq
entry:
  %safepoint_token = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %a)]
  %rel1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %res1 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  call void @consume(i32 addrspace(1)* %rel1)
  ret i1 %res1
}
; test pointer variables intermixed with pointer constants
define void @test_mixed(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_mixed
; CHECK-VREG:    %2:gr64 = COPY $rdx
; CHECK-VREG:    %1:gr64 = COPY $rsi
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    %3:gr64, %4:gr64, %5:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, %2, %2(tied-def 0), 2, 0, 2, 0, %1, %1(tied-def 1), %0, %0(tied-def 2), csr_64
; CHECK-VREG:    %6:gr32 = MOV32r0 implicit-def dead $eflags
; CHECK-VREG:    %7:gr64 = SUBREG_TO_REG 0, killed %6, %subreg.sub_32bit
; CHECK-VREG:    $rdi = COPY %5
; CHECK-VREG:    $rsi = COPY %7
; CHECK-VREG:    $rdx = COPY %4
; CHECK-VREG:    $rcx = COPY %7
; CHECK-VREG:    $r8 = COPY %3
; CHECK-VREG:    CALL64pcrel32 @consume5, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit $rdx, implicit $rcx, implicit $r8, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_mixed
; CHECK-PREG:    renamable $r14 = COPY $rdx
; CHECK-PREG:    renamable $r15 = COPY $rsi
; CHECK-PREG:    renamable $rbx = COPY $rdi
; CHECK-PREG:    renamable $r14, renamable $r15, renamable $rbx = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, killed renamable $r14, renamable $r14(tied-def 0), 2, 0, 2, 0, killed renamable $r15, renamable $r15(tied-def 1), killed renamable $rbx, renamable $rbx(tied-def 2), csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-PREG:    $rdi = COPY killed renamable $rbx
; CHECK-PREG:    dead $esi = MOV32r0 implicit-def dead $eflags, implicit-def $rsi
; CHECK-PREG:    $rdx = COPY killed renamable $r15
; CHECK-PREG:    dead $ecx = MOV32r0 implicit-def dead $eflags, implicit-def $rcx
; CHECK-PREG:    $r8 = COPY killed renamable $r14
; CHECK-PREG:    CALL64pcrel32 @consume5, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit $rdx, implicit killed $rcx, implicit killed $r8, implicit-def $rsp, implicit-def $ssp

; CHECK-ASM-LABEL: test_mixed:
; CHECK-ASM:        # %bb.0:                                # %entry
; CHECK-ASM-NEXT:	pushq	%r15
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	pushq	%r14
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 24
; CHECK-ASM-NEXT:	pushq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 32
; CHECK-ASM-NEXT:	.cfi_offset %rbx, -32
; CHECK-ASM-NEXT:	.cfi_offset %r14, -24
; CHECK-ASM-NEXT:	.cfi_offset %r15, -16
; CHECK-ASM-NEXT:	movq	%rdx, %r14
; CHECK-ASM-NEXT:	movq	%rsi, %r15
; CHECK-ASM-NEXT:	movq	%rdi, %rbx
; CHECK-ASM-NEXT:	callq	func
; CHECK-ASM-NEXT:.Ltmp1:
; CHECK-ASM-NEXT:	movq	%rbx, %rdi
; CHECK-ASM-NEXT:	xorl	%esi, %esi
; CHECK-ASM-NEXT:	movq	%r15, %rdx
; CHECK-ASM-NEXT:	xorl	%ecx, %ecx
; CHECK-ASM-NEXT:	movq	%r14, %r8
; CHECK-ASM-NEXT:	callq	consume5
; CHECK-ASM-NEXT:	popq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 24
; CHECK-ASM-NEXT:	popq	%r14
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	popq	%r15
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 8
; CHECK-ASM-NEXT:	retq
entry:
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %a, i32 addrspace(1)* null, i32 addrspace(1)* %b, i32 addrspace(1)* null, i32 addrspace(1)* %c)]
  %rel1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %rel2 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 1, i32 1)
  %rel3 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 2, i32 2)
  %rel4 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 3, i32 3)
  %rel5 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 4, i32 4)
  call void @consume5(i32 addrspace(1)* %rel1, i32 addrspace(1)* %rel2, i32 addrspace(1)* %rel3, i32 addrspace(1)* %rel4, i32 addrspace(1)* %rel5)
  ret void
}

; same as above, but for alloca
define i32 addrspace(1)* @test_alloca(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_alloca
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    MOV64mr %stack.0.alloca, 1, $noreg, 0, $noreg, %0 :: (store 8 into %ir.alloca)
; CHECK-VREG:    %1:gr64 = STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, %0, %0(tied-def 0), 0, %stack.0.alloca, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def $al :: (volatile load store 8 on %stack.0.alloca)
; CHECK-VREG:    %2:gr8 = COPY $al
; CHECK-VREG:    %3:gr64 = MOV64rm %stack.0.alloca, 1, $noreg, 0, $noreg :: (dereferenceable load 8 from %ir.alloca)
; CHECK-VREG:    $rdi = COPY %1
; CHECK-VREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_alloca
; CHECK-PREG:    renamable $rbx = COPY $rdi
; CHECK-PREG:    MOV64mr %stack.0.alloca, 1, $noreg, 0, $noreg, renamable $rbx :: (store 8 into %ir.alloca)
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, killed renamable $rbx, renamable $rbx(tied-def 0), 0, %stack.0.alloca, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def dead $al :: (volatile load store 8 on %stack.0.alloca)
; CHECK-PREG:    renamable $r14 = MOV64rm %stack.0.alloca, 1, $noreg, 0, $noreg :: (dereferenceable load 8 from %ir.alloca)
; CHECK-PREG:    $rdi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

; CHECK-ASM-LABEL: test_alloca:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:	pushq	%r14
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	pushq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 24
; CHECK-ASM-NEXT:	pushq	%rax
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 32
; CHECK-ASM-NEXT:	.cfi_offset %rbx, -24
; CHECK-ASM-NEXT:	.cfi_offset %r14, -16
; CHECK-ASM-NEXT:	movq	%rdi, %rbx
; CHECK-ASM-NEXT:	movq	%rdi, (%rsp)
; CHECK-ASM-NEXT:	callq	return_i1
; CHECK-ASM-NEXT:  .Ltmp2:
; CHECK-ASM-NEXT:	movq	(%rsp), %r14
; CHECK-ASM-NEXT:	movq	%rbx, %rdi
; CHECK-ASM-NEXT:	callq	consume
; CHECK-ASM-NEXT:	movq	%r14, %rax
; CHECK-ASM-NEXT:	addq	$8, %rsp
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 24
; CHECK-ASM-NEXT:	popq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	popq	%r14
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 8
; CHECK-ASM-NEXT:	retq
entry:
  %alloca = alloca i32 addrspace(1)*, align 8
  store i32 addrspace(1)* %ptr, i32 addrspace(1)** %alloca
  %safepoint_token = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)** %alloca, i32 addrspace(1)* %ptr)]
  %rel1 = load i32 addrspace(1)*, i32 addrspace(1)** %alloca
  %rel2 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 1, i32 1)
  call void @consume(i32 addrspace(1)* %rel2)
  ret i32 addrspace(1)* %rel1
}

; test base != derived
define void @test_base_derived(i32 addrspace(1)* %base, i32 addrspace(1)* %derived) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_base_derived
; CHECK-VREG:    %1:gr64 = COPY $rsi
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    MOV64mr %stack.0, 1, $noreg, 0, $noreg, %0 :: (store 8 into %stack.0)
; CHECK-VREG:    %2:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 1, 8, %stack.0, 0, %1(tied-def 0), csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store 8 on %stack.0)
; CHECK-VREG:    $rdi = COPY %2
; CHECK-VREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_base_derived
; CHECK-PREG:    renamable $rbx = COPY $rsi
; CHECK-PREG:    MOV64mr %stack.0, 1, $noreg, 0, $noreg, killed renamable $rdi :: (store 8 into %stack.0)
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 1, 8, %stack.0, 0, killed renamable $rbx(tied-def 0), csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store 8 on %stack.0)
; CHECK-PREG:    $rdi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

; CHECK-ASM-LABEL: test_base_derived:
; CHECK-ASM:       # %bb.0:
; CHECK-ASM-NEXT:	pushq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	subq	$16, %rsp
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 32
; CHECK-ASM-NEXT:	.cfi_offset %rbx, -16
; CHECK-ASM-NEXT:	movq	%rsi, %rbx
; CHECK-ASM-NEXT:	movq	%rdi, 8(%rsp)
; CHECK-ASM-NEXT:	callq	func
; CHECK-ASM-NEXT:  .Ltmp3:
; CHECK-ASM-NEXT:	movq	%rbx, %rdi
; CHECK-ASM-NEXT:	callq	consume
; CHECK-ASM-NEXT:	addq	$16, %rsp
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	popq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 8
; CHECK-ASM-NEXT:	retq
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %base, i32 addrspace(1)* %derived)]
  %reloc = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 1)
  call void @consume(i32 addrspace(1)* %reloc)
  ret void
}

; deopt GC pointer not present in GC args must be spilled
define void @test_deopt_gcpointer(i32 addrspace(1)* %a, i32 addrspace(1)* %b) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_deopt_gcpointer
; CHECK-VREG:    %1:gr64 = COPY $rsi
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    MOV64mr %stack.0, 1, $noreg, 0, $noreg, %0 :: (store 8 into %stack.0)
; CHECK-VREG:    %2:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 1, 1, 8, %stack.0, 0, %1, %1(tied-def 0), csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store 8 on %stack.0)
; CHECK-VREG:    $rdi = COPY %2
; CHECK-VREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    RET 0

; CHECK-PREG-LABEL: name:            test_deopt_gcpointer
; CHECK-PREG:    renamable $rbx = COPY $rsi
; CHECK-PREG:    MOV64mr %stack.0, 1, $noreg, 0, $noreg, killed renamable $rdi :: (store 8 into %stack.0)
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 1, 1, 8, %stack.0, 0, killed renamable $rbx, renamable $rbx(tied-def 0), csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store 8 on %stack.0)
; CHECK-PREG:    $rdi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

; CHECK-ASM-LABEL: test_deopt_gcpointer:
; CHECK-ASM:       # %bb.0:
; CHECK-ASM-NEXT:	pushq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	subq	$16, %rsp
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 32
; CHECK-ASM-NEXT:	.cfi_offset %rbx, -16
; CHECK-ASM-NEXT:	movq	%rsi, %rbx
; CHECK-ASM-NEXT:	movq	%rdi, 8(%rsp)
; CHECK-ASM-NEXT:	callq	func
; CHECK-ASM-NEXT:  .Ltmp4:
; CHECK-ASM-NEXT:	movq	%rbx, %rdi
; CHECK-ASM-NEXT:	callq	consume
; CHECK-ASM-NEXT:	addq	$16, %rsp
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	popq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 8
; CHECK-ASM-NEXT:	retq
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["deopt" (i32 addrspace(1)* %a), "gc-live" (i32 addrspace(1)* %b)]
  %rel = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  call void @consume(i32 addrspace(1)* %rel)
  ret void
}

;; Two gc.relocates of the same input, should require only a single spill/fill
define void @test_gcrelocate_uniqueing(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_gcrelocate_uniqueing
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    %1:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 2, %0, 2, 4278124286, %0, %0(tied-def 0), csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    $rdi = COPY %1
; CHECK-VREG:    $rsi = COPY %1
; CHECK-VREG:    CALL64pcrel32 @consume2, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_gcrelocate_uniqueing
; CHECK-PREG:    renamable $rbx = COPY $rdi
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 2, killed renamable $rbx, 2, 4278124286, renamable $rbx, renamable $rbx(tied-def 0), csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-PREG:    $rdi = COPY renamable $rbx
; CHECK-PREG:    $rsi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @consume2, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit killed $rsi, implicit-def $rsp, implicit-def $ssp

; CHECK-ASM-LABEL: test_gcrelocate_uniqueing:
; CHECK-ASM:       # %bb.0:
; CHECK-ASM-NEXT:	pushq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	.cfi_offset %rbx, -16
; CHECK-ASM-NEXT:	movq	%rdi, %rbx
; CHECK-ASM-NEXT:	callq	func
; CHECK-ASM-NEXT: .Ltmp5:
; CHECK-ASM-NEXT:	movq	%rbx, %rdi
; CHECK-ASM-NEXT:	movq	%rbx, %rsi
; CHECK-ASM-NEXT:	callq	consume2
; CHECK-ASM-NEXT:	popq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 8
; CHECK-ASM-NEXT:	retq
  %tok = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["deopt" (i32 addrspace(1)* %ptr, i32 undef), "gc-live" (i32 addrspace(1)* %ptr, i32 addrspace(1)* %ptr)]
  %a = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok, i32 0, i32 0)
  %b = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok, i32 1, i32 1)
  call void @consume2(i32 addrspace(1)* %a, i32 addrspace(1)* %b)
  ret void
}

; Two gc.relocates of a bitcasted pointer should only require a single spill/fill
define void @test_gcptr_uniqueing(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_gcptr_uniqueing
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG:    %2:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 2, %0, 2, 4278124286, %0, %0(tied-def 0), csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    ADJCALLSTACKUP64 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG:    %1:gr64 = COPY %2
; CHECK-VREG:    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG:    $rdi = COPY %2
; CHECK-VREG:    $rsi = COPY %1
; CHECK-VREG:    CALL64pcrel32 @use1, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_gcptr_uniqueing
; CHECK-PREG:    renamable $rbx = COPY $rdi
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 2, killed renamable $rbx, 2, 4278124286, renamable $rbx, renamable $rbx(tied-def 0), csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-PREG:    $rdi = COPY renamable $rbx
; CHECK-PREG:    $rsi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @use1, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit killed $rsi, implicit-def $rsp, implicit-def $ssp

; CHECK-ASM-LABEL: test_gcptr_uniqueing:
; CHECK-ASM:       # %bb.0:
; CHECK-ASM-NEXT:	pushq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:	.cfi_offset %rbx, -16
; CHECK-ASM-NEXT:	movq	%rdi, %rbx
; CHECK-ASM-NEXT:	callq	func
; CHECK-ASM-NEXT:  .Ltmp6:
; CHECK-ASM-NEXT:	movq	%rbx, %rdi
; CHECK-ASM-NEXT:	movq	%rbx, %rsi
; CHECK-ASM-NEXT:	callq	use1
; CHECK-ASM-NEXT:	popq	%rbx
; CHECK-ASM-NEXT:	.cfi_def_cfa_offset 8
; CHECK-ASM-NEXT:	retq
  %ptr2 = bitcast i32 addrspace(1)* %ptr to i8 addrspace(1)*
  %tok = tail call token (i64, i32, void ()*, i32, i32, ...)
      @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["deopt" (i32 addrspace(1)* %ptr, i32 undef), "gc-live" (i32 addrspace(1)* %ptr, i8 addrspace(1)* %ptr2)]
  %a = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok, i32 0, i32 0)
  %b = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %tok, i32 1, i32 1)
  call void @use1(i32 addrspace(1)* %a, i8 addrspace(1)* %b)
  ret void
}

;
; Cross-basicblock relocates are handled with spilling for now.
; No need to check post-RA output
define i1 @test_cross_bb(i32 addrspace(1)* %a, i1 %external_cond) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_cross_bb
; CHECK-VREG:  bb.0.entry:
; CHECK-VREG:         %1:gr32 = COPY $esi
; CHECK-VREG-NEXT:    %0:gr64 = COPY $rdi
; CHECK-VREG-NEXT:    %3:gr8 = COPY %1.sub_8bit
; CHECK-VREG-NEXT:    MOV64mr %stack.0, 1, $noreg, 0, $noreg, %0 :: (store 8 into %stack.0)
; CHECK-VREG-NEXT:    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG-NEXT:    STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, 1, 8, %stack.0, 0, 1, 8, %stack.0, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def $al :: (volatile load store 8 on %stack.0)
; CHECK-VREG-NEXT:    ADJCALLSTACKUP64 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG-NEXT:    %4:gr8 = COPY $al
; CHECK-VREG-NEXT:    %2:gr8 = COPY %4
; CHECK-VREG-NEXT:    TEST8ri killed %3, 1, implicit-def $eflags
; CHECK-VREG-NEXT:    JCC_1 %bb.2, 4, implicit $eflags
; CHECK-VREG-NEXT:    JMP_1 %bb.1
; CHECK-VREG:       bb.1.left:
; CHECK-VREG-NEXT:    %6:gr64 = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load 8 from %stack.0)
; CHECK-VREG-NEXT:    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG-NEXT:    $rdi = COPY %6
; CHECK-VREG-NEXT:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG-NEXT:    ADJCALLSTACKUP64 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG-NEXT:    $al = COPY %2
; CHECK-VREG-NEXT:    RET 0, $al
; CHECK-VREG:       bb.2.right:
; CHECK-VREG-NEXT:    %5:gr8 = MOV8ri 1
; CHECK-VREG-NEXT:    $al = COPY %5
; CHECK-VREG-NEXT:    RET 0, $al

entry:
  %safepoint_token = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %a)]
  br i1 %external_cond, label %left, label %right

left:
  %call1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %call2 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  call void @consume(i32 addrspace(1)* %call1)
  ret i1 %call2

right:
  ret i1 true
}

; No need to check post-regalloc output as it is the same
define i1 @duplicate_reloc() gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            duplicate_reloc
; CHECK-VREG:  bb.0.entry:
; CHECK-VREG:    STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    %0:gr8 = MOV8ri 1
; CHECK-VREG:    $al = COPY %0
; CHECK-VREG:    RET 0, $al

; CHECK-ASM-LABEL: duplicate_reloc:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:          pushq	%rax
; CHECK-ASM-NEXT:          .cfi_def_cfa_offset 16
; CHECK-ASM-NEXT:          callq	func
; CHECK-ASM-NEXT:  .Ltmp8:
; CHECK-ASM-NEXT:          callq	func
; CHECK-ASM-NEXT:  .Ltmp9:
; CHECK-ASM-NEXT:          movb	$1, %al
; CHECK-ASM-NEXT:          popq	%rcx
; CHECK-ASM-NEXT:          .cfi_def_cfa_offset 8
; CHECK-ASM-NEXT:          retq
entry:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* null, i32 addrspace(1)* null)]
  %base = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %derived = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 1)
  %safepoint_token2 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %base, i32 addrspace(1)* %derived)]
  %base_reloc = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2,  i32 0, i32 0)
  %derived_reloc = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2,  i32 0, i32 1)
  %cmp1 = icmp eq i32 addrspace(1)* %base_reloc, null
  %cmp2 = icmp eq i32 addrspace(1)* %derived_reloc, null
  %cmp = and i1 %cmp1, %cmp2
  ret i1 %cmp
}

; Vectors cannot go in VRegs
; No need to check post-regalloc output as it is lowered using old scheme
define <2 x i8 addrspace(1)*> @test_vector(<2 x i8 addrspace(1)*> %obj) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_vector
; CHECK-VREG:    %0:vr128 = COPY $xmm0
; CHECK-VREG:    MOVAPSmr %stack.0, 1, $noreg, 0, $noreg, %0 :: (store 16 into %stack.0)
; CHECK-VREG:    STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 1, 16, %stack.0, 0, 1, 16, %stack.0, 0, csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store 16 on %stack.0)
; CHECK-VREG:    %1:vr128 = MOVAPSrm %stack.0, 1, $noreg, 0, $noreg :: (load 16 from %stack.0)
; CHECK-VREG:    $xmm0 = COPY %1
; CHECK-VREG:    RET 0, $xmm0

; CHECK-ASM-LABEL: test_vector:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:          subq	$24, %rsp
; CHECK-ASM-NEXT:          .cfi_def_cfa_offset 32
; CHECK-ASM-NEXT:          movaps	%xmm0, (%rsp)
; CHECK-ASM-NEXT:          callq	func
; CHECK-ASM-NEXT:  .Ltmp10:
; CHECK-ASM-NEXT:          movaps	(%rsp), %xmm0
; CHECK-ASM-NEXT:          addq	$24, %rsp
; CHECK-ASM-NEXT:          .cfi_def_cfa_offset 8
; CHECK-ASM-NEXT:          retq
entry:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (<2 x i8 addrspace(1)*> %obj)]
  %obj.relocated = call coldcc <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token %safepoint_token, i32 0, i32 0) ; (%obj, %obj)
  ret <2 x i8 addrspace(1)*> %obj.relocated
}


; test limit on amount of vregs
define void @test_limit(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c, i32 addrspace(1)* %d, i32 addrspace(1)*  %e) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_limit
; CHECK-VREG:    %4:gr64 = COPY $r8
; CHECK-VREG:    %3:gr64 = COPY $rcx
; CHECK-VREG:    %2:gr64 = COPY $rdx
; CHECK-VREG:    %1:gr64 = COPY $rsi
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    MOV64mr %stack.0, 1, $noreg, 0, $noreg, %0 :: (store 8 into %stack.0)
; CHECK-VREG:    %5:gr64, %6:gr64, %7:gr64, %8:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, %4, %4(tied-def 0), %3, %3(tied-def 1), %2, %2(tied-def 2), %1, %1(tied-def 3), 1, 8, %stack.0, 0, 1, 8, %stack.0, 0, csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store 8 on %stack.0)
; CHECK-VREG:    %9:gr64 = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load 8 from %stack.0)
; CHECK-VREG:    $rdi = COPY %9
; CHECK-VREG:    $rsi = COPY %8
; CHECK-VREG:    $rdx = COPY %7
; CHECK-VREG:    $rcx = COPY %6
; CHECK-VREG:    $r8 = COPY %5
; CHECK-VREG:    CALL64pcrel32 @consume5, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit $rdx, implicit $rcx, implicit $r8, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    RET 0
entry:
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c, i32 addrspace(1)* %d, i32 addrspace(1)* %e)]
  %rel1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %rel2 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 1, i32 1)
  %rel3 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 2, i32 2)
  %rel4 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 3, i32 3)
  %rel5 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 4, i32 4)
  call void @consume5(i32 addrspace(1)* %rel1, i32 addrspace(1)* %rel2, i32 addrspace(1)* %rel3, i32 addrspace(1)* %rel4, i32 addrspace(1)* %rel5)
  ret void
}

declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32)
declare <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token, i32, i32)
declare i1 @llvm.experimental.gc.result.i1(token)

; CHECK-ASM-LABEL: .section .llvm_stackmaps
; CHECK-ASM-NEXT:  __LLVM_StackMaps:
; Entry for test_relocate
; CHECK-ASM:	        .quad	0
; CHECK-ASM-NEXT:     	.long	.Ltmp0-test_relocate
; CHECK-ASM-NEXT:	.short	0
; Num locations
; CHECK-ASM-NEXT:	.short	5
; Location 1 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 2 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 3 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 4 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 5 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
;  Entry for test_mixed
; CHECK-ASM:     	.quad	0
; CHECK-ASM-NEXT:	.long	.Ltmp1-test_mixed
; CHECK-ASM-NEXT:	.short	0
; Num locations
; CHECK-ASM-NEXT:	.short	11
; Location 1 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 2 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 3 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 4 Register $r14
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	14
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 5 Register $r14
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	14
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 6 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 7 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 8 Register $r15
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	15
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 9 Register $r15
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	15
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 10 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 11 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Entry for test_alloca
; CHECK-ASM:     	.quad	0
; CHECK-ASM-NEXT:	.long	.Ltmp2-test_alloca
; CHECK-ASM-NEXT:	.short	0
; Num locations
; CHECK-ASM-NEXT:	.short	6
; Location 1 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 2 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 3 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 4 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 5 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 6 Direct $rsp + 0
; CHECK-ASM-NEXT:	.byte	2
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	7
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Entry for test_base_derive
; CHECK-ASM:     	.quad	0
; CHECK-ASM-NEXT:	.long	.Ltmp3-test_base_derived
; CHECK-ASM-NEXT:	.short	0
; Num locations
; CHECK-ASM-NEXT:	.short	5
; Location 1 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 2 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 3 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 4 Indirect $rsp + 8
; CHECK-ASM-NEXT:	.byte	3
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	7
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	8
; Location 5 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Entry for test_deopt_gcpointer
; CHECK-ASM:     	.quad	0
; CHECK-ASM-NEXT:	.long	.Ltmp4-test_deopt_gcpointer
; CHECK-ASM-NEXT:	.short	0
; Num locations
; CHECK-ASM-NEXT:	.short	6
; Location 1 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 2 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 3 Constant 1
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	1
; Location 4Indirect $rsp + 8
; CHECK-ASM-NEXT:	.byte	3
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	7
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	8
; Location 5 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 6
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Entry for test_gcrelocate_uniqueing
; CHECK-ASM:     	.quad	0
; CHECK-ASM-NEXT:	.long	.Ltmp5-test_gcrelocate_uniqueing
; CHECK-ASM-NEXT:	.short	0
; Num locations
; CHECK-ASM-NEXT:	.short	7
; Location 1 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 2 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 3 Constant 2
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	2
; Location 4 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 5 Constant Index 0
; CHECK-ASM-NEXT:	.byte	5
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 6 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 7 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Entry for test_gcptr_uniqueing
; CHECK-ASM:     	.long	.Ltmp6-test_gcptr_uniqueing
; CHECK-ASM-NEXT:	.short	0
; Num locations
; CHECK-ASM-NEXT:	.short	7
; Location 1 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 2 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 3 Constant 2
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	2
; Location 4 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 5 Constant Index 0
; CHECK-ASM-NEXT:	.byte	5
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 6 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 7 Register $rbx
; CHECK-ASM-NEXT:	.byte	1
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	3
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Entry for test_cross_bb
; CHECK-ASM:     	.quad	0
; CHECK-ASM-NEXT:	.long	.Ltmp7-test_cross_bb
; CHECK-ASM-NEXT:	.short	0
; Num locations
; CHECK-ASM-NEXT:	.short	5
; Location 1 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 2 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 3 Constant 0
; CHECK-ASM-NEXT:	.byte	4
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 4 Indirect $rsp + 0
; CHECK-ASM-NEXT:	.byte	3
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	7
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
; Location 5 Indirect $rsp + 0
; CHECK-ASM-NEXT:	.byte	3
; CHECK-ASM-NEXT:	.byte	0
; CHECK-ASM-NEXT:	.short	8
; CHECK-ASM-NEXT:	.short	7
; CHECK-ASM-NEXT:	.short	0
; CHECK-ASM-NEXT:	.long	0
