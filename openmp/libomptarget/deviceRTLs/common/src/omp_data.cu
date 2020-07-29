//===------------ omp_data.cu - OpenMP GPU objects --------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the data objects used on the GPU device.
//
//===----------------------------------------------------------------------===//

#include "common/omptarget.h"
#include "common/device_environment.h"

////////////////////////////////////////////////////////////////////////////////
// global device environment
////////////////////////////////////////////////////////////////////////////////

#ifdef __AMDGCN__
// Keeping the variable out of bss allows it to be initialized before
// loading the device image
__attribute__((section(".data")))
#endif
DEVICE omptarget_device_environmentTy omptarget_device_environment;

////////////////////////////////////////////////////////////////////////////////
// global data holding OpenMP state information
////////////////////////////////////////////////////////////////////////////////

#ifndef __AMDGCN__

DEVICE
omptarget_nvptx_Queue<omptarget_nvptx_ThreadPrivateContext, OMP_STATE_COUNT>
    omptarget_nvptx_device_State[MAX_SM];

#else
__attribute__((used))
EXTERN uint64_t const constexpr omptarget_nvptx_device_State_size = sizeof(omptarget_nvptx_Queue<omptarget_nvptx_ThreadPrivateContext, OMP_STATE_COUNT>[MAX_SM]);

DEVICE
omptarget_nvptx_Queue<omptarget_nvptx_ThreadPrivateContext, OMP_STATE_COUNT>
    *omptarget_nvptx_device_State;

template <size_t D, size_t N>
constexpr bool is_exact_divisor() {
  return D * (N / D) == N;
}

// Uses one wavefront per block as they execute independently
using vec_t =
    __attribute__((__may_alias__)) __attribute__((ext_vector_type(4))) uint32_t;

static_assert(alignof(decltype(*omptarget_nvptx_device_State)) <=
                  alignof(vec_t),
              "");


// largest prime factor of similar order of magnitude to number of active wavefronts
__attribute__((used))
EXTERN uint64_t const constexpr init_func_number_blocks = 31;

static_assert(is_exact_divisor<sizeof(vec_t)*WARPSIZE*init_func_number_blocks, omptarget_nvptx_device_State_size>(), "");

EXTERN void __devicertl_init_func(void) {
  vec_t *base = reinterpret_cast<vec_t *>(omptarget_nvptx_device_State);

  constexpr size_t bytes = omptarget_nvptx_device_State_size;
  constexpr size_t words = bytes / sizeof(vec_t);
  static_assert(bytes == words * sizeof(vec_t), "");

  constexpr size_t blocks = init_func_number_blocks;
  constexpr size_t per_block = words / blocks;
  static_assert(words == blocks * per_block, "");

  constexpr size_t per_thread = per_block / WARPSIZE;
  static_assert(words == blocks * per_thread * WARPSIZE, "");
  
  size_t block_id = GetBlockIdInKernel();
  assert(block_id < blocks);
  vec_t *my_block = &base[block_id * per_block];

  uint64_t my_id = GetLaneId();
  for (uint64_t i = 0; i < per_thread; i++) {
    my_block[my_id] = 0;
    my_block += WARPSIZE;
  }
}
#endif

DEVICE void *omptarget_nest_par_call_stack;
DEVICE uint32_t omptarget_nest_par_call_struct_size =
    sizeof(class omptarget_nvptx_TaskDescr);

DEVICE omptarget_nvptx_SimpleMemoryManager
    omptarget_nvptx_simpleMemoryManager;
DEVICE SHARED uint32_t usedMemIdx;
DEVICE SHARED uint32_t usedSlotIdx;
DEVICE SHARED uint8_t parallelLevel[MAX_THREADS_PER_TEAM / WARPSIZE];
DEVICE SHARED uint16_t threadLimit;
DEVICE SHARED uint16_t threadsInTeam;
DEVICE SHARED uint16_t nThreads;
// Pointer to this team's OpenMP state object
DEVICE SHARED omptarget_nvptx_ThreadPrivateContext
    *omptarget_nvptx_threadPrivateContext;

////////////////////////////////////////////////////////////////////////////////
// The team master sets the outlined parallel function in this variable to
// communicate with the workers.  Since it is in shared memory, there is one
// copy of these variables for each kernel, instance, and team.
////////////////////////////////////////////////////////////////////////////////
volatile DEVICE SHARED omptarget_nvptx_WorkFn omptarget_nvptx_workFn;
#ifdef __AMDGCN__
DEVICE SHARED bool omptarget_workers_active;
DEVICE SHARED bool omptarget_master_active;
#endif

    ////////////////////////////////////////////////////////////////////////////////
    // OpenMP kernel execution parameters
    ////////////////////////////////////////////////////////////////////////////////
    DEVICE SHARED uint32_t execution_param;

    ////////////////////////////////////////////////////////////////////////////////
    // Data sharing state
    ////////////////////////////////////////////////////////////////////////////////
    DEVICE SHARED DataSharingStateTy DataSharingState;

    ////////////////////////////////////////////////////////////////////////////////
    // Scratchpad for teams reduction.
    ////////////////////////////////////////////////////////////////////////////////
    DEVICE SHARED void *ReductionScratchpadPtr;

    ////////////////////////////////////////////////////////////////////////////////
    // Data sharing related variables.
    ////////////////////////////////////////////////////////////////////////////////
    DEVICE SHARED omptarget_nvptx_SharedArgs omptarget_nvptx_globalArgs;
