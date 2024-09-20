module  {
  func private @tick() -> i32
  func private @dumb(%arg0: i32, %arg1: (i32, memref<?xi32>) -> (), %arg2: (i32, memref<?xi32>) -> i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c0_i32_0 = arith.constant 0 : i32
    %c0_1 = arith.constant 0 : index
    %true = arith.constant true
    scf.while : () -> () {
      scf.condition(%true)
    } do {
      %false = arith.constant false
      %0 = memref.alloc() : memref<1xi32>
      %1 = memref.cast %0 : memref<1xi32> to memref<?xi32>
      call_indirect %arg1(%c0_i32_0, %1) : (i32, memref<?xi32>) -> ()
      %2 = memref.load %1[%c0_1] : memref<?xi32>
      %3 = memref.alloc() : memref<1xi32>
      %4 = memref.cast %3 : memref<1xi32> to memref<?xi32>
      memref.store %2, %4[%c0] : memref<?xi32>
      %5 = call_indirect %arg2(%c0_i32, %4) : (i32, memref<?xi32>) -> i32
      %6 = call @tick() : () -> i32
      scf.yield
    }
    return
  }
  func private @sched_read_input_memrefxi32(i32, memref<?xi32>)
  func private @sched_write_output_memrefxi32(i32, memref<?xi32>) -> i32
  func private @dumb_start(%arg0: i32) {
    %f = constant @sched_read_input_memrefxi32 : (i32, memref<?xi32>) -> ()
    %f_0 = constant @sched_write_output_memrefxi32 : (i32, memref<?xi32>) -> i32
    call @dumb(%arg0, %f, %f_0) : (i32, (i32, memref<?xi32>) -> (), (i32, memref<?xi32>) -> i32) -> ()
    return
  }
  func private @sch_set_instance(i32, (i32) -> (), i32, i32)
  func private @sched_set_input_memrefxi32(i32, i32, i32, i32, memref<?xi32>)
  func private @sched_set_output_memrefxi32(i32, i32, i32, i32, memref<?xi32>)
  func private @inst(i32)
  func private @dumb_inst(%arg0: i32, %arg1: memref<?xi32>, %arg2: memref<?xi32>) {
    %f = constant @dumb_start : (i32) -> ()
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_0 = arith.constant 1 : i32
    call @sch_set_instance(%arg0, %f, %c1_i32, %c1_i32_0) : (i32, (i32) -> (), i32, i32) -> ()
    %c0_i32 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    call @sched_set_input_memrefxi32(%arg0, %c0_i32, %c1_i32_1, %c4_i32, %arg1) : (i32, i32, i32, i32, memref<?xi32>) -> ()
    %c0_i32_2 = arith.constant 0 : i32
    %c1_i32_3 = arith.constant 1 : i32
    %c4_i32_4 = arith.constant 4 : i32
    call @sched_set_output_memrefxi32(%arg0, %c0_i32_2, %c1_i32_3, %c4_i32_4, %arg2) : (i32, i32, i32, i32, memref<?xi32>) -> ()
    call @inst(%arg0) : (i32) -> ()
    return
  }
  func private @dumb2(%arg0: i32, %arg1: (i32, memref<?xi32>) -> (), %arg2: (i32, memref<?xi32>) -> i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c0_i32_0 = arith.constant 0 : i32
    %c0_1 = arith.constant 0 : index
    %true = arith.constant true
    scf.while : () -> () {
      scf.condition(%true)
    } do {
      %c2_i32 = arith.constant 2 : i32
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c2_i32_4 = arith.constant 2 : i32
      %false = arith.constant false
      %0 = memref.alloc() : memref<1xi32>
      %1 = memref.cast %0 : memref<1xi32> to memref<?xi32>
      call_indirect %arg1(%c0_i32_0, %1) : (i32, memref<?xi32>) -> ()
      %2 = memref.load %1[%c0_1] : memref<?xi32>
      %3 = memref.alloc() : memref<1xi32>
      %4 = memref.cast %3 : memref<1xi32> to memref<?xi32>
      memref.store %2, %4[%c0_2] : memref<?xi32>
      %5 = memref.alloc() : memref<1xi32>
      %6 = memref.cast %5 : memref<1xi32> to memref<?xi32>
      call @dumb_inst(%c2_i32, %4, %6) : (i32, memref<?xi32>, memref<?xi32>) -> ()
      %7 = memref.load %6[%c0_3] : memref<?xi32>
      %8 = memref.alloc() : memref<1xi32>
      %9 = memref.cast %8 : memref<1xi32> to memref<?xi32>
      memref.store %7, %9[%c0] : memref<?xi32>
      %10 = call_indirect %arg2(%c0_i32, %9) : (i32, memref<?xi32>) -> i32
      %11 = call @tick() : () -> i32
      scf.yield
    }
    return
  }
}

