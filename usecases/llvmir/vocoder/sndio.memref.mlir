// Basic sound conversion algorithms that can be written in
// MLIR (that do not depend on the soundcard).



// Copy and convert %size samples from the sound buffer to a
// specific %offset in the target buffer.
func @stereo2mono(%stereo:memref<512xi16>,%mono:memref<256xi16>) {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %2 = arith.constant 2 : index
  %size = memref.dim %mono,%0 : memref<256xi16>
  scf.for %idx=%0 to %size step %1 {
    %id = arith.muli %idx, %2 : index
    %x = memref.load %stereo[%id] : memref<512xi16>
    memref.store %x,%mono[%idx] : memref<256xi16>
  }
  return
}
// Copy and convert %size samples from a specific offset in the
// source buffer to the sound buffer
func @mono2stereo(%mono:memref<?xi16>,%stereo:memref<?xi16>) {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %2 = arith.constant 2 : index 
  %size = memref.dim %mono,%0 : memref<?xi16>
  scf.for %idx=%0 to %size step %1 {
    %x = memref.load %mono[%idx] : memref<?xi16>
    %adr0 = arith.muli %idx, %2 : index
    %adr1 = arith.addi %adr0, %1 : index
    memref.store %x,%stereo[%adr0] : memref<?xi16>
    memref.store %x,%stereo[%adr1] : memref<?xi16>
  }
  return
}

