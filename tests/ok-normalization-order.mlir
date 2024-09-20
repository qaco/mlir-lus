lus.node @reorder(%a:i32) -> (i32) {
  %c = arith.addi %a,%b:i32
  %b = arith.constant 1: i32
  lus.yield (%c:i32)
}
