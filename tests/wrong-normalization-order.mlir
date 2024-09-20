lus.node @reorder_wrong(%a:i32) -> (i32) {
  %c = arith.addi %a,%b:i32
  %b = arith.addi %a,%c: i32
  lus.yield (%c:i32)
}
