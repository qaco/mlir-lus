lus.node_test @whenmerge(%a:i32,%b:i32,%c:i1) -> (i32) {
  %b1 = lus.when not %c %b: i32
  %r = lus.merge %c %a %b1: i32
  lus.yield(%r:i32)
}

lus.node_test @when(%a:i32,%b:i32,%c:i1) -> (i32) {
  %a1 = lus.when %c %a: i32
  %b1 = lus.when not %c %b: i32
  %r = arith.addi %a1,%b1: i32
  lus.yield(%r:i32)
}

lus.node_test @unif_explicit(%a:i32,%clk1:i1)->(%clk2: i1, %r:i32)
  clock { lus.on_clock ((base,base) -> (base, base on %clk2)) } {
  %a1 = lus.on_clock ((base,base) -> (base on %clk1)) { lus.when not %clk1 %a: i32 }
  %b = lus.on_clock (() -> (base on %clk1)) { arith.constant 2: i32 }
  %c = lus.on_clock ((base on %clk1, base on %clk1) -> (base on %clk1)) { arith.addi %a1,%b: i32 }
  lus.yield (%clk1:i1, %c: i32)
}

lus.node_test @selfdep2(%clk1:i1,%clk2:i1)->(%r:tensor<i32>)
  clock { lus.on_clock ((base on %clk2, base on %clk1) -> (base on %clk1)) } {
  lus.yield (%clk1:i1)
}
