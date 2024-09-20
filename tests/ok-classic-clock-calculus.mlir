lus.node_test @output_clocked(%a:tensor<i32>,%clk1:i1)->(%clk2: i1, %r:tensor<i32>)
  clock { lus.on_clock_node ((base,base) -> (base, base on %clk2)) } {
  %b = lus.when %clk1 %a: tensor<i32>
  lus.yield (%clk1:i1, %b: tensor<i32>)
}

lus.node_test @unif(%a:i32,%clk1:i1)->(%clk2: i1, %r:i32)
  clock { lus.on_clock_node ((base,base) -> (base, base on %clk2)) } {
  %a1 = lus.when %clk1 %a: i32
  %b = arith.constant 2: i32
  %c = arith.addi %a1,%b: i32
  lus.yield (%clk1:i1, %c: i32)
}

lus.node_test @merge(%clk1:i1, %a: i32)->(%r:i32)
  clock { lus.on_clock_node ((base,base on %clk1) -> (base)) } {
  %b = arith.constant 2: i32
  %c = lus.merge %clk1 %a %b: i32
  lus.yield (%c: i32)
}

lus.node_test @whenmerge(%a:i32,%b:i32,%c:i1) -> (i32) {
  %a1 = lus.when %c %a: i32
  %b1 = lus.when not %c %b: i32
  %r = lus.merge %c %a1 %b1: i32
  lus.yield(%r:i32)
}

lus.node_test @unif_explicit(%a:i32,%clk1:i1)->(%clk2: i1, %r:i32)
  clock { lus.on_clock_node ((base,base) -> (base, base on %clk2)) } {
  %a1 = lus.on_clock ((base) : (base on %clk1)) { lus.when %clk1 %a: i32 }
  %b = lus.on_clock ((base) : (base on %clk1)) { arith.constant 2: i32 }
  %c = lus.on_clock ((base on %clk1) : (base on %clk1)) { arith.addi %a1,%b: i32 }
  lus.yield (%clk1:i1, %c: i32)
}

lus.node_test @merge_alt(%a: i32, %clk1:i1)->(%r:i32)
  clock { lus.on_clock_node ((base on %clk1,base) -> (base)) } {
  %b = arith.constant 2: i32
  %c = lus.merge %clk1 %a %b: i32
  lus.yield (%c: i32)
}
