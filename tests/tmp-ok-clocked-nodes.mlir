lus.node_test @test3(%a:tensor<i32>,%clk1:i1)->() {
  %b = lus.when %clk1 %a: tensor<i32>
  lus.yield ()
} 

lus.node_test @test4(%a:tensor<i32>,%clk1:i1)->(%clk2: i1, %r:tensor<i32>) {
  %b = lus.when %clk1 %a: tensor<i32>
  lus.yield (%clk1:i1, %b: tensor<i32>)
} 

lus.node_test @test5(%a:tensor<i32>,%clk1:i1)->(%clk2: i1, %r:tensor<i32>)
  clock { lus.on_clock ((base,base) -> (base, base on %clk2)) } {
  %b = lus.when %clk1 %a: tensor<i32>
  lus.yield (%clk1:i1, %b: tensor<i32>)
}

lus.node_test dom @test6(%a:tensor<i32>,%clk1:i1)->(%clk2: i1, %r:tensor<i32>)
  clock { lus.on_clock ((base,base) -> (base, base on %clk2)) } {
  %b = lus.when %clk1 %a: tensor<i32>
  lus.yield (%clk1:i1, %b: tensor<i32>)
}

lus.node_test dom @test7(%a:tensor<i32>,%clk1:i1) state (%s:tensor<i32>)
  -> (%clk2: i1, %r:tensor<i32>)
  clock { lus.on_clock ((base,base) -> (base, base on %clk2)) } {
  %b = lus.when %clk1 %a: tensor<i32>
  lus.yield state (%s:tensor<i32>) (%clk1:i1, %b: tensor<i32>)
}

lus.node_test @test8(%a:tensor<i32>,%clk1:i1)->(%clk2: i1, %r:tensor<i32>)
  clock { lus.on_clock ((base,base) -> (base, base on %clk2)) } {
  %res = arith.addi %b,%i: tensor<i32>
  %b = lus.when %clk1 %a: tensor<i32>
  %i = arith.constant dense<42>:tensor<i32>
  lus.yield (%clk1:i1, %res: tensor<i32>)
}
