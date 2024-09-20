lus.node @test3(%a:tensor<i32>,%clk1:i1)->(tensor<i32>) {
  %b = lus.on_clock ((base,base) -> (base on %clk1)) { lus.when %clk1 %a: tensor<i32> }
  lus.yield (%b: tensor<i32>)
}

lus.node @test4(%a:tensor<i32>,%clk1:i1)->(tensor<i32>) {
  %b = lus.on_clock (base,base -> base on not %clk1) { lus.when not %clk1 %a: tensor<i32> }
  lus.yield (%b: tensor<i32>)
}
