lus.node @test1(%a:tensor<i32>,%b:tensor<i32>)->(tensor<i32>) {
  %c = lus.on_clock (base) { arith.addi %a,%b: tensor<i32> }
  lus.yield (%c: tensor<i32>)
}

lus.node @test2(%a:tensor<i32>,%b:tensor<i32>, %clk1:i1)->(tensor<i32>) {
  %c = lus.on_clock (base on %clk1) { arith.addi %a,%b: tensor<i32> }
  lus.yield (%c: tensor<i32>)
}

lus.node @test3(%a:tensor<i32>,%b:tensor<i32>, %clk1:i1, %clk2:i1)->(tensor<i32>) {
  %c = lus.on_clock (base on %clk1 on not %clk2) { arith.addi %a,%b: tensor<i32> }
  %d = lus.on_clock (base on %clk1 on not %clk2) { "tf.AddV2"(%a,%c): (tensor<i32>,tensor<i32>) -> tensor<i32> }
  lus.yield (%d: tensor<i32>)
}

lus.node @test4(%a:tensor<i32>,%b:tensor<i32>, %clk1:i1, %clk2: i1)->(tensor<i32>) {
  %c = lus.on_clock (base on not %clk1 on %clk2) { arith.addi %a,%b: tensor<i32> }
  %d = lus.on_clock (base on not %clk1 on %clk2) { "tf.AddV2"(%a,%c): (tensor<i32>,tensor<i32>) -> tensor<i32> }
  lus.yield (%d: tensor<i32>)
}
