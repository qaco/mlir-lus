// mlirlus ok-instance-test.mlir --all-fbys-on-base-clock --fbys-centralization --explicit-signals --recompute-order --explicit-clocks --scf-clocks --node-to-reactive-func --sync-to-std


lus.node_test @main(%i: i32) -> (i32) {
  %c = arith.constant 0: i1
  %i1 = lus.when %c %i: i32
  %r2 = lus.instance_test @merge(%c,%i1): (i1, i32) -> i32
  lus.yield(%r2: i32)
}

lus.node_test @merge(%clk1:i1, %a: i32)->(%r:i32)
  clock { lus.on_clock_node ((base,base on %clk1) -> (base)) } {
  %b = arith.constant 2: i32
  %c = lus.merge %clk1 %a %b: i32
  lus.yield (%c: i32)
}
