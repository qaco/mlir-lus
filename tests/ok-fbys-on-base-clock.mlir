// mlirlus ok-fbys-on-base-clock.mlir --all-fbys-on-base-clock --fbys-centralization --explicit-signals --recompute-order --explicit-clocks --scf-clocks --node-to-reactive-func --sync-to-std

lus.node_test @clocked_fby(%i: i32, %c: i1) -> (i32) {
  %j = arith.constant 0: i32
  %ic = lus.when %c %i: i32
  %k = lus.fby %j %ic: i32
  lus.yield(%k:i32)
}
