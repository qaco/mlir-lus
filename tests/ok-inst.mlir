lus.node @dumb(%a:i32) -> (i32) {
  lus.yield(%a:i32)
}

lus.node @dumb2(%i: i32) -> (i32) {
  %k2 = lus.instance @dumb(%i):(i32)->(i32)
  lus.yield(%k2:i32)
}

// lus.node @clocked_fby(%i: i32, %c: i1) -> (i32) {
//   %j = arith.constant 0: i32
//   %ic = lus.when %c %i: i32
//   %k = lus.fby %j %ic: i32
//   %k2 = lus.instance @dumb(%k):(i32)->(i32)
//   lus.yield(%k:i32)
// }
