lus.node @pre(%a:i32) -> (i32) {
  %b = lus.pre %a: i32
  lus.yield(%b:i32)
}

lus.node @fby_depends_on_i(%a: i32, %b:i32) -> (i32) {
  %c = lus.fby %a %b: i32
  lus.yield(%c:i32)
}



lus.node @clocked_fby(%i: i32, %c: i1) -> (i32) {
  %j = arith.constant 0: i32
  %ic = lus.when %c %i: i32
  %k = lus.fby %j %ic: i32
  lus.yield(%k:i32)
}
