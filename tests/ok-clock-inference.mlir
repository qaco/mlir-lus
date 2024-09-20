lus.node @whenmerge(%a:i32,%b:i32,%c:i1) -> (i32) {
  %a1 = lus.when %c %a: i32
  %b1 = lus.when not %c %b: i32
  %r = lus.merge %c %a1 %b1: i32
  lus.yield(%r:i32)
}

lus.node @clockedoutput(%a:i32,%c:i1) -> (i32) {
  %r = lus.when %c %a: i32
  lus.yield(%r:i32)
}

lus.node @clockedinput(%a:i32,%b:i32,%c:i1) -> (i32) {
  %a1 = lus.when %c %a: i32
  %r = arith.addi %a1,%b:i32
  lus.yield(%r:i32)
}

func private @clk()->(i1)

lus.node @clockedclock(%a:i32,%b:i32,%c1:i1) -> (i32) {
  %c2 = call @clk():()->(i1)
  %a1 = lus.when %c1 %a: i32
  %r = lus.when %c2 %a1: i32
  lus.yield(%r:i32)
}
