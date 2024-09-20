lus.node @kp_singleton() -> (i1) {
  %b = lus.kperiodic "(1)"
  lus.yield(%b:i1)
}

lus.node @kp_headless() -> (i1) {
  %b = lus.kperiodic "(101)"
  lus.yield(%b:i1)
}

lus.node @kp_pair() -> (i1) {
  %b = lus.kperiodic "0(1)"
  lus.yield(%b:i1)
}

lus.node @complex_kp(%a:i32)->(i1) {
  %b = lus.kperiodic "00(101)"
  lus.yield(%b:i1)
}

lus.node @kp_double_complex_kp() -> (i1,i1) {
  %b = lus.kperiodic "00(10)"
  %c = lus.kperiodic "01(01)"
  lus.yield(%b:i1,%c:i1)
}
