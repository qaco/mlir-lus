lus.node @fby_clocked_and_input_dependent(%i: i32, %j:i32, %c:i1) -> (i32) {
  %k = lus.when %c %j: i32
  %r = lus.fby %i %k: i32
  lus.yield(%r:i32)
}

// !!! Bizarrerie
// Au cycle 0 si c est fausse fby est quand même initialisé
// Non ce n'est pas bizarre car du fait de fby i a la même clock que k !!! donc fby init à undef
// lus.node  dom @fby_clocked_and_input_dependent state (%r_old:i32, %arg4:i1)(%i:i32, %j:i32, %c:i1) -> (i32) {
// %false = arith.constant false
// %true = arith.constant true
// %0 = lus.macro_merge_kp01(%arg4, %false) : i1 // false true true true
// %k = lus.when %c %j :i32 //
// %2 = lus.when not %0 %i :i32 // premier cycle
// %3 = lus.when %0 %r_old :i32 // cycle n
// %4 = lus.merge %0 %3 %2 :i32 // raw = i au premier cycle, r_old sinon
// %r = lus.when %c %4 :i32 // r si c
// %6 = lus.when not %c %4 :i32 
// %r_norm = lus.merge %c %k %6 :i32 // si c alors j, si non c alors raw (i absent au premier cycle, r_old ensuite)
// lus.yield state (%r_norm:i32, %true:i1) (%r:i32)
// }
