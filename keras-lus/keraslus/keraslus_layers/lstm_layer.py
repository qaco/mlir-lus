from keraslus.utilities.op_base import Identity
from keraslus.utilities import lit
from keraslus.simple_ops.constant import Constant
from keraslus.simple_ops.arith_ops import Add
from keraslus.simple_ops.arith_ops import Mul
from keraslus.simple_ops.arith_ops import LessEqual
from keraslus.simple_ops.arith_ops import FloorMod
from keraslus.simple_ops.arith_ops import matmul
from keraslus.simple_ops.split import split
from keraslus.simple_ops.split import StridedSlice
from keraslus.mlir_lus.FbyOp import FbyOp
from keraslus.mlir_lus.WhenOp import WhenOp
from keraslus.mlir_tensor.ExtractOp import ExtractOp
from keraslus.utilities.weight import Weight
from keraslus.simple_ops.arith_ops import WithBias
from keraslus.simple_ops.activation import Activation
from keraslus.simple_ops.arith_ops import biasadd
from keraslus.simple_ops.control import Select

class LSTM(Identity, WithBias):
    count = -1
    def __init__(
            self,
            units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            kernel_initializer=lit.INIT_GLOROT,
            recurrent_initializer=lit.INIT_ORTHOGONAL,
            bias_initializer=lit.INIT_ZEROS,
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            time_major=False,
            unroll=False,
            name = "lstm",
            **kwargs):
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        init_shape = lit.POLY_SHAPE_1D + (self.units,)
        Identity.__init__(self,init_shape = init_shape)

        # The time clock
        
        self.time_incr = Constant(value=1, dtype="i32", shape=())
        self.time_steps = Constant(value=-1, dtype="i32", shape=())
        self.time_init = Constant(value=0, dtype="i32", shape=())
        self.time_fby = FbyOp(dtype="i32", init_shape=())
        self.time_plusplus = Add(dtype="i32", init_shape=())
        self.time_mod = FloorMod(dtype="i32", init_shape=())
        self.time_floor = Constant(value=-1, dtype="i32", shape=())
        self.time_less = LessEqual()
        self.time_clk = ExtractOp(dtype="i1", init_shape=())

        # The weights

        LSTM.count += 1
        if LSTM.count > 0:
            cell = "lstm_cell_" + str(LSTM.count)
        else:
            cell = "lstm_cell"
        
        self.kernel_w = Weight(value = None,
                               p_name = name,
                               k_name = cell + "/kernel:0",
                               initf = kernel_initializer,
                               # must depends on input shape
                               shape = (lit.POLY_DIM, units*4))
        self.rec_kernel_w = Weight(value = None,
                                   p_name = name,
                                   k_name = cell + "/recurrent_kernel:0",
                                   initf = recurrent_initializer,
                                   shape = (units, units*4))
        self.bias = Weight(value = None,
                           p_name = name,
                           k_name = cell + "/bias:0",
                           initf = bias_initializer,
                           shape = (units*4,))

        # Split things

        self.bias_split = split(num_or_size_splits=4, axis=0)
        self.kern_split = split(num_or_size_splits=4, axis=1)
        self.rec_part0 = StridedSlice(begin=[0,0],
                                      end = [0,self.units],
                                      strides=[1,1],
                                      begin_mask = 3,
                                      end_mask = 1)
        self.rec_part1 = StridedSlice(begin=[0,self.units],
                                      end = [0,self.units*2],
                                      strides=[1,1],
                                      begin_mask = 1,
                                      end_mask = 1)
        self.rec_part2 = StridedSlice(begin=[0,self.units*2],
                                      end = [0,self.units*3],
                                      strides=[1,1],
                                      begin_mask = 1,
                                      end_mask = 1)
        self.rec_part3 = StridedSlice(begin=[0,self.units*3],
                                      end = [0,0],
                                      strides=[1,1],
                                      begin_mask = 1,
                                      end_mask = 3)
        
        # The reccurrence
        self.state0_init = Constant(value=0.0, dtype="f32",
                                    shape=(lit.POLY_DIM, units))
        self.state0_fby = FbyOp(dtype="f32",
                                init_shape=(lit.POLY_DIM, units))
        self.state0_select = Select()
        self.state1_init = Constant(value=0.0, dtype="f32",
                                    shape=(lit.POLY_DIM, units))
        self.state1_fby = FbyOp(dtype="f32",
                                init_shape=(lit.POLY_DIM, units))
        self.state1_select = Select()

        # Matmuls rec kernel & state0

        self.matmul_rec0 = matmul(dtype=None, shape=(lit.POLY_DIM, units))
        self.matmul_rec1 = matmul(dtype=None, shape=(lit.POLY_DIM, units))
        self.matmul_rec2 = matmul(dtype=None, shape=(lit.POLY_DIM, units))
        self.matmul_rec3 = matmul(dtype=None, shape=(lit.POLY_DIM, units))
                                  
        # LSTM core

        # recurrent activations
        
        self.matmul_kern0 = matmul(dtype=None, shape=(lit.POLY_DIM, units))
        self.bias_add0 = biasadd(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.add0 = Add(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.rec_act0 = Activation(recurrent_activation)

        self.matmul_kern1 = matmul(dtype=None, shape=(lit.POLY_DIM, units))
        self.bias_add1 = biasadd(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.add1 = Add(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.rec_act1 = Activation(recurrent_activation)

        self.matmul_kern3 = matmul(dtype=None, shape=(lit.POLY_DIM, units))
        self.bias_add3 = biasadd(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.add3 = Add(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.rec_act3 = Activation(recurrent_activation)

        # activation

        self.matmul_kern2 = matmul(dtype=None, shape=(lit.POLY_DIM, units))
        self.bias_add2 = biasadd(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.add2 = Add(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.act2 = Activation(activation)

        # new states & output production

        self.mul0 = Mul(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.mul1 = Mul(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.prod_state1 = Add(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.act = Activation(activation)
        self.prod_state0 = Mul(dtype=None, init_shape=(lit.POLY_DIM, units))
        self.when = WhenOp(dtype=None, init_shape=(lit.POLY_DIM,units))
                        
    def __call__(self, x):

        if len(x.shape) == 2:
            ntime = x.shape[0]
            x.shape = (1, x.shape[-1])
        elif len(x.shape) == 3:
            ntime = x.shape[0]
            x.shape = (x.shape[1], x.shape[2])
        else:
            assert(False)
        batch,seq = x.shape

        # The time clock
        
        self.time_steps.update(value=ntime, shape=())
        myTime = self.time_fby(init=self.time_init.res,
                               next_cycle=self.time_plusplus.res)
        myIncr = self.time_plusplus([myTime, self.time_incr.res])
        myMod = self.time_mod(myIncr, self.time_steps.res)
        self.time_floor.update(value = ntime -1, shape=())
        myTest = self.time_less(myMod, self.time_floor.res)
        myClock = self.time_clk(myTest)

        # The weights
        
        self.kernel_w.initialize(x.elt_type, (seq,self.units*4))
        self.rec_kernel_w.initialize(x.elt_type, (self.units,self.units*4))
        self.bias.initialize(x.elt_type, (self.units*4,))

        # Split things

        b0, b1, b2, b3 = self.bias_split(self.bias.res)
        k0, k1, k2, k3 = self.kern_split(self.kernel_w.res)
        rk0 = self.rec_part0(self.rec_kernel_w.res)
        rk1 = self.rec_part1(self.rec_kernel_w.res)
        rk2 = self.rec_part2(self.rec_kernel_w.res)
        rk3 = self.rec_part3(self.rec_kernel_w.res)
        
        # The reccurrence
        
        self.state0_init.update(value=0.0, shape=(batch, self.units))
        state0_tmp = self.state0_fby(init=self.state0_init.res,
                                     next_cycle=self.prod_state0.res)
        state0 = self.state0_select(myTest, state0_tmp, self.state0_init.res)
        self.state1_init.update(value=0.0, shape=(batch, self.units))
        state1_tmp = self.state1_fby(init=self.state1_init.res,
                                     next_cycle=self.prod_state1.res)
        state1 = self.state1_select(myTest, state1_tmp, self.state1_init.res)

        # Matmuls rec kernel & state0

        mrk0 = self.matmul_rec0(state0, rk0)
        mrk1 = self.matmul_rec1(state0, rk1)
        mrk2 = self.matmul_rec2(state0, rk2)
        mrk3 = self.matmul_rec3(state0, rk3)

        # LSTM core

        # recurrent activations
        
        mk0 = self.matmul_kern0(x, k0)
        ba0 = self.bias_add0(mk0, b0)
        a0 = self.add0([ba0,mrk0])
        
        ract0 = self.rec_act0(a0)

        mk1 = self.matmul_kern1(x, k1)
        ba1 = self.bias_add1(mk1, b1)
        a1 = self.add1([ba1,mrk1])
        
        ract1 = self.rec_act1(a1)

        mk3 = self.matmul_kern3(x, k3)
        ba3 = self.bias_add3(mk3, b3)
        a3 = self.add3([ba3,mrk3])
        
        ract3 = self.rec_act3(a3)

        # activation

        mk2 = self.matmul_kern2(x, k2)
        ba2 = self.bias_add2(mk2, b2)
        a2 = self.add2([ba2,mrk2])
        
        act2 = self.act2(a2)

        # new states & output production

        v0 = self.mul0([ract0,act2])
        v1 = self.mul1([ract1,state1])
        nstate1 = self.prod_state1([v1,v0])
        act = self.act(nstate1)
        nstate0 = self.prod_state0([ract3,act])
        clocked_res = self.when(myClock, nstate0)
        clocked_res = Identity.__call__(self, clocked_res)
        
        return clocked_res

class LSTM2(Identity, WithBias):
    def __init__(
            self,
            units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            kernel_initializer=lit.INIT_GLOROT,
            recurrent_initializer=lit.INIT_ORTHOGONAL,
            bias_initializer=lit.INIT_ZEROS,
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            time_major=False,
            unroll=False,
            name = "lstm",
            **kwargs):
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        init_shape = lit.POLY_SHAPE_1D + (self.units,)
        Identity.__init__(self,init_shape = init_shape)
        WithBias.__init__(self, use_bias, (self.units*4,),
                          bias_initializer,
                          bias_regularizer, bias_constraint,
                          name,
                          "lstm_cell/bias:0")

        # The time clock
        self.time_incr = Constant(value=1, dtype="i32", shape=())
        self.time_steps = Constant(value=-1, dtype="i32", shape=())
        self.time_init = Constant(value=0, dtype="i32", shape=())
        self.time_fby = FbyOp(dtype="i32", init_shape=())
        self.time_plusplus = Add(dtype="i32", init_shape=())
        self.time_mod = FloorMod(dtype="i32", init_shape=())
        self.time_less = LessEqual()
        self.time_clk = ExtractOp(dtype="i1", init_shape=())

        # The weights
        self.kernel_w = Weight(value = None,
                               p_name = name,
                               k_name = "lstm_cell/kernel:0",
                               initf = kernel_initializer,
                               # must depends on input shape
                               shape = (lit.POLY_DIM, units*4))
        self.rec_kernel_w = Weight(value = None,
                                   p_name = name,
                                   k_name = "lstm_cell/recurrent_kernel:0",
                                   initf = recurrent_initializer,
                                   shape = (units, units*4))

        # The reccurrence
        self.state0_init = Constant(value=0.0, dtype="f32",
                                    shape=(lit.POLY_SHAPE_1D,)+(units,))
        self.state0_fby = FbyOp(dtype="f32",
                                init_shape=(lit.POLY_SHAPE_1D,)+(units,))
        self.state0_select = Select()
        self.state1_init = Constant(value=0.0, dtype="f32",
                                    shape=(lit.POLY_SHAPE_1D,)+(units,))
        self.state1_fby = FbyOp(dtype="f32",
                                init_shape=(lit.POLY_SHAPE_1D,)+(units,))
        self.state1_select = Select()
        
        # LSTM core
        self.matmul_rec_kernel = matmul(dtype=None,
                                        shape=self.kernel_w.res.shape)
        self.matmul_kernel = matmul(dtype=None,
                                    shape=self.kernel_w.res.shape)
        self.add_kernels = Add(dtype=None, init_shape=self.kernel_w.res.shape)
        self.split = split(4,1)
        self.act1 = Activation(activation)
        self.rec_act1 = Activation(recurrent_activation)
        self.mul1 = Mul(dtype=None, init_shape=(lit.POLY_SHAPE_1D,)+(units,))
        self.rec_act2 = Activation(recurrent_activation)
        self.mul2 = Mul(dtype=None, init_shape=(lit.POLY_SHAPE_1D,)+(units,))
        self.add_out=Add(dtype=None,init_shape=(lit.POLY_SHAPE_1D,)+(units,))
        self.state1_act = Activation(activation)
        self.rec_act3 = Activation(recurrent_activation)
        self.mul3 = Mul(dtype=None, init_shape=(lit.POLY_SHAPE_1D,)+(units,))
        self.when = WhenOp(dtype=None,
                           init_shape=(lit.POLY_SHAPE_1D,)+(units,))
        
    def __call__(self, x):

        if len(x.shape) != 3:
            print(x.shape)
            assert(False)
        ntime = x.shape[1]
        x.shape = (x.shape[0], x.shape[2])
        a,b = x.shape
                           
        ntime = x.shape[0]
        x.shape = (1, x.shape[1])
        batch,seq = x.shape

        # The time clock
        self.time_steps.update(value=ntime, shape=())
        myTime = self.time_fby(init=self.time_init.res,
                               next_cycle=self.time_plusplus.res)
        myIncr = self.time_plusplus([myTime, self.time_incr.res])
        myMod = self.time_mod(myIncr, self.time_steps.res)
        myTest = self.time_less(myMod, self.time_steps.res)
        myClock = self.time_clk(myTest)

        # The weights
        self.kernel_w.initialize(x.elt_type, (seq,self.units*4))
        self.rec_kernel_w.initialize(x.elt_type, (self.units,self.units*4))

        # The reccurrence
        self.state0_init.update(value=0.0, shape=(b, self.units))
        state0_tmp = self.state0_fby(init=self.state0_init.res,
                                     next_cycle=self.mul3.res)
        state0 = self.state0_select(myTest, state0_tmp, self.state0_init.res)
        self.state1_init.update(value=0.0, shape=(b, self.units))
        state1_tmp = self.state1_fby(init=self.state1_init.res,
                                     next_cycle=self.mul2.res)
        state1 = self.state1_select(myTest, state1_tmp, self.state1_init.res)
        
        # LSTM core
        v5 = self.matmul_rec_kernel(state0, self.rec_kernel_w.res)
        v10 = self.matmul_kernel(x, self.kernel_w.res)
        v11 = self.add_kernels([v5,v10])
        myBias = self.apply_biasadd(v11)
        s0, s1, s2, s3 = self.split(myBias)
        v14 = self.act1(s2)
        v15 = self.rec_act1(s0)
        v16 = self.mul1([v15,v14])
        v17 = self.rec_act2(s1)
        v18 = self.mul2([v17,state1])
        state1_out = self.add_out([v18,v16])
        v21 = self.state1_act(state1_out)
        v22 = self.rec_act3(s3)
        state0_out = self.mul3([v22,v21])

        clocked_res = self.when(myClock, state1_out)
        clocked_res = Identity.__call__(self, clocked_res)
        
        return clocked_res     
