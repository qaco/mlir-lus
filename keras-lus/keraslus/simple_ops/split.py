from keraslus.utilities.op_base import TfOp
from keraslus.simple_ops.constant import Constant
from keraslus.utilities.value import Value
from keraslus.utilities import lit

class split(TfOp):
    def __init__(self, num_or_size_splits, axis=0):
        super().__init__()
        self.res = None
        self.axis = axis
        self.const_axis = Constant(value=axis, dtype="i32", shape=())
        self.mlir_name = "\"tf.Split\""
        self.splits = num_or_size_splits

    def __call__(self, x):
        l_shape = list(x.shape)
        l_shape[self.axis] = l_shape[self.axis]//self.splits
        split_shape = tuple(l_shape)
        self.args = (self.const_axis.res, x)
        self.tail_res = ()
        for i in range(self.splits):
            self.tail_res += (Value(x.elt_type, split_shape, self),)
        return self.tail_res

# TODO masks
class StridedSlice(TfOp):
    def __init__(self,
                 begin,
                 end,
                 strides,
                 begin_mask=0,
                 end_mask=0,
                 ellipsis_mask=0,
                 new_axis_mask=0,
                 shrink_axis_mask=0,
                 var=None,
                 name=None):
        super().__init__()
        self.begin = begin
        self.end = end
        self.strides = strides
        self.mlir_name = "\"tf.StridedSlice\""
        self.set_attr(lit.BEGIN_MASK,
                      (repr(begin_mask) + ": i64"))
        self.set_attr(lit.ELLIPSIS_MASK,
                      (repr(ellipsis_mask) + ": i64"))
        self.set_attr(lit.NEW_AXIS_MASK,
                      (repr(new_axis_mask) + ": i64"))
        self.set_attr(lit.END_MASK,
                      (repr(end_mask) + ": i64"))
        self.set_attr(lit.SHRINK_AXIS_MASK,
                      (repr(shrink_axis_mask) + ": i64"))
                      

    def __call__(self,
                 input_):
        self.const_begin = Constant(value=self.begin, dtype="i32",
                                    shape=(len(input_.shape),))
        self.const_end = Constant(value=self.end, dtype="i32",
                                  shape=(len(input_.shape),))
        self.const_strides = Constant(value=self.strides, dtype="i32",
                                      shape=(len(input_.shape),))
        
        self.args = (input_,
                     self.const_begin.res,
                     self.const_end.res,
                     self.const_strides.res)

        self.res.shape = ()
        for d1,d2,d3 in zip(self.begin,
                            self.end,
                            input_.shape):
            if d1==0 and d2 == 0:
                self.res.shape += (d3,)
            elif d2 == 0:
                self.res.shape += (d3-d1,)
            else:
                self.res.shape += (d2-d1,)
        return self.res
        
        
