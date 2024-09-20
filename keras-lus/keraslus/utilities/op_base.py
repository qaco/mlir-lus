from keraslus.utilities.value import Value
from keraslus.utilities import lit

class Op:
    def __init__(self,
                 dtype=None,
                 init_shape = lit.POLY_SHAPE_XD,
                 tail_res = [],
                 forced_res = None):
        if forced_res == None:
            self.res = Value(dtype, init_shape, self)
        else:
            self.res = forced_res
            self.res.producer = self

        self.tail_res = ()
        for tr in tail_res:
            self.tail_res += (Value(tr[0], tr[1], tr[2]),)
        self.attrs = {}

    def res_string(self):
        myRes = ""
        assert(self.res != None or len(self.tail_res) > 0)
        if self.res == None and len(self.tail_res) > 0:
            first = self.tail_res[0]
            tail = self.tail_res[1:]
        elif self.res != None:
            first = self.res
            tail = self.tail_res
            
        myRes += first.mlir_name
        for r in tail:
            myRes += ", " + r.mlir_name
        return myRes
        
    def args_string(self):
        myArgs = "("
        if (len(self.args) > 0):
            myArgs += self.args[0].mlir_name
            for a in self.args[1:]:
                myArgs += ", " + a.mlir_name
        myArgs += ")"
        return myArgs

    def attrs_string(self):
        myAttrs = " {"
        i = 0
        for kv in self.attrs.items():
            myAttrs += kv[0] + " = " + kv[1]
            if i != (len(self.attrs.items()) - 1):
                myAttrs += ", "
            i += 1
        myAttrs += "}"
        return myAttrs

    def sig_string(self):
        assert(self.res != None or len(self.tail_res) > 0)
        if self.res == None and len(self.tail_res) > 0:
            first = self.tail_res[0]
            tail = self.tail_res[1:]
        elif self.res != None:
            first = self.res
            tail = self.tail_res

        mySig = ""
        if (len(tail) > 0):
            mySig += "("
            
        mySig += first.shape_str()
        
        for r in tail:
            mySig += ", " + r.shape_str()

        if (len(tail) > 0):
            mySig += ")"

        return mySig
        
    
    def set_attr(self, key, value):
        self.attrs[key] = value

class TfOp(Op):
    def __init__(self,
                 dtype=None,
                 init_shape = lit.POLY_SHAPE_XD,
                 tail_res = [],
                 forced_res = None):
        super().__init__(dtype, init_shape, tail_res, forced_res)

    def __str__(self):
        myOp = self.res_string() + " = " + self.mlir_name
        myArgs = self.args_string()
        myAttrs = self.attrs_string()
        mySig = " : " + self.sig_string()
        return myOp + myArgs + myAttrs + mySig

    def sig_string(self):
        mySig = "("
        if (len(self.args) > 0):
            mySig += self.args[0].shape_str()
            for a in self.args[1:]:
                mySig += ", " + a.shape_str()
        mySig += ") -> "
        mySig += super().sig_string()
        return mySig

class Identity(TfOp):
    def __init__(self,
                 dtype=None,
                 init_shape=lit.POLY_SHAPE_XD,
                 ignore = True):
        self.mlir_name = "\"tf.Identity\""
        self.ignore = ignore
        TfOp.__init__(self, dtype=dtype,init_shape=init_shape)

    def __call__(self, x):
        self.args = (x,)
        if self.ignore:
            self.res = x
        else:
            self.res.shape = x.shape
            self.res.elt_type = x.elt_type
        return self.res
