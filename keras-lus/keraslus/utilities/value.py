class Value:
    count = 0

    def __init__(self, dtype=None, shape=None, producer = None):
        self.producer = producer
        self.shape = shape
        self.mlir_name = "%v" + str(Value.count)
        self.set_dtype(dtype)
        Value.count = Value.count + 1

    def set_dtype(self, dtype):
        if dtype == None or dtype == "f32":
            self.elt_type = "f32"
        elif dtype == "tf.int32" or dtype == "i32":
            self.elt_type = "i32"
        elif dtype == "i1":
            self.elt_type = "i1"
        else:
            assert(False)

    def read_dim(self, s):
        if s == -1:
            return "?"
        elif s == -2:
            return "*"
        else:
            return str(s)
            
    def shape_str(self):
        myStr = "tensor<"
        if len(self.shape) > 0:
            myStr += self.read_dim(self.shape[0])
            for s in self.shape[1:]:
                myStr += "x" + self.read_dim(s)
            myStr += "x"
        myStr += str(self.elt_type) + ">"
        return myStr
        
    def __str__(self):
     return self.mlir_name
