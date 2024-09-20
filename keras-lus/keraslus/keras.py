from keraslus.utilities.value import Value
from keraslus.utilities.weight import Weight
import h5py
import numpy
import sys

# batch size ignored for now
class Input(Value):
    def __init__(
            self,
            shape,
            batch_size=None,
            name=None,
            dtype=None,
            sparse=None,
            tensor=None,
            ragged=None,
            type_spec=None,
            **kwargs
    ):
        if batch_size == None:
            super().__init__(dtype, shape)
        else:
            super().__init__(dtype, (batch_size,) + shape)
            
class Model:
    def __init__(self, inputs, outputs, name=None,
                 debug = False, **kwargs):
        self.weights_prefix = ""
        self.inputs = inputs
        self.outputs = outputs
        self.name = name
        self.debug = debug

    def to_string_aux(self, layer, layers_done):
        res_string = ""
        for arg in layer.args:
            if (arg.producer != None
                and arg.producer not in layers_done):
                layers_done.append(arg.producer)
                deeper_string = self.to_string_aux(arg.producer,
                                                   layers_done)
                res_string += deeper_string
        return res_string + str(layer) + "\n"

    def __str__(self):
        if self.name == None:
            name = "@model"
        else:
            name = "@" + self.name
        signature = "lus.node " + name + "("
        signature += str(self.inputs) + ": " + str(self.inputs.shape_str())
        signature += ") -> ("
        signature += str(self.outputs.shape_str())
        signature += ") {\n\n"

        body = self.to_string_aux(self.outputs.producer, [])

        terminator = "lus.yield("
        terminator += str(self.outputs) + ": " + str(self.outputs.shape_str())
        terminator += ")\n}"
        
        return "\n" + signature + body + terminator + "\n"
    
    def load_weights_aux(self, h5_weights, layer, layers_done):
        for arg in layer.args:
            
            if (arg.producer != None
                and arg.producer not in layers_done):
                
                layers_done.append(arg.producer)
                self.load_weights_aux(h5_weights, arg.producer,
                                      layers_done)
                
                if isinstance(arg.producer, Weight) :
                    
                    p_name = arg.producer.p_name
                    k_name = arg.producer.k_name
                    if self.debug:
                        print("Loading " + self.weights_prefix
                              +p_name+'/'+p_name+'/'+k_name+':',
                              file=sys.stderr)
                    val = h5_weights[self.weights_prefix + p_name+'/'+p_name+'/'
                                     +k_name][:]
                    
                    if self.debug:
                        print('shape inferred: ' + str(arg.shape))
                        print('shape loaded: ' + str(val.shape))
                        
                    arg.producer.load(val)
                        

    def load_weights(self, h5_file_path):
        f = h5py.File(h5_file_path)
        self.load_weights_aux(f, self.outputs.producer, [])
