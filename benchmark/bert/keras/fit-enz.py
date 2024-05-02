import onnx
from jaxonnxruntime import backend as jax_backend
import numpy as np

def load_and_inspect_model(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    input_names = [input.name for input in model.graph.input]
    output_names = [output.name for output in model.graph.output]
    print("Input Names:", input_names)
    print("Output Names:", output_names)

    return model, input_names, output_names

def create_dummy_inputs(input_names, batch_size=16, seq_length=64, hidden_dims=1024):
    inputs = {name: np.random.rand(batch_size * seq_length, hidden_dims) for name in input_names}
    return inputs

def run_model(model, inputs):
    backend_rep = jax_backend.BackendRep(model)
    primary_input = {key: val for key, val in inputs.items() if key == 'data'}
    outputs = backend_rep.run(primary_input)
    return outputs


# Use the function to load and inspect the model
model_path = "./bert.onnx"
model, input_names, output_names = load_and_inspect_model(model_path)

# Create dummy inputs based on the input names from the model
inputs = create_dummy_inputs(["data"])
print(inputs)
output = run_model(model, inputs)
print("Model output:", output)

