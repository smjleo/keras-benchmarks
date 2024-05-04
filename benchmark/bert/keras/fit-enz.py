import onnx
from jaxonnxruntime import backend as jax_backend
import jax
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
 
# Use the function to load and inspect the model
model_path = "./bert-optimized.onnx"
model, input_names, output_names = load_and_inspect_model(model_path)
 
def run_model(inputs):
    backend_rep = jax_backend.BackendRep(model)
    primary_input = {key: val for key, val in inputs.items() if key == 'data'}
    outputs = backend_rep.run(primary_input)
    return outputs
 
from enzyme_ad.jax import enzyme_jax_ir, NewXLAPipeline, OldXLAPipeline, JaXPipeline
 
pipeline = JaXPipeline("""
builtin.module(
inline{default-pipeline=canonicalize max-iterations=4},
canonicalize,cse,
canonicalize,
enzyme-hlo-generate-td{
patterns=
compare_op_canon<16>;
broadcast_in_dim_op_canon<16>;
convert_op_canon<16>;
dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;
chained_dynamic_broadcast_in_dim_canonicalization<16>;
dynamic_broadcast_in_dim_all_dims_non_expanding<16>;
noop_reduce_op_canon<16>;
empty_reduce_op_canon<16>;
dynamic_reshape_op_canon<16>;
get_tuple_element_op_canon<16>;
real_op_canon<16>;
imag_op_canon<16>;
get_dimension_size_op_canon<16>;
gather_op_canon<16>;
reshape_op_canon<16>;
merge_consecutive_reshapes<16>;
transpose_is_reshape<16>;
zero_extent_tensor_canon<16>;
reorder_elementwise_and_shape_op<16>;
 
cse_broadcast_in_dim<16>;
cse_slice<16>;
cse_transpose<16>;
cse_convert<16>;
cse_pad<16>;
cse_dot_general<16>;
cse_reshape<16>;
cse_mul<16>;
cse_div<16>;
cse_add<16>;
cse_subtract<16>;
cse_min<16>;
cse_max<16>;
cse_neg<16>;
cse_concatenate<16>;
 
concatenate_op_canon<16>(1024);
select_op_canon<16>(1024);
add_simplify<16>;
sub_simplify<16>;
and_simplify<16>;
max_simplify<16>;
min_simplify<16>;
or_simplify<16>;
negate_simplify<16>;
mul_simplify<16>;
div_simplify<16>;
rem_simplify<16>;
pow_simplify<16>;
sqrt_simplify<16>;
cos_simplify<16>;
sin_simplify<16>;
noop_slice<16>;
const_prop_through_barrier<16>;
slice_slice<16>;
shift_right_logical_simplify<16>;
pad_simplify<16>;
negative_pad_to_slice<16>;
tanh_simplify<16>;
exp_simplify<16>;
slice_simplify<16>;
convert_simplify<16>;
reshape_simplify<16>;
dynamic_slice_to_static<16>;
dynamic_update_slice_elim<16>;
concat_to_broadcast<16>;
reduce_to_reshape<16>;
broadcast_to_reshape<16>;
gather_simplify<16>;
iota_simplify<16>(1024);
broadcast_in_dim_simplify<16>(1024);
convert_concat<1>;
dynamic_update_to_concat<1>;
slice_of_dynamic_update<1>;
slice_elementwise<1>;
slice_pad<1>;
dot_reshape_dot<1>;
concat_const_prop<1>;
concat_fuse<1>;
pad_reshape_pad<1>;
pad_pad<1>;
concat_push_binop_add<1>;
concat_push_binop_mul<1>;
scatter_to_dynamic_update_slice<1>;
reduce_concat<1>;
slice_concat<1>;
 
bin_broadcast_splat_add<1>;
bin_broadcast_splat_subtract<1>;
bin_broadcast_splat_div<1>;
bin_broadcast_splat_mul<1>;
reshape_iota<16>;
slice_reshape_slice<1>;
dot_general_simplify<16>;
transpose_simplify<16>;
reshape_empty_broadcast<1>;
add_pad_pad_to_concat<1>;
broadcast_reshape<1>;
 
slice_reshape_concat<1>;
slice_reshape_elementwise<1>;
slice_reshape_transpose<1>;
slice_reshape_dot_general<1>;
concat_pad<1>;
 
reduce_pad<1>;
broadcast_pad<1>;
 
zero_product_reshape_pad<1>;
mul_zero_pad<1>;
div_zero_pad<1>;
 
binop_const_reshape_pad<1>;
binop_const_pad_add<1>;
binop_const_pad_subtract<1>;
binop_const_pad_mul<1>;
binop_const_pad_div<1>;
 
slice_reshape_pad<1>;
binop_binop_pad_pad_add<1>;
binop_binop_pad_pad_mul<1>;
binop_pad_pad_add<1>;
binop_pad_pad_subtract<1>;
binop_pad_pad_mul<1>;
binop_pad_pad_div<1>;
binop_pad_pad_min<1>;
binop_pad_pad_max<1>;
 
unary_pad_push_convert<1>;
unary_pad_push_tanh<1>;
unary_pad_push_exp<1>;
 
transpose_pad<1>;
 
transpose_dot_reorder<1>;
dot_transpose<1>;
convert_convert_float<1>;
concat_to_pad<1>;
concat_appending_reshape<1>;
reshape_iota<1>;
 
broadcast_reduce<1>;
slice_dot_general<1>;
 
dot_reshape_pad<1>;
pad_dot_general<1>(1);
pad_dot_general<1>(0);
},
transform-interpreter,
enzyme-hlo-remove-transform
)""")
 
f = jax.jit(enzyme_jax_ir(pipeline_options=pipeline, jit_options={"donate_argnums":0})(run_model))
 
# Create dummy inputs based on the input names from the model
inputs = create_dummy_inputs(["data"])
print(inputs)
output = f(inputs)
print("Model output:", output)