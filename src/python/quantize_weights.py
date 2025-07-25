import ast

def read_weights(txt_file):
    with open(txt_file, 'r') as f:
        data = f.read()
    data = data.replace("{", "[").replace("}", "]")

    return ast.literal_eval(data)



layer_3_weights = read_weights("layer_3_weights.txt")
layer_2_weights = read_weights("layer_2_weights.txt")
layer_1_weights = read_weights("layer_1_weights.txt")



def float_to_fixed(val, frac_bits=8):
    fixed = int(round(val * (1 << frac_bits)))
    # Clamp to 16-bit signed
    fixed = max(-32768, min(32767, fixed))
    return fixed

def convert_weights(weights, name):
    sv = f"logic signed [15:0] {name} [][{len(weights[0])}] = '{{\n"
    for row in weights:
        sv_row = ', '.join([f"16'sd{float_to_fixed(w)}" for w in row])
        sv += f"    '{{{sv_row}}},\n"
    sv += "};\n"
    return sv

def save_verilog(weights, name):
    with open(f"{name}.sv", "w") as f:
        f.write(convert_weights(weights, name))

save_verilog(layer_1_weights, "weights1")
save_verilog(layer_2_weights, "weights2")
save_verilog(layer_3_weights, "weights3")

print(convert_weights(layer_1_weights, "weights1"))
print(convert_weights(layer_2_weights, "weights2"))
print(convert_weights(layer_3_weights, "weights3"))