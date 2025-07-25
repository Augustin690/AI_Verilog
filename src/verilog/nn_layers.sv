module nn_layers #(
    parameter N_INPUTS = 4,
    parameter N_OUTPUTS = 3
)(
    input  logic signed [15:0] input_vec [N_INPUTS],
    input  logic signed [15:0] weights   [N_INPUTS][N_OUTPUTS],
    output logic signed [15:0] output_vec[N_OUTPUTS]
);
    integer i, j;
    always_comb begin
        for (j = 0; j < N_OUTPUTS; j++) begin
            output_vec[j] = 0;
            for (i = 0; i < N_INPUTS; i++)
                output_vec[j] += input_vec[i] * weights[i][j];
        end
    end
endmodule

module relu #(
    parameter N = 3
)(
    input  logic signed [15:0] in_vec [N],
    output logic signed [15:0] out_vec[N]
);
    integer i;
    always_comb begin
        for (i = 0; i < N; i++)
            out_vec[i] = (in_vec[i] > 0) ? in_vec[i] : 0;
    end
endmodule

module argmax #(
    parameter N = 3
)(
    input  logic signed [15:0] in_vec [N],
    output logic [$clog2(N)-1:0] pred
);
    integer i;
    logic signed [31:0] max_val;
    always_comb begin
        max_val = -32768;
        pred = 0;
        for (i = 0; i < N; i++) begin
            if (in_vec[i] > max_val) begin
                max_val = in_vec[i];
                pred = i;
            end
        end
    end
endmodule
