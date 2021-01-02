inline double sigmoid (double a) { return 1.0 / (1.0 + exp (-a)); }
inline int idx_2d(int x, int y, int width) { return x*width+y; }
inline float dp4a_fp32(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
#define TILE_SIZE 8
__kernel void lstm_gemm(__global float *input,
                          __global float *initial_hiddens,
                          __global float *weights,
                          __global float *bias,
                          __global float *out_gates,
                          int M, int K, int N,
                          int input_size, int hidden_size, int timestep) {
    __local float4 inputs_sm[TILE_SIZE * TILE_SIZE];
    __local float4 weights_sm[TILE_SIZE * TILE_SIZE];
    int lid_x = get_local_id(0);
    int lid_y = get_local_id(1);
    int m = lid_x + get_group_id(0) * get_local_size(0);
    int n = lid_y + get_group_id(1) * get_local_size(1);
    //if (!(blockIdx.x == 0 && blockIdx.y == 0)) return;
    int c_wr_idx = idx_2d(m,n,N);
    float acc = 0.;
    __global float4* input4 = (__global float4*)(input);
    __global float4* hidden4 = (__global float4*)(initial_hiddens);
    __global float4* weights4 = (__global float4*)(weights);
    for (int k = 0; k < K/4; k+=TILE_SIZE) {
        int mat_a_shared_write_offset = lid_x * TILE_SIZE + lid_y;
        int mat_b_shared_write_offset = lid_y * TILE_SIZE + lid_x;

        float4 a_matrix_elem = k < input_size/4 ? input4[idx_2d(m,k+lid_y,input_size/4) + M*input_size/4*timestep ]
                                                : hidden4[idx_2d(m,k+lid_y-input_size/4,hidden_size/4) + M*hidden_size/4*timestep];
        float4 b_matrix_elem = weights4[idx_2d(n,k+lid_x,K/4)];
        inputs_sm[mat_a_shared_write_offset] = a_matrix_elem;
        weights_sm[mat_b_shared_write_offset] = b_matrix_elem;

        __syncthreads();

        #pragma unroll
        for (int dp_iter = 0; dp_iter < TILE_SIZE; dp_iter++) {
            float4 a_tile = inputs_sm[idx_2d(lid_x, dp_iter, TILE_SIZE)];
            float4 b_tile = weights_sm[idx_2d(lid_y, dp_iter, TILE_SIZE)];
            acc += dp4a_fp32(a_tile, b_tile);
            //TILE_SIZEif(m == 0 && n == 0) {printf("INPUTS "); printf4(a_tile);}
            //if(m == 0 && n == 0) {printf("WEIGHTS "); printf4(b_tile);}
            
        }
        __syncthreads();
    }
    out_gates[c_wr_idx] = acc + bias[n];
}

__kernel void lstm_eltwise(__global float* inout_cell,
                             __global float *out_gates,
                             __global float*hidden_out,
                             int hidden_size,
                             int batch_size,
                             int timestep) {

    int m = get_local_id(0) + get_group_id(0) * get_local_size(0);
    int n = get_local_id(1) + get_group_id(1) * get_local_size(1);

    int i_idx = idx_2d(m,0*hidden_size+n,4*hidden_size);
    int f_idx = idx_2d(m,1*hidden_size+n,4*hidden_size);
    int g_idx = idx_2d(m,2*hidden_size+n,4*hidden_size);
    int o_idx = idx_2d(m,3*hidden_size+n,4*hidden_size);

    float i = out_gates[i_idx];
    float f = out_gates[f_idx];
    float g = out_gates[g_idx];
    float o = out_gates[o_idx];

    float cell = sigmoid(f) * inout_cell[idx_2d(m, n, hidden_size) + batch_size * hidden_size * timestep]  + sigmoid(i) * tanh(g);
    float hidden = sigmoid(o) * tanh(cell);

    int hidden_wr_timestep_offset = batch_size * hidden_size * (timestep+1);
    int hidden_wr_idx = idx_2d(m, n, hidden_size) + hidden_wr_timestep_offset;
    int cell_wr_idx = idx_2d(m, n, hidden_size) + hidden_wr_timestep_offset;
    hidden_out[hidden_wr_idx] = hidden;
    inout_cell[cell_wr_idx] = cell;
}

