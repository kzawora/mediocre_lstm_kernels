__device__ __forceinline__ double sigmoid (double a) { return 1.0 / (1.0 + exp (-a)); }
__device__ __forceinline__ int idx_2d(int x, int y, int width) { return x*width+y; }
__global__ void lstm_gemm(float *input,
                          float *initial_hiddens,
                          float *weights,
                          float *bias,
                          float *out_gates,
                          int M, int K, int N,
                          int input_size, int hidden_size) {
    int m = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
  //  if (m != 0 || n != 0) return;
    int c_wr_idx = idx_2d(m,n,N);
    for (int k = 0; k < K; k++) {
        int b_rd_idx = idx_2d(k,n,N);
        float a_matrix_elem = k < input_size ? input[idx_2d(m,k,input_size)]
                                             : initial_hiddens[idx_2d(m,k-input_size,hidden_size)];
        out_gates[c_wr_idx] += a_matrix_elem * weights[b_rd_idx];
    //    if (k >= input_size) printf("k %d; k-hidden %d; %f * %f\n", k, k-hidden_size, a_matrix_elem, weights[b_rd_idx]);
    }
    out_gates[c_wr_idx] += bias[n];
}

__global__ void lstm_eltwise(float* in_cell,
                             float *out_gates,
                             float*hidden_out,
                             float*cell_out,
                             int hidden_size) {

    int m = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
//    if (n >= hidden_size) return;

    int i_idx = idx_2d(m,0*hidden_size+n,4*hidden_size);
    int f_idx = idx_2d(m,1*hidden_size+n,4*hidden_size);
    int g_idx = idx_2d(m,2*hidden_size+n,4*hidden_size);
    int o_idx = idx_2d(m,3*hidden_size+n,4*hidden_size);

    float i = out_gates[i_idx];
    float f = out_gates[f_idx];
    float g = out_gates[g_idx];
    float o = out_gates[o_idx];

    // todo 0 -> prev cell
    float cell = sigmoid(f) * in_cell[idx_2d(m, n, hidden_size)]  + sigmoid(i) * tanh(g);
    float hidden = sigmoid(o) * tanh(cell);

    int hidden_wr_timestep_offset = 0;//batch_size * hidden_size * timestep;
    int hidden_wr_idx = idx_2d(m, n, hidden_size) + hidden_wr_timestep_offset;
    int cell_wr_idx = idx_2d(m, n, hidden_size);
    hidden_out[hidden_wr_idx] = hidden;
    cell_out[cell_wr_idx] = cell;
}

