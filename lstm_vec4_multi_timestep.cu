__device__ __forceinline__ double sigmoid (double a) { return 1.0 / (1.0 + exp (-a)); }
__device__ __forceinline__ int idx_2d(int x, int y, int width) { return x*width+y; }
__device__ __forceinline__ float dp4a_fp32(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ __forceinline__ void printf4(float4 a) {  printf("%f %f %f %f\n", a.x, a.y, a.z, a.w); }

__global__ void lstm_gemm(float *input,
                          float *initial_hiddens,
                          float *weights,
                          float *bias,
                          float *out_gates,
                          int M, int K, int N,
                          int input_size, int hidden_size, int timestep) {
    int m = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
   // if (!(m == 0 && n == 0)) return;
    int c_wr_idx = idx_2d(m,n,N);
    float acc = 0.;
    float4* input4 = reinterpret_cast<float4*>(input);
    float4* hidden4 = reinterpret_cast<float4*>(initial_hiddens);
    float4* weights4 = reinterpret_cast<float4*>(weights);
    for (int k = 0; k < K/4; k++) {
        float4 a_matrix_elem = k < input_size/4 ? input4[idx_2d(m,k,input_size/4) + M*input_size/4*timestep ]
                                                : hidden4[idx_2d(m,k-input_size/4,hidden_size/4) + M*hidden_size/4*timestep];
        float4 b_matrix_elem = weights4[idx_2d(n,k,K/4)];
        acc += dp4a_fp32(a_matrix_elem, b_matrix_elem);
    }
    out_gates[c_wr_idx] = acc + bias[n];
}

__global__ void lstm_eltwise(float* inout_cell,
                             float *out_gates,
                             float*hidden_out,
                             int hidden_size,
                             int batch_size,
                             int timestep) {

    int m = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;

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

