__device__ __forceinline__ double sigmoid (double a) { return 1.0 / (1.0 + exp (-a)); }
__device__ __forceinline__ int idx_2d(int x, int y, int width) { return x*width+y; }
__device__ __forceinline__ float dp4a_fp32(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ __forceinline__ void printf4(float4 a) {  printf("%f %f %f %f\n", a.x, a.y, a.z, a.w); }
#define TILE_SIZE 8
__global__ void lstm_gemm(float *input,
                          float *initial_hiddens,
                          float *weights,
                          float *bias,
                          float *out_gates,
                          int M, int K, int N,
                          int input_size, int hidden_size, int timestep) {
    __shared__ float4 inputs_sm[TILE_SIZE * TILE_SIZE];
    __shared__ float4 weights_sm[TILE_SIZE * TILE_SIZE];
    int m = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;
    //if (!(blockIdx.x == 0 && blockIdx.y == 0)) return;
    int c_wr_idx = idx_2d(m,n,N);
    float acc = 0.;
    float4* input4 = reinterpret_cast<float4*>(input);
    float4* hidden4 = reinterpret_cast<float4*>(initial_hiddens);
    float4* weights4 = reinterpret_cast<float4*>(weights);
    for (int k = 0; k < K/4; k+=TILE_SIZE) {
        int mat_a_shared_write_offset = threadIdx.x * TILE_SIZE + threadIdx.y;
        int mat_b_shared_write_offset = threadIdx.y * TILE_SIZE + threadIdx.x;

        float4 a_matrix_elem = k < input_size/4 ? input4[idx_2d(m,k+threadIdx.y,input_size/4) + M*input_size/4*timestep ]
                                                : hidden4[idx_2d(m,k+threadIdx.y-input_size/4,hidden_size/4) + M*hidden_size/4*timestep];
        float4 b_matrix_elem = weights4[idx_2d(n,k+threadIdx.x,K/4)];
        inputs_sm[mat_a_shared_write_offset] = a_matrix_elem;
        weights_sm[mat_b_shared_write_offset] = b_matrix_elem;

        __syncthreads();

        #pragma unroll
        for (int dp_iter = 0; dp_iter < TILE_SIZE; dp_iter++) {
            float4 a_tile = inputs_sm[idx_2d(threadIdx.x, dp_iter, TILE_SIZE)];
            float4 b_tile = weights_sm[idx_2d(threadIdx.y, dp_iter, TILE_SIZE)];
            acc += dp4a_fp32(a_tile, b_tile);
            //TILE_SIZEif(m == 0 && n == 0) {printf("INPUTS "); printf4(a_tile);}
            //if(m == 0 && n == 0) {printf("WEIGHTS "); printf4(b_tile);}
            
        }
        __syncthreads();
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

