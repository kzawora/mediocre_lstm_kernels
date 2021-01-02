from cu_lstm import CuLSTM
import numpy as np


def test_implementations(impls, input_size, hidden_size, Wh, Wi, B, y_ref, hx_ref, cx_ref):
    for impl in impls:
        cuda_obj = CuLSTM(input_size, hidden_size, Wh, Wi, B, impl)
        (cuda_y, cuda_hx, cuda_cx), cuda_exec_time = cuda_obj.forward(x, h0, c0)
        print(f'{impl} execution time(ms): {cuda_exec_time}')
        np.testing.assert_allclose(cuda_y, y_ref, atol=1e-2 * seq_len, rtol=1e-3)
        np.testing.assert_allclose(cuda_hx, hx_ref, atol=1e-2 * seq_len, rtol=1e-3)
        np.testing.assert_allclose(cuda_cx, cx_ref, atol=1e-2 * seq_len, rtol=1e-3)
        print(f'{impl} - torch comparison passed!')


seq_len = np.int32(15)
batch_size = np.int32(512)
input_size = np.int32(1024)
hidden_size = np.int32(512)
np.random.seed(0)
random_dist = lambda shape: np.random.normal(loc=0, scale=1, size=shape)
Wh = random_dist((4 * hidden_size, hidden_size)).astype(np.float32)
Wi = random_dist((4 * hidden_size, input_size)).astype(np.float32)
B = random_dist((4 * hidden_size)).astype(np.float32)
x = random_dist((seq_len, batch_size, input_size)).astype(np.float32)
h0 = random_dist((batch_size, hidden_size)).astype(np.float32)
c0 = random_dist((batch_size, hidden_size)).astype(np.float32)

torch_obj = CuLSTM(input_size, hidden_size, Wh, Wi, B, 'torch')
(torch_y, torch_hx, torch_cx), torch_exec_time = torch_obj.forward(x, h0, c0)
print(f'Torch execution time(ms): {torch_exec_time}')
#impls = ['ocl_vec4_interleaved']
impls = ['torch_gpu',
         'cuda_naive', 'cuda_naive_acc', 'cuda_vec4', 'cuda_vec4_sm', 'cuda_vec4_interleaved',
         'ocl_naive',  'ocl_naive_acc',  'ocl_vec4',  'ocl_vec4_sm',  'ocl_vec4_interleaved']
test_implementations(impls, input_size, hidden_size, Wh, Wi, B, torch_y, torch_hx, torch_cx)
