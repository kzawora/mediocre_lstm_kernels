import numpy as np
import torch
import datetime
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pyopencl as cl


class CuLSTM():
    def __init__(self, input_size, hidden_size, Wh, Wi, B, impl):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wh = Wh
        self.Wi = Wi
        self.B = B
        self.impl = impl
        self.func_testing = False

    def forward(self, x, h0, c0):
        # TORCH
        if self.impl == 'torch':
            return self.__forward_torch(x, h0, c0)
        if self.impl == 'torch_gpu':
            return self.__forward_torch_gpu(x, h0, c0)
        # CUDA
        if self.impl == 'cuda_naive':
            return self.__forward_cuda_naive_multi_timestep(x, h0, c0, acc=False)
        if self.impl == 'cuda_naive_acc':
            return self.__forward_cuda_naive_multi_timestep(x, h0, c0, acc=True)
        if self.impl == 'cuda_vec4' :
            return self.__forward_cuda_vec4_multi_timestep(x, h0, c0, sm=False)
        if self.impl == 'cuda_vec4_sm':
            return self.__forward_cuda_vec4_multi_timestep(x, h0, c0, sm=True)
        if self.impl == 'cuda_vec4_interleaved':
            return self.__forward_cuda_vec4_multi_timestep_interleaved(x, h0, c0, sm=False)
        # OCL
        if self.impl == 'ocl_naive':
            return self.__forward_ocl_naive(x, h0, c0, acc=False)
        if self.impl == 'ocl_naive_acc':
            return self.__forward_ocl_naive(x, h0, c0, acc=True)
        if self.impl == 'ocl_vec4':
            return self.__forward_ocl_vec4(x, h0, c0, sm=False)
        if self.impl == 'ocl_vec4_sm':
            return self.__forward_ocl_vec4(x, h0, c0, sm=True)
        if self.impl == 'ocl_vec4_interleaved':
            return self.__forward_ocl_vec4_interleaved(x, h0, c0, sm=False)

        raise NotImplementedError(f'Implementation not found: {self.impl}')

    def __forward_torch(self, x, h0, c0):
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)
        torch_lstm = torch.nn.LSTM(x.shape[2], self.hidden_size)
        torch_lstm.weight_ih_l0.data = torch.Tensor(self.Wi)
        torch_lstm.weight_hh_l0.data = torch.Tensor(self.Wh)
        torch_lstm.bias_ih_l0.data = torch.Tensor(self.B)
        torch_lstm.bias_hh_l0.data = torch.Tensor(torch.zeros_like(torch_lstm.bias_ih_l0.data))
        timer_start = datetime.datetime.now()
        if h0 is not None and c0 is not None:
            y, (ht, ct) = torch_lstm(torch.Tensor(x),
                                     (torch.unsqueeze(torch.Tensor(h0), dim=0), torch.unsqueeze(torch.Tensor(c0), dim=0)))
        else:
            y, (ht, ct) = torch_lstm(torch.Tensor(x))
        execution_time = (datetime.datetime.now() - timer_start).total_seconds() * 1000
        results = (y.detach().numpy(), ht.detach().numpy(), ct.detach().numpy())
        return results, execution_time

    def __forward_torch_gpu(self, x, h0, c0):
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)
        torch_lstm = torch.nn.LSTM(x.shape[2], self.hidden_size)
        torch_lstm.cuda()
        torch_lstm.weight_ih_l0.data = torch.Tensor(self.Wi).cuda()
        torch_lstm.weight_hh_l0.data = torch.Tensor(self.Wh).cuda()
        torch_lstm.bias_ih_l0.data = torch.Tensor(self.B).cuda()
        torch_lstm.bias_hh_l0.data = torch.zeros_like(torch_lstm.bias_ih_l0.data)
        x_t = torch.Tensor(x).cuda()
        h0_t = torch.Tensor(h0).cuda() if h0 is not None else None
        c0_t = torch.Tensor(c0).cuda() if h0 is not None else None
        torch_lstm.flatten_parameters()

        timer_start = datetime.datetime.now()
        if h0 is not None and c0 is not None:
            y, (ht, ct) = torch_lstm(x_t,(torch.unsqueeze(h0_t, dim=0), torch.unsqueeze(c0_t, dim=0)))
        else:
            y, (ht, ct) = torch_lstm(x_t)
        execution_time = (datetime.datetime.now() - timer_start).total_seconds() * 1000
        results = (y.cpu().detach().numpy(), ht.cpu().detach().numpy(), ct.cpu().detach().numpy())
        return results, execution_time

    def __forward_cuda_naive(self, x, h0, c0):
        # old prototype for single timestep
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        weights = np.concatenate((np.transpose(self.Wi), np.transpose(self.Wh)), 0).astype(np.float32)
        ifcos = np.zeros((batch_size, 4 * self.hidden_size)).astype(np.float32)
        hy = np.zeros((seq_len, batch_size, self.hidden_size)).astype(np.float32)
        cy = np.zeros((batch_size, self.hidden_size)).astype(np.float32)

        # Allocate on device
        cuda_alloc = lambda buf: cuda.mem_alloc(buf.size * buf.dtype.itemsize)
        x_gpu = cuda_alloc(x)
        h0_gpu = cuda_alloc(h0)
        c0_gpu = cuda_alloc(c0)
        weights_gpu = cuda_alloc(weights)
        bias_gpu = cuda_alloc(self.B)
        ifcos_gpu = cuda_alloc(ifcos)
        hy_gpu = cuda_alloc(hy)
        cy_gpu = cuda_alloc(cy)

        # Copy from host to device
        cuda.memcpy_htod(x_gpu, x.flatten())
        cuda.memcpy_htod(h0_gpu, h0.flatten())
        cuda.memcpy_htod(c0_gpu, c0.flatten())
        cuda.memcpy_htod(weights_gpu, weights.flatten())
        cuda.memcpy_htod(bias_gpu, self.B.flatten())
        cuda.memcpy_htod(ifcos_gpu, ifcos.flatten())
        cuda.memcpy_htod(hy_gpu, hy.flatten())
        cuda.memcpy_htod(cy_gpu, cy.flatten())

        kernelSource = ''
        with open('lstm_naive.cu', 'r') as file:
            kernelSource = file.read()
        source = SourceModule(kernelSource)
        lstm_gemm = source.get_function('lstm_gemm')
        lstm_eltwise = source.get_function('lstm_eltwise')

        M = np.int32(batch_size)
        K = np.int32(self.input_size + self.hidden_size)
        N = np.int32(4 * self.hidden_size)
        gemm_threads_per_block = (1, 1, 1)
        gemm_blocks_per_grid = int(M / gemm_threads_per_block[0]), int(N / gemm_threads_per_block[1])
        eltwise_threads_per_block = (1, 1, 1)
        eltwise_blocks_per_grid = int(M / eltwise_threads_per_block[0]), int(self.hidden_size / eltwise_threads_per_block[1])
        timer_start = datetime.datetime.now()
        lstm_gemm(x_gpu,
                  h0_gpu,
                  weights_gpu,
                  bias_gpu,
                  ifcos_gpu,
                  M, K, N,
                  np.int32(self.input_size), np.int32(self.hidden_size),
                  block=gemm_threads_per_block, grid=gemm_blocks_per_grid)
        drv.Context.synchronize()
        lstm_eltwise(c0_gpu,
                     ifcos_gpu,
                     hy_gpu,
                     cy_gpu,
                     np.int32(self.hidden_size),
                     block=eltwise_threads_per_block, grid=eltwise_blocks_per_grid)
        drv.Context.synchronize()
        execution_time = (datetime.datetime.now() - timer_start).total_seconds() * 1000

        cuda.memcpy_dtoh(ifcos, ifcos_gpu)
        ifcos_ref = np.concatenate([x[0], h0], axis=1) @ weights + self.B
        np.testing.assert_allclose(ifcos_ref, ifcos, atol=1e-5, rtol=0.01)
        print("CUDA_naive: matrix multiplication results are correct!")
        cuda.memcpy_dtoh(hy, hy_gpu)
        cuda.memcpy_dtoh(cy, cy_gpu)
        results = hy[-1], hy, cy[-1]
        return results, execution_time

    def __forward_cuda_naive_multi_timestep(self, x, h0, c0, acc):
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        weights = np.concatenate((np.transpose(self.Wi), np.transpose(self.Wh)), 0).astype(np.float32)
        ifcos = np.zeros((batch_size, 4 * self.hidden_size)).astype(np.float32)
        hy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        cy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        hy[0] = h0
        cy[0] = c0

        # Allocate on device
        cuda_alloc = lambda buf: cuda.mem_alloc(buf.size * buf.dtype.itemsize)
        x_gpu = cuda_alloc(x)
        #        h0_gpu = cuda_alloc(h0)
        #        c0_gpu = cuda_alloc(c0)
        weights_gpu = cuda_alloc(weights)
        bias_gpu = cuda_alloc(self.B)
        ifcos_gpu = cuda_alloc(ifcos)
        hy_gpu = cuda_alloc(hy)
        cy_gpu = cuda_alloc(cy)

        # Copy from host to device
        cuda.memcpy_htod(x_gpu, x.flatten())
        #        cuda.memcpy_htod(h0_gpu, h0.flatten())
        #        cuda.memcpy_htod(c0_gpu, c0.flatten())
        cuda.memcpy_htod(weights_gpu, weights.flatten())
        cuda.memcpy_htod(bias_gpu, self.B.flatten())
        cuda.memcpy_htod(ifcos_gpu, ifcos.flatten())
        cuda.memcpy_htod(hy_gpu, hy.flatten())
        cuda.memcpy_htod(cy_gpu, cy.flatten())

        kernelSource = ''
        filename = 'lstm_naive_multi_timestep.cu' if acc is None else 'lstm_naive_multi_timestep_acc.cu'
        with open(filename, 'r') as file:
            kernelSource = file.read()
        source = SourceModule(kernelSource)
        lstm_gemm = source.get_function('lstm_gemm')
        lstm_eltwise = source.get_function('lstm_eltwise')

        M = np.int32(batch_size)
        K = np.int32(self.input_size + self.hidden_size)
        N = np.int32(4 * self.hidden_size)
        gemm_threads_per_block = (1, 1, 1)
        gemm_blocks_per_grid = int(M / gemm_threads_per_block[0]), int(N / gemm_threads_per_block[1])
        eltwise_threads_per_block = (1, 1, 1)
        eltwise_blocks_per_grid = int(M / eltwise_threads_per_block[0]), int(self.hidden_size / eltwise_threads_per_block[1])

        cuda_stream = drv.Stream()
        for i in range(0, seq_len):
            lstm_gemm(x_gpu,
                      hy_gpu,
                      weights_gpu,
                      bias_gpu,
                      ifcos_gpu,
                      M, K, N,
                      np.int32(self.input_size), np.int32(self.hidden_size), np.int32(i),
                      block=gemm_threads_per_block, grid=gemm_blocks_per_grid, stream=cuda_stream)
            lstm_eltwise(cy_gpu,
                         ifcos_gpu,
                         hy_gpu,
                         np.int32(self.hidden_size), np.int32(batch_size), np.int32(i),
                         block=eltwise_threads_per_block, grid=eltwise_blocks_per_grid, stream=cuda_stream)
        timer_start = datetime.datetime.now()
        cuda_stream.synchronize()
        execution_time = (datetime.datetime.now() - timer_start).total_seconds() * 1000

        cuda.memcpy_dtoh(ifcos, ifcos_gpu)
        cuda.memcpy_dtoh(hy, hy_gpu)
        cuda.memcpy_dtoh(cy, cy_gpu)
        if self.func_testing:
            ifcos_ref = np.concatenate([x[seq_len - 1], hy[seq_len - 1]], axis=1) @ weights + self.B
            np.testing.assert_allclose(ifcos_ref, ifcos, atol=1e-5, rtol=0.01)
            print(f'CUDA_naive ts({i}): matrix multiplication results are correct!')

        results = hy[1:], hy[-1:], cy[-1:]
        return results, execution_time

    def __forward_cuda_vec4_multi_timestep(self, x, h0, c0, sm):
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        weights = np.concatenate((np.transpose(self.Wi), np.transpose(self.Wh)), 0).T.astype(np.float32)
        ifcos = np.zeros((batch_size, 4 * self.hidden_size)).astype(np.float32)
        hy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        cy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        hy[0] = h0
        cy[0] = c0
        cy[0] = c0

        # Allocate on device
        cuda_alloc = lambda buf: cuda.mem_alloc(buf.size * buf.dtype.itemsize)
        x_gpu = cuda_alloc(x)
        #        h0_gpu = cuda_alloc(h0)
        #        c0_gpu = cuda_alloc(c0)
        weights_gpu = cuda_alloc(weights)
        bias_gpu = cuda_alloc(self.B)
        ifcos_gpu = cuda_alloc(ifcos)
        hy_gpu = cuda_alloc(hy)
        cy_gpu = cuda_alloc(cy)

        # Copy from host to device
        cuda.memcpy_htod(x_gpu, x.flatten())
        #        cuda.memcpy_htod(h0_gpu, h0.flatten())
        #        cuda.memcpy_htod(c0_gpu, c0.flatten())
        cuda.memcpy_htod(weights_gpu, weights.flatten())
        cuda.memcpy_htod(bias_gpu, self.B.flatten())
        cuda.memcpy_htod(ifcos_gpu, ifcos.flatten())
        cuda.memcpy_htod(hy_gpu, hy.flatten())
        cuda.memcpy_htod(cy_gpu, cy.flatten())

        kernelSource = ''
        kernelFilename = 'lstm_vec4_multi_timestep.cu' if sm is False else 'lstm_vec4_sm_multi_timestep.cu'
        with open(kernelFilename, 'r') as file:
            kernelSource = file.read()
        source = SourceModule(kernelSource)
        lstm_gemm = source.get_function('lstm_gemm')
        lstm_eltwise = source.get_function('lstm_eltwise')

        M = np.int32(batch_size)
        K = np.int32(self.input_size + self.hidden_size)
        N = np.int32(4 * self.hidden_size)
        gemm_threads_per_block = (8, 8, 1)
        gemm_blocks_per_grid = int(M / gemm_threads_per_block[0]), int(N / gemm_threads_per_block[1])
        eltwise_threads_per_block = (8, 8, 1)
        eltwise_blocks_per_grid = int(M / eltwise_threads_per_block[0]), int(self.hidden_size / eltwise_threads_per_block[1])

        cuda_stream = drv.Stream()
        #        drv.Context.synchronize()
        for i in range(0, seq_len):
            lstm_gemm(x_gpu,
                      hy_gpu,
                      weights_gpu,
                      bias_gpu,
                      ifcos_gpu,
                      M, K, N,
                      np.int32(self.input_size), np.int32(self.hidden_size), np.int32(i),
                      block=gemm_threads_per_block, grid=gemm_blocks_per_grid, stream=cuda_stream)
            lstm_eltwise(cy_gpu,
                         ifcos_gpu,
                         hy_gpu,
                         np.int32(self.hidden_size), np.int32(batch_size), np.int32(i),
                         block=eltwise_threads_per_block, grid=eltwise_blocks_per_grid, stream=cuda_stream)
        #        drv.Context.synchronize()
        timer_start = datetime.datetime.now()
        cuda_stream.synchronize()
        execution_time = (datetime.datetime.now() - timer_start).total_seconds() * 1000

        cuda.memcpy_dtoh(ifcos, ifcos_gpu)
        cuda.memcpy_dtoh(hy, hy_gpu)
        cuda.memcpy_dtoh(cy, cy_gpu)
        if True:
            ifcos_ref = np.concatenate([x[seq_len - 1], hy[seq_len - 1]], axis=1) @ weights.T + self.B
            np.testing.assert_allclose(ifcos_ref, ifcos, atol=1e-2, rtol=0.01)
            print(f'CUDA_naive ts({i}): matrix multiplication results are correct!')

        results = hy[1:], hy[-1:], cy[-1:]
        return results, execution_time

    def __forward_cuda_vec4_multi_timestep_interleaved(self, x, h0, c0, sm):
        def interleave(matrix):
            if len(matrix.shape) == 1:
                return np.squeeze(interleave(np.expand_dims(matrix, axis=0)), axis=0).copy()
            new = np.zeros_like(matrix)
            a, b, c, d = np.hsplit(matrix, 4)
            simd = 4
            rng = np.arange(0, matrix.shape[1], simd)
            new[:, rng] = a
            new[:, rng + 1] = b
            new[:, rng + 2] = c
            new[:, rng + 3] = d
            return new.copy()

        seq_len = x.shape[0]
        batch_size = x.shape[1]
        weights = interleave(np.concatenate((np.transpose(self.Wi), np.transpose(self.Wh)), 0)).T.astype(np.float32)
        # todo: weight and bias interleaving
        ifcos = np.zeros((batch_size, 4 * self.hidden_size)).astype(np.float32)
        hy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        cy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        hy[0] = h0
        cy[0] = c0
        cy[0] = c0
        # Allocate on device
        cuda_alloc = lambda buf: cuda.mem_alloc(buf.size * buf.dtype.itemsize)
        x_gpu = cuda_alloc(x)
        #        h0_gpu = cuda_alloc(h0)
        #        c0_gpu = cuda_alloc(c0)
        weights_gpu = cuda_alloc(weights)
        bias_gpu = cuda_alloc(self.B)
        ifcos_gpu = cuda_alloc(ifcos)
        hy_gpu = cuda_alloc(hy)
        cy_gpu = cuda_alloc(cy)

        # Copy from host to device
        cuda.memcpy_htod(x_gpu, x.flatten())
        #        cuda.memcpy_htod(h0_gpu, h0.flatten())
        #        cuda.memcpy_htod(c0_gpu, c0.flatten())
        cuda.memcpy_htod(weights_gpu, weights.flatten())
        cuda.memcpy_htod(bias_gpu, interleave(self.B).flatten())
        cuda.memcpy_htod(ifcos_gpu, ifcos.flatten())
        cuda.memcpy_htod(hy_gpu, hy.flatten())
        cuda.memcpy_htod(cy_gpu, cy.flatten())

        kernelSource = ''
        kernelFilename = 'lstm_vec4_multi_timestep_interleaved.cu' if sm is False else 'lstm_vec4_sm_multi_timestep.cu'
        with open(kernelFilename, 'r') as file:
            kernelSource = file.read()
        source = SourceModule(kernelSource)
        lstm_gemm = source.get_function('lstm_gemm')
        lstm_eltwise = source.get_function('lstm_eltwise')

        M = np.int32(batch_size)
        K = np.int32(self.input_size + self.hidden_size)
        N = np.int32(4 * self.hidden_size)
        gemm_threads_per_block = (8, 8, 1)
        gemm_blocks_per_grid = int(M / gemm_threads_per_block[0]), int(N / gemm_threads_per_block[1])
        eltwise_threads_per_block = (8, 8, 1)
        eltwise_blocks_per_grid = int(M / eltwise_threads_per_block[0]), int(self.hidden_size / eltwise_threads_per_block[1])

        cuda_stream = drv.Stream()
        #        drv.Context.synchronize()
        for i in range(0, seq_len):
            lstm_gemm(x_gpu,
                      hy_gpu,
                      weights_gpu,
                      bias_gpu,
                      ifcos_gpu,
                      M, K, N,
                      np.int32(self.input_size), np.int32(self.hidden_size), np.int32(i),
                      block=gemm_threads_per_block, grid=gemm_blocks_per_grid, stream=cuda_stream)
            lstm_eltwise(cy_gpu,
                         ifcos_gpu,
                         hy_gpu,
                         np.int32(self.hidden_size), np.int32(batch_size), np.int32(i),
                         block=eltwise_threads_per_block, grid=eltwise_blocks_per_grid, stream=cuda_stream)
        #        drv.Context.synchronize()
        timer_start = datetime.datetime.now()
        cuda_stream.synchronize()
        execution_time = (datetime.datetime.now() - timer_start).total_seconds() * 1000

        cuda.memcpy_dtoh(ifcos, ifcos_gpu)
        cuda.memcpy_dtoh(hy, hy_gpu)
        cuda.memcpy_dtoh(cy, cy_gpu)
        if True:
            ifcos_ref = np.concatenate([x[seq_len - 1], hy[seq_len - 1]], axis=1) @ weights.T + interleave(self.B)
            np.testing.assert_allclose(ifcos_ref, ifcos, atol=1e-2, rtol=0.01)
            print(f'CUDA_naive ts({i}): matrix multiplication results are correct!')

        results = hy[1:], hy[-1:], cy[-1:]
        return results, execution_time

    def __forward_ocl_naive(self, x, h0, c0, acc):
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        weights = np.concatenate((np.transpose(self.Wi), np.transpose(self.Wh)), 0).astype(np.float32)
        ifcos = np.zeros((batch_size, 4 * self.hidden_size)).astype(np.float32)
        hy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        cy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        hy[0] = h0
        cy[0] = c0

        platform = cl.get_platforms()[0]  # Select the first platform [0]
        device = platform.get_devices()[0]  # Select the first device on this platform [0]
        context = cl.Context([device])  # Create a context with your device
        queue = cl.CommandQueue(context)  # Create a command queue with your context

        # Allocate on device
        x_gpu = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=x.copy('C'))  # cuda_alloc(x)
        weights_gpu = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=weights.copy('C'))  # cuda_alloc(weights)
        bias_gpu = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.B.copy('C'))  # cuda_alloc(self.B)
        ifcos_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=ifcos.copy('C'))  # cuda_alloc(ifcos)
        hy_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=hy.copy('C'))  # cuda_alloc(hy)
        cy_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy.copy('C'))  # cuda_alloc(cy)

        kernelSource = ''
        kernelFilename = 'lstm_naive.cl' if acc is False else 'lstm_naive_acc.cl'
        with open(kernelFilename, 'r') as file:
            kernelSource = file.read()

        program = cl.Program(context, kernelSource).build()

        M = np.int32(batch_size)
        K = np.int32(self.input_size + self.hidden_size)
        N = np.int32(4 * self.hidden_size)

        gemm_lws = (1, 1, 1)
        gemm_gws = int(M), int(N), gemm_lws[2]
        eltwise_lws = (1, 1, 1)
        eltwise_gws = int(M), int(self.hidden_size), eltwise_lws[2]
        events = []
        for i in range(0, seq_len):
            gemm_kernel = program.lstm_gemm
            gemm_kernel.set_args(x_gpu,
                                 hy_gpu,
                                 weights_gpu,
                                 bias_gpu,
                                 ifcos_gpu,
                                 M, K, N,
                                 np.int32(self.input_size), np.int32(self.hidden_size), np.int32(i))
            ev1 = cl.enqueue_nd_range_kernel(queue, gemm_kernel, gemm_gws, gemm_lws)
            events.append(ev1)
            cl.enqueue_barrier(queue)
            eltwise_kernel = program.lstm_eltwise
            eltwise_kernel.set_args(cy_gpu,
                                    ifcos_gpu,
                                    hy_gpu,
                                    np.int32(self.hidden_size), np.int32(batch_size), np.int32(i))
            ev2 = cl.enqueue_nd_range_kernel(queue, eltwise_kernel, eltwise_gws, eltwise_lws)
            events.append(ev2)
            cl.enqueue_barrier(queue)

        timer_start = datetime.datetime.now()
        cl.wait_for_events(events)
        execution_time = (datetime.datetime.now() - timer_start).total_seconds() * 1000

        cl.enqueue_copy(queue, ifcos, ifcos_gpu)
        cl.enqueue_copy(queue, hy, hy_gpu)
        cl.enqueue_copy(queue, cy, cy_gpu)

        queue.finish()

        results = hy[1:], hy[-1:], cy[-1:]

        return results, execution_time

    def __forward_ocl_vec4(self, x, h0, c0, sm=False):
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        weights = np.concatenate((np.transpose(self.Wi), np.transpose(self.Wh)), 0).T.astype(np.float32)
        ifcos = np.zeros((batch_size, 4 * self.hidden_size)).astype(np.float32)
        hy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        cy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        hy[0] = h0
        cy[0] = c0

        platform = cl.get_platforms()[0]  # Select the first platform [0]
        device = platform.get_devices()[0]  # Select the first device on this platform [0]
        context = cl.Context([device])  # Create a context with your device
        queue = cl.CommandQueue(context)  # Create a command queue with your context

        # Allocate on device
        x_gpu = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=x.copy('C'))  # cuda_alloc(x)
        weights_gpu = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=weights.copy('C'))  # cuda_alloc(weights)
        bias_gpu = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.B.copy('C'))  # cuda_alloc(self.B)
        ifcos_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=ifcos.copy('C'))  # cuda_alloc(ifcos)
        hy_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=hy.copy('C'))  # cuda_alloc(hy)
        cy_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy.copy('C'))  # cuda_alloc(cy)

        kernelSource = ''
        kernelFilename = 'lstm_vec4.cl' if sm is False else 'lstm_vec4_sm.cl'
        with open(kernelFilename, 'r') as file:
            kernelSource = file.read()

        program = cl.Program(context, kernelSource).build()

        M = np.int32(batch_size)
        K = np.int32(self.input_size + self.hidden_size)
        N = np.int32(4 * self.hidden_size)
        gemm_lws = (8, 8, 1)
        gemm_gws = int(M), int(N), gemm_lws[2]
        eltwise_lws = (8, 8, 1)
        eltwise_gws = int(M), int(self.hidden_size), eltwise_lws[2]
        events = []
        for i in range(0, seq_len):
            gemm_kernel = program.lstm_gemm
            gemm_kernel.set_args(x_gpu,
                                 hy_gpu,
                                 weights_gpu,
                                 bias_gpu,
                                 ifcos_gpu,
                                 M, K, N,
                                 np.int32(self.input_size), np.int32(self.hidden_size), np.int32(i))
            ev1 = cl.enqueue_nd_range_kernel(queue, gemm_kernel, gemm_gws, gemm_lws)
            events.append(ev1)
            cl.enqueue_barrier(queue)
            eltwise_kernel = program.lstm_eltwise
            eltwise_kernel.set_args(cy_gpu,
                                    ifcos_gpu,
                                    hy_gpu,
                                    np.int32(self.hidden_size), np.int32(batch_size), np.int32(i))
            ev2 = cl.enqueue_nd_range_kernel(queue, eltwise_kernel, eltwise_gws, eltwise_lws)
            events.append(ev2)
            cl.enqueue_barrier(queue)

        timer_start = datetime.datetime.now()
        cl.wait_for_events(events)
        execution_time = (datetime.datetime.now() - timer_start).total_seconds() * 1000

        cl.enqueue_copy(queue, ifcos, ifcos_gpu)
        cl.enqueue_copy(queue, hy, hy_gpu)
        cl.enqueue_copy(queue, cy, cy_gpu)

        queue.finish()
        # Copy the data for array c back to the host

        results = hy[1:], hy[-1:], cy[-1:]

        return results, execution_time

    def __forward_ocl_vec4_interleaved(self, x, h0, c0, sm=False):
        def interleave(matrix):
            if len(matrix.shape) == 1:
                return np.squeeze(interleave(np.expand_dims(matrix, axis=0)), axis=0).copy()
            new = np.zeros_like(matrix)
            a, b, c, d = np.hsplit(matrix, 4)
            simd = 4
            rng = np.arange(0, matrix.shape[1], simd)
            new[:, rng] = a
            new[:, rng + 1] = b
            new[:, rng + 2] = c
            new[:, rng + 3] = d
            return new.copy()

        seq_len = x.shape[0]
        batch_size = x.shape[1]
        weights = interleave(np.concatenate((np.transpose(self.Wi), np.transpose(self.Wh)), 0)).T.astype(np.float32)
        ifcos = np.zeros((batch_size, 4 * self.hidden_size)).astype(np.float32)
        hy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        cy = np.zeros((seq_len + 1, batch_size, self.hidden_size)).astype(np.float32)
        hy[0] = h0
        cy[0] = c0

        platform = cl.get_platforms()[0]  # Select the first platform [0]
        device = platform.get_devices()[0]  # Select the first device on this platform [0]
        context = cl.Context([device])  # Create a context with your device
        queue = cl.CommandQueue(context)  # Create a command queue with your context

        # Allocate on device
        x_gpu = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=x.copy('C'))  # cuda_alloc(x)
        weights_gpu = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=weights.copy('C'))  # cuda_alloc(weights)
        bias_gpu = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=interleave(self.B).copy('C'))  # cuda_alloc(self.B)
        ifcos_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=ifcos.copy('C'))  # cuda_alloc(ifcos)
        hy_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=hy.copy('C'))  # cuda_alloc(hy)
        cy_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy.copy('C'))  # cuda_alloc(cy)

        kernelSource = ''
        kernelFilename = 'lstm_vec4_interleaved.cl'
        with open(kernelFilename, 'r') as file:
            kernelSource = file.read()

        program = cl.Program(context, kernelSource).build()

        M = np.int32(batch_size)
        K = np.int32(self.input_size + self.hidden_size)
        N = np.int32(4 * self.hidden_size)
        gemm_lws = (8, 8, 1)
        gemm_gws = int(M), int(N), gemm_lws[2]
        eltwise_lws = (8, 8, 1)
        eltwise_gws = int(M), int(self.hidden_size), eltwise_lws[2]
        events = []
        for i in range(0, seq_len):
            gemm_kernel = program.lstm_gemm
            gemm_kernel.set_args(x_gpu,
                                 hy_gpu,
                                 weights_gpu,
                                 bias_gpu,
                                 ifcos_gpu,
                                 M, K, N,
                                 np.int32(self.input_size), np.int32(self.hidden_size), np.int32(i))
            ev1 = cl.enqueue_nd_range_kernel(queue, gemm_kernel, gemm_gws, gemm_lws)
            events.append(ev1)
            cl.enqueue_barrier(queue)
            eltwise_kernel = program.lstm_eltwise
            eltwise_kernel.set_args(cy_gpu,
                                    ifcos_gpu,
                                    hy_gpu,
                                    np.int32(self.hidden_size), np.int32(batch_size), np.int32(i))
            ev2 = cl.enqueue_nd_range_kernel(queue, eltwise_kernel, eltwise_gws, eltwise_lws)
            events.append(ev2)
            cl.enqueue_barrier(queue)

        timer_start = datetime.datetime.now()
        cl.wait_for_events(events)
        execution_time = (datetime.datetime.now() - timer_start).total_seconds() * 1000

        cl.enqueue_copy(queue, ifcos, ifcos_gpu)
        cl.enqueue_copy(queue, hy, hy_gpu)
        cl.enqueue_copy(queue, cy, cy_gpu)

        queue.finish()
        # Copy the data for array c back to the host

        results = hy[1:], hy[-1:], cy[-1:]

        return results, execution_time
