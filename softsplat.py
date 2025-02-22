import torch

import triton
import triton.language as tl


def softsplat(tenIn:torch.Tensor, tenFlow:torch.Tensor, tenMetric:torch.Tensor, strMode:str):
    assert(strMode.split('-')[0] in ['sum', 'avg', 'linear', 'soft'])

    if strMode == 'sum': assert(tenMetric is None)
    if strMode == 'avg': assert(tenMetric is None)
    if strMode.split('-')[0] == 'linear': assert(tenMetric is not None)
    if strMode.split('-')[0] == 'soft': assert(tenMetric is not None)

    if strMode == 'avg':
        tenIn = torch.cat([tenIn, tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]])], 1)

    elif strMode.split('-')[0] == 'linear':
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)

    elif strMode.split('-')[0] == 'soft':
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    tenOut = softsplat_func.apply(tenIn, tenFlow)

    if strMode.split('-')[0] in ['avg', 'linear', 'soft']:
        tenNormalize = tenOut[:, -1:, :, :]

        if len(strMode.split('-')) == 1:
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split('-')[1] == 'addeps':
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split('-')[1] == 'zeroeps':
            tenNormalize[tenNormalize == 0.0] = 1.0

        elif strMode.split('-')[1] == 'clipeps':
            tenNormalize = tenNormalize.clip(0.0000001, None)


        tenOut = tenOut[:, :-1, :, :] / tenNormalize


    return tenOut


@triton.jit
def isfinite(x):
    return (x == x) & (tl.abs(x) < float('inf'))


@triton.jit
def softsplat_fwd_kernel(
    tenIn_ptr, tenFlow_ptr, tenOut_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = N * C * H * W
    mask = offsets < n_elements

    intIndex = offsets
    intN = intIndex // (C * H * W)
    intC = (intIndex // (H * W)) % C
    intY = (intIndex // W) % H
    intX = intIndex % W

    tenIn = tl.load(tenIn_ptr + intN * C * H * W + intC * H * W + intY * W + intX, mask=mask)
    tenFlow_x = tl.load(tenFlow_ptr + intN * 2 * H * W + 0 * H * W + intY * W + intX, mask=mask)
    tenFlow_y = tl.load(tenFlow_ptr + intN * 2 * H * W + 1 * H * W + intY * W + intX, mask=mask)

    # new coordinates w.r.t flow (float)
    fltX = intX + tenFlow_x
    fltY = intY + tenFlow_y
    
    # Check for finite values
    is_finite = isfinite(fltX) & isfinite(fltY)
    fltX = tl.where(is_finite, fltX, 0.0)
    fltY = tl.where(is_finite, fltY, 0.0)

    intNorthwestX = tl.floor(fltX).to(tl.int32)
    intNorthwestY = tl.floor(fltY).to(tl.int32)

    intNortheastX = intNorthwestX + 1
    intNortheastY = intNorthwestY

    intSouthwestX = intNorthwestX
    intSouthwestY = intNorthwestY + 1

    intSoutheastX = intNorthwestX + 1
    intSoutheastY = intNorthwestY + 1

    # weights
    fltNorthwest = (intSoutheastX - fltX) * (intSoutheastY - fltY)
    fltNortheast = (fltX - intSouthwestX) * (intSouthwestY - fltY)
    fltSouthwest = (intNortheastX - fltX) * (fltY - intNortheastY)
    fltSoutheast = (fltX - intNorthwestX) * (fltY - intNorthwestY)
  
    for i in range(4):
        if i == 0:
            dx, dy, weight = intNorthwestX, intNorthwestY, fltNorthwest
        elif i == 1:
            dx, dy, weight = intNortheastX, intNortheastY, fltNortheast
        elif i == 2:
            dx, dy, weight = intSouthwestX, intSouthwestY, fltSouthwest
        else:
            dx, dy, weight = intSoutheastX, intSoutheastY, fltSoutheast
        within_bounds = (dx >=0) & (dx < W) & (dy >=0) & (dy < H)
        out_offset = intN * (C * H * W) + intC * (H * W) + dy * W + dx
        tl.atomic_add(tenOut_ptr + out_offset, tenIn * weight, mask=mask & within_bounds)   

@triton.jit
def softsplat_bwd_kernel_ingrad(
    tenIn_ptr, tenFlow_ptr, 
    tenOutgrad_ptr, tenIngrad_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = N * C * H * W
    mask = offsets < n_elements

    intIndex = offsets
    intN = intIndex // (C * H * W)
    intC = (intIndex // (H * W)) % C
    intY = (intIndex // W) % H
    intX = intIndex % W

    tenIn = tl.load(tenIn_ptr + intN * C * H * W + intC * H * W + intY * W + intX, mask=mask)
    tenFlow_x = tl.load(tenFlow_ptr + intN * 2 * H * W + 0 * H * W + intY * W + intX, mask=mask)
    tenFlow_y = tl.load(tenFlow_ptr + intN * 2 * H * W + 1 * H * W + intY * W + intX, mask=mask)

    fltX = intX + tenFlow_x
    fltY = intY + tenFlow_y

    # Check for finite values
    is_finite = isfinite(fltX) & isfinite(fltY)
    fltX = tl.where(is_finite, fltX, 0.0)
    fltY = tl.where(is_finite, fltY, 0.0)

    intNorthwestX = tl.floor(fltX).to(tl.int32)
    intNorthwestY = tl.floor(fltY).to(tl.int32)

    intNortheastX = intNorthwestX + 1
    intNortheastY = intNorthwestY

    intSouthwestX = intNorthwestX
    intSouthwestY = intNorthwestY + 1

    intSoutheastX = intNorthwestX + 1
    intSoutheastY = intNorthwestY + 1

    # weights
    fltNorthwest = (intSoutheastX - fltX) * (intSoutheastY - fltY)
    fltNortheast = (fltX - intSouthwestX) * (intSouthwestY - fltY)
    fltSouthwest = (intNortheastX - fltX) * (fltY - intNortheastY)
    fltSoutheast = (fltX - intNorthwestX) * (fltY - intNorthwestY)

    # Accumulate gradient for tenIngrad
    # fltIngrad = 0.0
    fltIngrad = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(4):
        if i == 0:
            dx, dy, weight = intNorthwestX, intNorthwestY, fltNorthwest
        elif i == 1:
            dx, dy, weight = intNortheastX, intNortheastY, fltNortheast
        elif i == 2:
            dx, dy, weight = intSouthwestX, intSouthwestY, fltSouthwest
        else:
            dx, dy, weight = intSoutheastX, intSoutheastY, fltSoutheast

        within_bounds = (dx >=0) & (dx < W) & (dy >=0) & (dy < H)
        outgrad_offset = intN * (C * H * W) + intC * (H * W) + dy * W + dx
        fltIngrad += tl.load(tenOutgrad_ptr + outgrad_offset, mask=mask & within_bounds) * weight

    tl.store(tenIngrad_ptr+intIndex, fltIngrad, mask=mask)


@triton.jit
def softsplat_bwd_kernel_flowgrad(
    tenIn_ptr, tenFlow_ptr, 
    tenOutgrad_ptr, tenFlowgrad_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = N * 2 * H * W
    mask = offsets < n_elements

    intIndex = offsets
    intN = intIndex // (2 * H * W)
    intC = (intIndex // (H * W)) % 2
    intY = (intIndex // W) % H
    intX = intIndex % W

    # tl.static_print(f"intC={intC}")


    tenFlow_x = tl.load(tenFlow_ptr + intN * 2 * H * W + 0 * H * W + intY * W + intX, mask=mask)
    tenFlow_y = tl.load(tenFlow_ptr + intN * 2 * H * W + 1 * H * W + intY * W + intX, mask=mask)

    fltX = intX + tenFlow_x
    fltY = intY + tenFlow_y

    # Check for finite values
    is_finite = isfinite(fltX) & isfinite(fltY)
    fltX = tl.where(is_finite, fltX, 0.0)
    fltY = tl.where(is_finite, fltY, 0.0)

    intNorthwestX = tl.floor(fltX).to(tl.int32)
    intNorthwestY = tl.floor(fltY).to(tl.int32)

    intNortheastX = intNorthwestX + 1
    intNortheastY = intNorthwestY

    intSouthwestX = intNorthwestX
    intSouthwestY = intNorthwestY + 1

    intSoutheastX = intNorthwestX + 1
    intSoutheastY = intNorthwestY + 1


    # fltFlowgrad = 0.0
    fltFlowgrad = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    mask_flow_x = intC == 0  

    # grad for flow_x
    fltNorthwest_x = -1.0 * (intSoutheastY - fltY)
    fltNortheast_x = 1.0 * (intSouthwestY - fltY)
    fltSouthwest_x = -1.0 * (fltY - intNortheastY)
    fltSoutheast_x = 1.0 * (fltY - intNorthwestY)

    # grad for flow_y
    fltNorthwest_y = -1.0 * (intSoutheastX - fltX)
    fltNortheast_y = -1.0 * (fltX - intSouthwestX)
    fltSouthwest_y = 1.0 * (intNortheastX - fltX)
    fltSoutheast_y = 1.0 * (fltX - intNorthwestX)

    fltNorthwest = tl.where(mask_flow_x, fltNorthwest_x, fltNorthwest_y)
    fltNortheast = tl.where(mask_flow_x, fltNortheast_x, fltNortheast_y)
    fltSouthwest = tl.where(mask_flow_x, fltSouthwest_x, fltSouthwest_y)
    fltSoutheast = tl.where(mask_flow_x, fltSoutheast_x, fltSoutheast_y)

    # Accumulate gradient for tenFlowgrad
    for i in range(4):
        if i == 0:
            dx, dy, weight = intNorthwestX, intNorthwestY, fltNorthwest
        elif i == 1:
            dx, dy, weight = intNortheastX, intNortheastY, fltNortheast
        elif i == 2:
            dx, dy, weight = intSouthwestX, intSouthwestY, fltSouthwest
        else:
            dx, dy, weight = intSoutheastX, intSoutheastY, fltSoutheast

        within_bounds = (dx >=0) & (dx < W) & (dy >=0) & (dy < H)
        for intChannel in range(C):
            in_offset = intN * (C * H * W) + intChannel * (H * W) + intY * W + intX
            fltIn = tl.load(tenIn_ptr + in_offset, mask = mask & within_bounds)

            outgrad_offset = intN * (C * H * W) + intChannel * (H * W) + dy * W + dx
            outgrad_val = tl.load(tenOutgrad_ptr + outgrad_offset, mask = mask & within_bounds)
            fltFlowgrad += outgrad_val * fltIn * weight

    tl.store(tenFlowgrad_ptr+intIndex, fltFlowgrad, mask=mask)


class softsplat_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tenIn, tenFlow):
        N, C, H, W = tenIn.shape
        tenOut = torch.zeros_like(tenIn)

        # Launch forward kernel
        grid = lambda meta: (triton.cdiv(N * C * H * W, meta['BLOCK_SIZE']),)
        softsplat_fwd_kernel[grid](
            tenIn, tenFlow, tenOut, N, C, H, W, BLOCK_SIZE=1024
        )

        ctx.save_for_backward(tenIn, tenFlow)
        return tenOut

    @staticmethod
    def backward(ctx, tenOutgrad):
        tenIn, tenFlow = ctx.saved_tensors
        N, C, H, W = tenIn.shape
        tenOutgrad = tenOutgrad.contiguous(); assert(tenOutgrad.is_cuda == True)

        tenIngrad = torch.zeros_like(tenIn) if ctx.needs_input_grad[0] else None
        tenFlowgrad = torch.zeros_like(tenFlow) if ctx.needs_input_grad[1] else None

        if tenIngrad is not None:
            grid_ingrad = lambda meta: (triton.cdiv(N * C * H * W, meta['BLOCK_SIZE']),)
            softsplat_bwd_kernel_ingrad[grid_ingrad](
                tenIn, tenFlow, tenOutgrad, tenIngrad, N, C, H, W, BLOCK_SIZE=1024
            )

        if tenFlowgrad is not None:
            grid_flowgrad = lambda meta: (triton.cdiv(N * 2 * H * W, meta['BLOCK_SIZE']),)
            softsplat_bwd_kernel_flowgrad[grid_flowgrad](
                tenIn, tenFlow, tenOutgrad, tenFlowgrad, N, C, H, W, BLOCK_SIZE=1024
            )

        return tenIngrad, tenFlowgrad
