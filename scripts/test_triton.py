import torch, triton, triton.language as tl

print("triton", triton.__version__, "| torch", torch.__version__)


@triton.jit
def addk(x, y, o, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o + off, tl.load(x + off, mask=m) + tl.load(y + off, mask=m), mask=m)


a = torch.randn(4096, device="cuda")
b = torch.randn(4096, device="cuda")
o = torch.empty_like(a)
try:
    addk[(4,)](a, b, o, 4096, BLOCK=1024)
    torch.cuda.synchronize()
    print("KERNEL OK:", bool(torch.allclose(o, a + b)))
except Exception as e:
    print("KERNEL FAIL:", type(e).__name__, str(e)[:400])
