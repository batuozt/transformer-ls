import warnings
import torch
import torch.nn as nn
import math
import numpy as np
import random
import time
from pytorch_wavelets import DWT1D

def _get_module(x):
    if torch.is_tensor(x):
        return torch
    else:
        return np

def _build_generator(module, seed=None, device="cpu"):
    if module == np:
        return np.random.RandomState(seed)
    elif module == torch:
        rng = torch.Generator(device)
        if seed is not None:
            rng = rng.manual_seed(seed)
        return rng
    else:
        raise ValueError("Module cp not supported")

def sample_tensor(A,ids,mode):
    """
    Args:
        A: Input tensor, shape b, h, n, d
        ids: Indices to sample, shape b*h, n, m
        Mode: Right or left. Right implements AS for tensors, left implements S^TA !!!!!
    """
    if mode == 'right':
        b, h, n, d = A.shape
        m = ids.shape[-1]
        A = A.reshape(b*h,n,d)
        idxs_3d = ids.unsqueeze(dim = 1)
        idxs_3d = idxs_3d.repeat(1,n,1)
        A_ids = torch.gather(A, 2, idxs_3d)
        A_ids = A_ids.reshape(b,h,n,m)
    elif mode == 'left':
        b, h, n, d = A.shape
        m = ids.shape[-1]
        A = A.reshape(b*h,n,d)
        A = A.transpose(1,2)
        idxs_3d = ids.unsqueeze(dim = 1)
        idxs_3d = idxs_3d.repeat(1,d,1)
        A_ids = torch.gather(A, 2, idxs_3d)
        A_ids = A_ids.transpose(1,2)
        A_ids = A_ids.reshape(b,h,m,d)
    return A_ids

def sample_pdf(pdf, m, replacement=True, seed=None, rng=None):
    """"Returns indices in range ``[0, len(pdf))`` sampled from pdf.
    Args:
        pdf (array-like): 1D representation of PDF.
        m (int): Number of samples.
        replacement (bool, optional): If ``True``, samples with replacement.
        seed (int, optional): Random seed to use.
        rng (``numpy.random.RandomState`` or ``torch.Generator``):
    """
    is_tensor = torch.is_tensor(pdf)
    if seed is not None and rng is not None:
        warnings.warn("specified both random seed and generator - defaulting to rng")
        seed = None
    if seed is not None:
        rng = _build_generator(_get_module(pdf), seed=seed, device=pdf.device if is_tensor else None)

    if is_tensor:
        idxs = torch.multinomial(pdf, num_samples=m, replacement=replacement, generator=rng)
    else:
        idxs = rng.choice(range(0, len(pdf)), p=pdf, size=m, replace=replacement)
    return idxs
    
def compute_attn(Q, K, relu=False):
    #A = (Q @ K.T) / torch.sqrt(d)
    d = Q.shape[-1]
    A = torch.matmul(Q, K.transpose(-1, -2)) / (d ** 0.5)
    #relu = True
    if relu:
        A = A * (A > 0)
    #A = softmax(A, axis=1)
    A = torch.nn.functional.softmax(A, dim = -1)
    return A

class SketchingAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.sketch_QT = config["sketch_QT"]
        self.sketch_dim_QT = config["sketch_dim_QT"]
        self.sketch_dim_AV = config["sketch_dim_AV"]
        self.method = config["sketching_method"]
        self.sampling = config["sampling_method"]
        self.num_scales = config["num_scales"]
        self.mask = False
        self.use_relu = False
        self.sketch_AV = True
        self.skip_V = True
        self.seed = 12345
        self.seq_len = config["seq_len"]
        self.additive = True
        self.dwt = DWT1D(wave='haar', J=3)
        
        if self.additive == True:
            init_weight = torch.tensor([1.0], dtype=torch.float32)
            self.weights = nn.ParameterList(
                    [torch.nn.Parameter(init_weight) for _ in range(self.num_scales)]
                )
            
        
        self.use_conv = "conv_kernel_size" in config
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (config["conv_kernel_size"], 1), padding = (config["conv_kernel_size"] // 2, 0),
                bias = False,
                groups = self.num_head)

    def forward(self, Q, K, V, mask):
        #t = time.time()
        if self.mask == True: 
            Q = Q * mask[:, None, :, None] 
            K = K * mask[:, None, :, None] 
        
        m1 = self.sketch_dim_QT  
        m2 = self.sketch_dim_AV  
        
        seed = self.seed
        method_params = {}
        m = m1
        sketch_QT = self.sketch_QT
        sketch_AV = self.sketch_AV
        use_relu = self.use_relu
        
        if sketch_QT:
            QS, S = self.right_sketch_3D(Q, K, int(m1), self.method, seed=seed, return_sketch_or_pdf=True, **method_params)
            KS = sample_tensor(K,S,'right') 
            #A_est = compute_attn(QS, KS, relu=use_relu)
        else:
            KS = K
            QS = Q
            #A_est = compute_attn(Q, K, relu=use_relu)
        
        KS = KS.transpose(-1,-2)
        
        if sketch_AV:
            if self.method == 'exact-norm-score':
                KS = KS.transpose(-1,-2)
                Att = compute_attn(QS, KS, relu=use_relu)
                
                A_est, S = self.right_sketch_3D(Att, V, int(m2), method="joint-norm-score", seed=2*seed, return_sketch_or_pdf=True, pdf=None,q=None)
                
                if self.method == "norm-score" or self.method == "joint-norm-score" or self.method == "exact-norm-score" or self.method == "uniform":
                    ST_V = sample_tensor(V,S,'left') 
                else: 
                    ST_V = S.transpose(-1,-2) @ V
                AV_est = A_est @ ST_V
            elif self.method == "multi-scale-averaging":
                
                for i in range(self.num_scales):
                    SKS, S = self.right_sketch_3D(KS, V, int(m2), method="averaging", seed=2*seed, return_sketch_or_pdf=True, pdf=None,q=None)
                    SKS = SKS.transpose(-1,-2)
                    
                    A_est = compute_attn(QS, SKS, relu=use_relu)         
                    ST_V = V.reshape(-1, self.num_head, m2, self.seq_len // m2, self.head_dim).mean(dim = -2)
                    
                    m2 = int(m2/2)
                    if i == 0:
                        AV_est = self.weights[i] * A_est @ ST_V
                    else:
                        AV_est = AV_est + self.weights[i] * A_est @ ST_V
                
                '''
                for i in range(self.num_scales):
                    cur_SKS, S = self.right_sketch_3D(KS, V, int(m2), method="averaging", seed=2*seed, return_sketch_or_pdf=True, pdf=None,q=None)
                    m2 = int(m2/2)
                    if i == 0:
                        SKS = cur_SKS
                    else:
                        SKS = torch.cat([SKS,cur_SKS],dim = -1)
                SKS = SKS.transpose(-1,-2)
                A_est = compute_attn(QS, SKS, relu=use_relu)  
                
                m2 = self.sketch_dim_AV  
                
                for i in range(self.num_scales):
                    cur_ST_V = V.reshape(-1, self.num_head, m2, self.seq_len // m2, self.head_dim).mean(dim = -2)
                    m2 = int(m2/2)
                    if i == 0:
                        ST_V = cur_ST_V
                    else:
                        ST_V = torch.cat([ST_V,cur_ST_V],dim = -2)
                AV_est = A_est @ ST_V
                ''' 
            else:
                SKS, S = self.right_sketch_3D(KS, V, int(m2), method=self.method, seed=2*seed, return_sketch_or_pdf=True, pdf=None,q=None)
                SKS = SKS.transpose(-1,-2)
                
                A_est = compute_attn(QS, SKS, relu=use_relu)
    
                if self.method == "norm-score" or self.method == "joint-norm-score" or self.method == "uniform":
                    ST_V = sample_tensor(V,S,'left')
                elif self.method == "averaging":
                    ST_V = V.reshape(-1, self.num_head, m2, self.seq_len // m2, self.head_dim).mean(dim = -2)
                elif self.method == "wavelet":
                    V_t = torch.reshape(V,(-1,V.shape[-2],V.shape[-1]))
                    V_t = V_t.transpose(-1,-2)
                    ll, lh = self.dwt(V_t)
                    #VW = torch.cat([ll,lh[0],lh[1],lh[2]],dim=-1)
                    #VW = VW[:,:,0:V.shape[-2]]
                    VW = ll
                    #VW = torch.reshape(VW,(-1,V.shape[-3],V.shape[-2],V.shape[-1]))
                    #ST_V = VW.reshape(-1, self.num_head, m2, self.seq_len // m2, self.head_dim).mean(dim = -2)
                    ST_V = VW.view(V.shape[0],V.shape[-3],-1,V.shape[-1])
                else: 
                    ST_V = S.transpose(-1,-2) @ V
                
                AV_est = A_est @ ST_V
        else:
            AV_est = A_est @ V

        #t_diff = time.time() - t
        #print('Sketching forward takes:' + str.format('{0:.4f}', t_diff))
        
        if self.skip_V == True:
            AV_est = AV_est + V
        return AV_est
    
    def right_sketch_3D(self, A, B, m: int, method: str=None, pdf=None, q=None, seed=None, return_sketch_or_pdf=False):
        """Perform right sketching :math:`AS`.
        Args:
            A: The 4D tensor to sketch.
            B: The 4D tensor to left sketch. Only used to calculate joint norm scores. 
            m (int): The sketching dimension.
            method (str, optional): The method to use. One of the following:
                * 'norm-score': Sample based on column norm scores
                * 'joint-norm-score': Sample based on column norm scores of A and B
                * 'uniform': Uniform column sampling. Columns are scaled by 
                * 'gaussian': Generates gaussian sketching matrix with variance ``1/m``
        """
        
        assert len(A.shape) == 4
        
        b, h, n, d = A.shape
        is_tensor = torch.is_tensor(A)
        
        if is_tensor:
            A = torch.reshape(A,(-1,n,d))
            B = torch.reshape(B,(-1,n,d))

        op_mod = _get_module(A)
        rng = _build_generator(op_mod, seed=seed, device=A.device if is_tensor else None)

        S_shape = (b*h, d, m)
        if method == "norm-score":
            A2 = (A ** 2)
            pdf = torch.div(A2.sum(1).T , A2.sum((1,2)))
            pdf = pdf.T
        elif method == "joint-norm-score":
            #t4 = time.time()
            A2 = (A ** 2)
            B2 = (B ** 2)
            AB2 = A2.sum(1).T * B2.sum(1).T 
            weights = torch.sqrt(AB2) 
            pdf = torch.div(weights, weights.sum(0))
            pdf = pdf.T
            #t5 = time.time()
            #print('Calculating norms pdf takes:' + str.format('{0:.4f}', t5-t4))
        elif method == "averaging":
            AS = A.reshape(-1, self.num_head, m, self.seq_len // m, self.head_dim).mean(dim = -2)
            AS = AS.transpose(-1,-2)
        elif method == "wavelet":
            ll, lh = self.dwt(A)
            #AW = torch.cat([ll,lh[0],lh[1],lh[2]],dim=-1)
            #AW = AW[:,:,0:d]
            #AS = AW.reshape(-1, self.num_head, m, self.seq_len // m, self.head_dim).mean(dim = -2)
            AS = ll
            AS = AS.view(b,h,n,-1)
            #AS = AS.transpose(-1,-2)
        elif method == "uniform":
            shape = (A.shape[0],A.shape[-1]) 
            pdf = torch.ones(*shape, device=A.device) if is_tensor else np.ones(shape)
            pdf = pdf / (A.shape[0] * A.shape[-1])
        elif method == "gaussian":
            S = torch.randn(S_shape, device=A.device, generator=rng) if is_tensor else rng.randn(*S_shape)
            S = 1/math.sqrt(m) * S
        elif method == "sparse-gaussian":
            q = 0.5
            if q is None or (q<0) or (q>=1):
                raise ValueError("Expected `q` in range [0, 1)")
            S = torch.randn(S_shape, device=A.device, generator=rng) if is_tensor else rng.randn(*S_shape)
            mask = torch.rand(S_shape, device=A.device, generator=rng) if is_tensor else rng.rand(*S_shape)
            mask = (mask >= q)
            S = 1 / math.sqrt(m * (1-q)) * (S * mask)
        elif method == "rademacher" or method == "bernoulli-rademacher":
            is_bernoulli = method == "bernoulli-rademacher"
            if is_bernoulli:
                samples = [-1, 0, 1]
                p = [1/6, 2/3, 1/6]
            else:
                samples = [-1, 1]
                p = [1/2, 1/2]

            if is_tensor:
                samples = torch.Tensor(samples).to(A.device)
                p = torch.tensor(p).to(A.device)
                idxs = p.multinomial(num_samples=int(np.prod(S_shape)), replacement=True, generator=rng)
                S = samples[idxs].reshape(S_shape)
            else:
                S = rng.choice(samples, replace=True, size=S_shape, p=p)
            S = 1/math.sqrt(m) * S
            if is_bernoulli:
                S *= math.sqrt(3)
        elif method is not None:
            raise ValueError(f"method '{method}' not supported")

        if pdf is None:
            if method == "averaging" or method == "wavelet":
                sketch_or_pdf = None
            else:
                AS = A @ S
                AS = AS.view(b,h,n,m)
                S = S.view(b,h,d,m)
                sketch_or_pdf = S
        else:
            #t1 = time.time()
            if self.sampling == 'pdf':
                idxs = sample_pdf(pdf, m, rng=rng)
            elif self.sampling == 'topk':
                _,idxs = torch.topk(pdf,m)
            
            #t2 = time.time()
            #print('topk from pdf takes:' + str.format('{0:.4f}', t2-t1))
            pdf_ids = torch.gather(pdf, 1, idxs)
            pdf_ids = pdf_ids.unsqueeze(dim = 1)
            idxs_3d = idxs.unsqueeze(dim = 1)
            idxs_3d = idxs_3d.repeat(1,n,1)
            A_ids = torch.gather(A, 2, idxs_3d)
            sketch_or_pdf = idxs
            AS = 1 / op_mod.sqrt(m * pdf_ids) * A_ids
            AS = AS.view(b,h,n,m)
            #t3 = time.time()
            #print('Slicing the ids takes:' + str.format('{0:.4f}', t3-t2))
        if return_sketch_or_pdf:
            return AS, sketch_or_pdf
        else:
            return AS
    
    def right_sketch(self, A, m: int, method: str=None, pdf=None, q=None, seed=None, return_sketch_or_pdf=False):
        """Perform right sketching :math:`AS`.
        Args:
            A: The 2D matrix to sketch.
            m (int): The sketching dimension.
            method (str, optional): The method to use. One of the following:
                * 'norm-score': Sample based on column norm scores
                * 'uniform': Uniform column sampling. Columns are scaled by 
                * 'gaussian': Generates gaussian sketching matrix with variance ``1/m``
        """
        
        assert len(A.shape) == 2
        n, d = A.shape
        is_tensor = torch.is_tensor(A)

        op_mod = _get_module(A)
        rng = _build_generator(op_mod, seed=seed, device=A.device if is_tensor else None)

        S_shape = (d, m)
        if method == "norm-score":
            A2 = (A ** 2)
            pdf = A2.sum(0) / A2.sum()
        elif method == "uniform":
            shape = (A.shape[1],)
            pdf = torch.ones(*shape, device=A.device) if is_tensor else np.ones(shape)
            pdf = pdf / A.shape[1]
        elif method == "gaussian":
            S = torch.randn(S_shape, device=A.device, generator=rng) if is_tensor else rng.randn(*S_shape)
            S = 1/math.sqrt(m) * S
        elif method == "sparse-gaussian":
            if q is None or (q<0) or (q>=1):
                raise ValueError("Expected `q` in range [0, 1)")
            S = torch.randn(S_shape, device=A.device, generator=rng) if is_tensor else rng.randn(*S_shape)
            mask = torch.rand(S_shape, device=A.device, generator=rng) if is_tensor else rng.rand(*S_shape)
            mask = (mask >= q)
            S = 1 / math.sqrt(m * (1-q)) * (S * mask)
        elif method == "rademacher" or method == "bernoulli-rademacher":
            is_bernoulli = method == "bernoulli-rademacher"
            if is_bernoulli:
                samples = [-1, 0, 1]
                p = [1/6, 2/3, 1/6]
            else:
                samples = [-1, 1]
                p = [1/2, 1/2]

            if is_tensor:
                samples = torch.Tensor(samples).to(A.device)
                p = torch.tensor(p).to(A.device)
                idxs = p.multinomial(num_samples=int(np.prod(S_shape)), replacement=True, generator=rng)
                S = samples[idxs].reshape(S_shape)
            else:
                S = rng.choice(samples, replace=True, size=S_shape, p=p)
            S = 1/math.sqrt(m) * S
            if is_bernoulli:
                S *= math.sqrt(3)
        elif method is not None:
            raise ValueError(f"method '{method}' not supported")

        if pdf is None:
            AS = A @ S
            sketch_or_pdf = S
        else:
            idxs = sample_pdf(pdf, m, rng=rng)
            sketch_or_pdf = idxs
            AS = 1 / op_mod.sqrt(m * pdf[idxs]) * A[:, idxs]

        if return_sketch_or_pdf:
            return AS, sketch_or_pdf
        else:
            return AS

    def extra_repr(self):
        return f'seq_len={self.seq_len}'
