import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt.linalg import batch_linalg as lalg

from metrics import Metric, MetricType

def tangent_vector(x):
    n = x.size(dim=0) + x.size(dim=1)
    p = x.size(dim=0)
    a = torch.zeros(n,n)
    a[:p,p:] = x
    a[p:,:p] = -torch.transpose(x,0,1) # -x.t()    
    return a

def tangent_vector_batch(x):
    n = x.size(dim=2) + x.size(dim=3)
    p = x.size(dim=2)
    a = torch.zeros(x.size(dim=0),x.size(dim=1),n,n)
    a[:,:,:p,p:] = x
    a[:,:,p:,:p] = -torch.transpose(x,2,3) # -x.t()    
    return a

def gr_identity(n, p):
    a = torch.eye(n)
    da = torch.zeros(n)
    da[:p] = 1
    a.as_strided([n], [n + 1]).copy_(da)
    return a

def cayley_map(X):
    n = X.size(-1)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)    
    return (Id - X) @ torch.inverse(Id + X)
    
def cayley_map_batch(X):
    n = X.size(-1)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)    
    #return (Id - X) @ torch.inverse(Id + X)
    
    Y = Id + X
    inv_Y = torch.zeros_like(Y, requires_grad=False)
    for i in range(Y.size(dim=0)):    
        for j in range(Y.size(dim=1)):    
            #inv_Y[i,j,:,:] = torch.inverse(torch.squeeze(Y[i,:,:],dim=0))
            inv_Y[i,j,:,:] = torch.inverse(Y[i,j,:,:])

    return torch.matmul(Id-X, inv_Y)

def productory(factors: torch.Tensor, dim=1) -> torch.Tensor:
    m = factors.size(dim)
    acum = factors.select(dim=dim, index=0)
    for i in range(1, m):
        current = factors.select(dim=dim, index=i)
        acum = acum @ current
    return acum

def artanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def p_exp_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    return torch.tanh(normv)*v/normv

def p_log_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10, 1-1e-5)
    return artanh(normv)*v/normv

def full_p_exp_map(x, v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    y = torch.tanh(normv/(1-sqxnorm)) * v/normv
    return p_sum(x, y)

def p_sum(x, y):
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5)
    dotxy = torch.sum(x*y, dim=-1, keepdim=True)
    numerator = (1+2*dotxy+sqynorm)*x + (1-sqxnorm)*y
    denominator = 1 + 2*dotxy + sqxnorm*sqynorm
    return numerator/denominator

class PairwiseMatman(nn.Module):
    def __init__(self, model):
        super(PairwiseMatman, self).__init__()
        self.matmanModel = model        
        self.posModel = self.matmanModel        
        self.negModel = self.matmanModel

    def forward(self, input):
        pos = self.posModel(input[0])
        neg = self.negModel(input[1])        
        combine = torch.cat([pos, neg], 0)
        return combine


class GrSca(nn.Module):
    def __init__(self, config):
        super(GrSca, self).__init__()

        INIT_EPS = 1e-3
        n_dim = config.n_dim
        p_dim = config.p_dim
        vocab_size = config.vocab_size

        init_fn = lambda n_points: torch.randn((n_points, p_dim, n_dim - p_dim)) * INIT_EPS
        self.question_embedding = torch.nn.Parameter(init_fn(vocab_size), requires_grad=True)
        self.answer_embedding   = torch.nn.Parameter(init_fn(vocab_size), requires_grad=True)

        self.question_transforms = torch.nn.Parameter(torch.rand((p_dim, n_dim - p_dim)) * 2 - 1.0, requires_grad=True)
        self.question_bias       = torch.nn.Parameter(torch.rand((p_dim, n_dim - p_dim)) * 2 - 1.0, requires_grad=True) 

        self.wf = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) 
        self.wb = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) 

        base_point = gr_identity(n_dim, p_dim) 
        self.register_buffer('base_point', base_point)

    def addition_tangents(self, x):        
        y     = cayley_map(x)
        n_tokens = y.size(dim=1)
        z     = y[:,0,:,:]        
        for i in range(1, n_tokens):
            z     = torch.matmul(z, y[:,i,:,:])

        inv_z = torch.inverse(z) 
        
        return torch.matmul( z, torch.matmul(self.base_point, inv_z) )

    def addition_tangent_point(self, t, x):
        y = cayley_map(t)
        inv_y = torch.inverse(y)    
        return torch.matmul(y, torch.matmul(x, inv_y) )

    def dist(self, x, y):
        return x - y

    def get_factors(self, x):
        question_ids = x.sentence_1
        answer_ids   = x.sentence_2

        question_token_emb = self.question_embedding[question_ids]
        answer_token_emb   = self.answer_embedding[answer_ids]
        return question_token_emb, answer_token_emb

    def forward(self, x):        
        question_ids = x.sentence_1
        answer_ids   = x.sentence_2

        question_bias       = tangent_vector(self.question_bias)        

        question_token_emb = self.question_transforms * self.question_embedding[question_ids]      
        question_token_emb = tangent_vector_batch(question_token_emb)                              

        question_emb = self.addition_tangents(question_token_emb)                           
        question_emb = self.addition_tangent_point(question_bias, question_emb)                                    

        answer_token_emb   = tangent_vector_batch(self.answer_embedding[answer_ids])        
        answer_emb   = self.addition_tangents(answer_token_emb)                             

        d_log = self.dist(question_emb, answer_emb)                                        

        dist = torch.norm(d_log.view(d_log.size(0),-1), p=2, dim=1)

        sim_scores = -self.wf * dist + self.wb

        return sim_scores
# End of GrSca class


class SpdScaGr(nn.Module):
    def __init__(self, config):
        super(SpdScaGr, self).__init__()

        INIT_EPS = 1e-3
        emb_dim = config.emb_dim
        vocab_size = config.vocab_size
        self.dist_factor = config.dist_factor

        init_fn_spd = lambda n_points: torch.randn((n_points, emb_dim, emb_dim)) * INIT_EPS
        self.question_embedding_spd = torch.nn.Parameter(init_fn_spd(vocab_size), requires_grad=True)
        self.answer_embedding_spd   = torch.nn.Parameter(init_fn_spd(vocab_size), requires_grad=True)

        self.question_transforms_spd = torch.nn.Parameter(torch.rand((emb_dim, emb_dim)) * 2 - 1.0, requires_grad=True)
        self.question_bias_spd       = torch.nn.Parameter(torch.rand((emb_dim, emb_dim)) * 2 - 1.0, requires_grad=True) 

        self.wf = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) 
        self.wb = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) 

        metric=MetricType.from_str(config.metric)
        self.metric = Metric.get(metric.value, emb_dim)

        n_dim = config.n_dim
        p_dim = config.p_dim

        init_fn_gr = lambda n_points: torch.randn((n_points, p_dim, n_dim - p_dim)) * INIT_EPS
        self.question_embedding_gr = torch.nn.Parameter(init_fn_gr(vocab_size), requires_grad=True)
        self.answer_embedding_gr   = torch.nn.Parameter(init_fn_gr(vocab_size), requires_grad=True)

        self.question_transforms_gr = torch.nn.Parameter(torch.rand((p_dim, n_dim - p_dim)) * 2 - 1.0, requires_grad=True)
        self.question_bias_gr       = torch.nn.Parameter(torch.rand((p_dim, n_dim - p_dim)) * 2 - 1.0, requires_grad=True)         

        base_point = gr_identity(n_dim, p_dim) 
        self.register_buffer('base_point', base_point)

    def addition_tangents_spd(self, x):        
        n_tokens = x.size(dim=1)
        y = lalg.sym_funcm(x[:,n_tokens-1,:,:], torch.exp)
        for i in range(n_tokens-1):
            sqrt_x = lalg.sym_funcm(x[:,n_tokens-i-2,:,:], lambda t: torch.sqrt(torch.exp(t)))
            y = sqrt_x @ y @ sqrt_x

        return y

    def addition_point_tangent_spd(self, x, t):
        y = lalg.sym_funcm(t, torch.exp)
        sqrt_x = lalg.sym_funcm(x, torch.sqrt)
        return sqrt_x @ y @ sqrt_x

    def scaling_tangent_point_spd(self, t, x):
        log_x = lalg.sym_funcm(x, torch.log)
        return lalg.sym_funcm(t*log_x, torch.exp) 

    def dist_spd(self, a, b):
        inv_sqrt_a = lalg.sym_inv_sqrtm1(a)                                    
        m = inv_sqrt_a @ b @ inv_sqrt_a
        m = lalg.sym(m)         

        try:
            eigvals, _ = torch.symeig(m, eigenvectors=True)                 
        except RuntimeError as e:
            log = get_logging()
            log.info(f"ERROR: torch.symeig in SPD dist did not converge. m = {m}")
            raise e

        log_eigvals = torch.log(eigvals)                                       
        res = self.metric.compute_metric(log_eigvals, keepdim=True)
        return res

    def addition_tangents_gr(self, x):        
        y     = cayley_map(x)
        n_tokens = y.size(dim=1)
        z     = y[:,0,:,:]        
        for i in range(1, n_tokens):
            z     = torch.matmul(z, y[:,i,:,:])

        inv_z = torch.inverse(z) 
        
        return torch.matmul( z, torch.matmul(self.base_point, inv_z) )

    def addition_tangent_point_gr(self, t, x):
        y = cayley_map(t)
        inv_y = torch.inverse(y)    
        return torch.matmul(y, torch.matmul(x, inv_y) )

    def dist_gr(self, x, y):
        return x - y

    def forward(self, x):        
        question_ids = x.sentence_1
        answer_ids   = x.sentence_2

        question_transforms_spd = lalg.sym(self.question_transforms_spd)
        question_bias_spd       = lalg.sym(self.question_bias_spd)

        question_token_emb_spd = lalg.sym(self.question_embedding_spd[question_ids])                    
        question_emb_spd = self.addition_tangents_spd(question_token_emb_spd)         
                            
        question_emb_spd = self.scaling_tangent_point_spd(question_transforms_spd, question_emb_spd)    
        question_emb_spd = self.addition_point_tangent_spd(question_emb_spd, question_bias_spd)

        answer_token_emb_spd   = lalg.sym(self.answer_embedding_spd[answer_ids])
        answer_emb_spd   = self.addition_tangents_spd(answer_token_emb_spd)           

        dist_spd = self.dist_spd(question_emb_spd, answer_emb_spd)

        question_bias_gr       = tangent_vector(self.question_bias_gr)        

        question_token_emb_gr = self.question_transforms_gr * self.question_embedding_gr[question_ids]      
        question_token_emb_gr = tangent_vector_batch(question_token_emb_gr)                         

        question_emb_gr = self.addition_tangents_gr(question_token_emb_gr)                  
        question_emb_gr = self.addition_tangent_point_gr(question_bias_gr, question_emb_gr)                                   

        answer_token_emb_gr   = tangent_vector_batch(self.answer_embedding_gr[answer_ids])  
        answer_emb_gr   = self.addition_tangents_gr(answer_token_emb_gr)                    

        d_log_gr = self.dist_gr(question_emb_gr, answer_emb_gr)                             

        dist_gr = torch.norm(d_log_gr.view(d_log_gr.size(0),-1), p=2, dim=1)

        dist_spd = torch.squeeze(dist_spd)

        dist = self.dist_factor * dist_spd + dist_gr

        sim_scores = -self.wf * dist + self.wb

        return sim_scores
# End of SpdScaGr class

class SpdRotGr(nn.Module):
    def __init__(self, config):
        super(SpdRotGr, self).__init__()

        INIT_EPS = 1e-3
        emb_dim = config.emb_dim
        n_dim = config.n_dim
        p_dim = config.p_dim
        vocab_size = config.vocab_size
        self.dist_factor = config.dist_factor

        self.emb_dim = emb_dim

        init_fn_spd = lambda n_points: torch.randn((n_points, emb_dim, emb_dim)) * INIT_EPS
        self.question_embedding_spd = torch.nn.Parameter(init_fn_spd(vocab_size), requires_grad=True)
        self.answer_embedding_spd   = torch.nn.Parameter(init_fn_spd(vocab_size), requires_grad=True)

        self.n_isom = emb_dim * (emb_dim - 1) // 2
        self.isom_init = lambda n: torch.rand((n, self.n_isom)) * 0.5 - 0.25    
        self.embed_index = self.get_isometry_embed_index(emb_dim)
        self.ref_params = torch.nn.Parameter(self.isom_init(1), requires_grad=True)

        self.question_bias_spd       = torch.nn.Parameter(torch.rand((emb_dim, emb_dim)) * 2 - 1.0, requires_grad=True)    

        metric=MetricType.from_str(config.metric)
        self.metric = Metric.get(metric.value, emb_dim)     

        init_fn_gr = lambda n_points: torch.randn((n_points, p_dim, n_dim - p_dim)) * INIT_EPS
        self.question_embedding_gr = torch.nn.Parameter(init_fn_gr(vocab_size), requires_grad=True)
        self.answer_embedding_gr   = torch.nn.Parameter(init_fn_gr(vocab_size), requires_grad=True)

        self.question_transforms_gr = torch.nn.Parameter(torch.rand((p_dim, n_dim - p_dim)) * 2 - 1.0, requires_grad=True)
        self.question_bias_gr       = torch.nn.Parameter(torch.rand((p_dim, n_dim - p_dim)) * 2 - 1.0, requires_grad=True)   

        self.wf = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) 
        self.wb = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)       

        base_point = gr_identity(n_dim, p_dim) 
        self.register_buffer('base_point', base_point)

    def get_isometry_embed_index(self, dims):        
        # indexes := 1 <= i < j < n. Using 1-based notation to make it equivalent to matrix math notation
        indexes = [(i, j) for i in range(1, dims + 1) for j in range(i + 1, dims + 1)]

        embed_index = []
        for i, j in indexes:
            row = []
            for c_i, c_j in [(i, i), (i, j), (j, i), (j, j)]:  # 4 combinations that we care for each (i, j) pair
                flatten_index = dims * (c_i - 1) + c_j - 1
                row.append(flatten_index)
            embed_index.append(row)
        return torch.LongTensor(embed_index).unsqueeze(0)

    def build_relation_isometry_matrices(self, isom_params):
        """
        :param isom_params: r x m x 4
        :return: r x n x n
        """
        # isom_params = self.compute_rotation_params(self.rot_params)  # r x m x 4
        embeded_rotations = self.embed_params(isom_params, self.emb_dim)  # r x m x n x n
        isom_rot = productory(embeded_rotations)  # r x n x n
        return isom_rot

    def embed_params(self, iso_params: torch.Tensor, dims: int) -> torch.Tensor:
        """        
        :param iso_params: b x m x 4, where m = dims * (dims - 1) / 2, which is the amount of isometries
        :param dims: (also called n) dimension of output identities, with params embedded
        :return: b x m x n x n
        """
        bs, m, _ = iso_params.size()
        target = torch.eye(dims, requires_grad=True, device=iso_params.device)
        target = target.reshape(1, 1, dims * dims).repeat(bs, m, 1)  # b x m x n * n
        scatter_index = self.embed_index.repeat(bs, 1, 1)  # b x m x 4
        embed_isometries = target.scatter(dim=-1, index=scatter_index, src=iso_params)  # b x m x n * n
        embed_isometries = embed_isometries.reshape(bs, m, dims, dims)  # b x m x n x n
        return embed_isometries 

    def get_isometry_params(self):
        """
        :return: tensor of r x m x 4
        """
        return self.compute_rotation_params(self.ref_params)

    def compute_rotation_params(self, params):
        """        
        :param params: b x m
        :return: b x m x 4
        """
        cos_x = torch.cos(params)
        sin_x = torch.sin(params)
        res = torch.stack([cos_x, -sin_x, sin_x, cos_x], dim=-1)     
        return res  

    def addition_tangents(self, x):        
        n_tokens = x.size(dim=1)
        y = lalg.sym_funcm(x[:,n_tokens-1,:,:], torch.exp)
        for i in range(n_tokens-1):
            sqrt_x = lalg.sym_funcm(x[:,n_tokens-i-2,:,:], lambda t: torch.sqrt(torch.exp(t)))
            y = sqrt_x @ y @ sqrt_x

        return y

    def addition_point_tangent(self, x, t):
        y = lalg.sym_funcm(t, torch.exp)
        sqrt_x = lalg.sym_funcm(x, torch.sqrt)
        return sqrt_x @ y @ sqrt_x

    def scaling_tangent_point(self, t, x):
        log_x = lalg.sym_funcm(x, torch.log)
        return lalg.sym_funcm(t*log_x, torch.exp) 

    def dist_spd(self, a, b):
        inv_sqrt_a = lalg.sym_inv_sqrtm1(a)                                    
        m = inv_sqrt_a @ b @ inv_sqrt_a
        m = lalg.sym(m)         

        try:
            eigvals, _ = torch.symeig(m, eigenvectors=True)                 
        except RuntimeError as e:
            log = get_logging()
            log.info(f"ERROR: torch.symeig in SPD dist did not converge. m = {m}")
            raise e

        log_eigvals = torch.log(eigvals)                                       
        res = self.metric.compute_metric(log_eigvals, keepdim=True)
        return res

    def addition_tangents_gr(self, x):        
        y     = cayley_map(x)
        n_tokens = y.size(dim=1)
        z     = y[:,0,:,:]        
        for i in range(1, n_tokens):
            z     = torch.matmul(z, y[:,i,:,:])

        inv_z = torch.inverse(z) 
        
        return torch.matmul( z, torch.matmul(self.base_point, inv_z) )

    def addition_tangent_point_gr(self, t, x):
        y = cayley_map(t)
        inv_y = torch.inverse(y)    
        return torch.matmul(y, torch.matmul(x, inv_y) )

    def dist_gr(self, x, y):
        return x - y

    def forward(self, x):        
        question_ids = x.sentence_1
        answer_ids   = x.sentence_2
        
        question_bias_spd       = lalg.sym(self.question_bias_spd)

        question_token_emb_spd = lalg.sym(self.question_embedding_spd[question_ids])                    
        question_emb_spd = self.addition_tangents(question_token_emb_spd)         
                                        
        isometry_params = self.get_isometry_params()
        all_relation_isometries = self.build_relation_isometry_matrices(isometry_params)  
        rel_isometries = all_relation_isometries[0]

        question_emb_spd = rel_isometries @ question_emb_spd @ rel_isometries.transpose(-1,-2)
        question_emb_spd = self.addition_point_tangent(question_emb_spd, question_bias_spd)

        answer_token_emb_spd   = lalg.sym(self.answer_embedding_spd[answer_ids])
        answer_emb_spd   = self.addition_tangents(answer_token_emb_spd)           

        dist_spd = self.dist_spd(question_emb_spd, answer_emb_spd)
         
        question_bias_gr       = tangent_vector(self.question_bias_gr)        

        question_token_emb_gr = self.question_transforms_gr * self.question_embedding_gr[question_ids]      
        question_token_emb_gr = tangent_vector_batch(question_token_emb_gr)                         

        question_emb_gr = self.addition_tangents_gr(question_token_emb_gr)                        
        question_emb_gr = self.addition_tangent_point_gr(question_bias_gr, question_emb_gr)
                                   
        answer_token_emb_gr   = tangent_vector_batch(self.answer_embedding_gr[answer_ids])     
        answer_emb_gr   = self.addition_tangents_gr(answer_token_emb_gr)                       

        d_log_gr = self.dist_gr(question_emb_gr, answer_emb_gr)                                      

        dist_gr = torch.norm(d_log_gr.view(d_log_gr.size(0),-1), p=2, dim=1)
        
        dist_spd = torch.squeeze(dist_spd)

        dist = self.dist_factor * dist_spd + dist_gr

        sim_scores = -self.wf * dist + self.wb

        return sim_scores
# End of SpdRotGr class


class SpdRefGr(nn.Module):
    def __init__(self, config):
        super(SpdRefGr, self).__init__()

        INIT_EPS = 1e-3
        emb_dim = config.emb_dim
        n_dim = config.n_dim
        p_dim = config.p_dim
        vocab_size = config.vocab_size
        self.dist_factor = config.dist_factor

        self.emb_dim = emb_dim

        init_fn_spd = lambda n_points: torch.randn((n_points, emb_dim, emb_dim)) * INIT_EPS
        self.question_embedding_spd = torch.nn.Parameter(init_fn_spd(vocab_size), requires_grad=True)
        self.answer_embedding_spd   = torch.nn.Parameter(init_fn_spd(vocab_size), requires_grad=True)

        self.n_isom = emb_dim * (emb_dim - 1) // 2
        self.isom_init = lambda n: torch.rand((n, self.n_isom)) * 0.5 - 0.25         
        self.embed_index = self.get_isometry_embed_index(emb_dim)
        self.ref_params = torch.nn.Parameter(self.isom_init(1), requires_grad=True)

        self.question_bias_spd       = torch.nn.Parameter(torch.rand((emb_dim, emb_dim)) * 2 - 1.0, requires_grad=True)     

        metric=MetricType.from_str(config.metric)
        self.metric = Metric.get(metric.value, emb_dim)    

        init_fn_gr = lambda n_points: torch.randn((n_points, p_dim, n_dim - p_dim)) * INIT_EPS
        self.question_embedding_gr = torch.nn.Parameter(init_fn_gr(vocab_size), requires_grad=True)
        self.answer_embedding_gr   = torch.nn.Parameter(init_fn_gr(vocab_size), requires_grad=True)

        self.question_transforms_gr = torch.nn.Parameter(torch.rand((p_dim, n_dim - p_dim)) * 2 - 1.0, requires_grad=True)
        self.question_bias_gr       = torch.nn.Parameter(torch.rand((p_dim, n_dim - p_dim)) * 2 - 1.0, requires_grad=True)   

        self.wf = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) 
        self.wb = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)       

        base_point = gr_identity(n_dim, p_dim) 
        self.register_buffer('base_point', base_point)

    def get_isometry_embed_index(self, dims):        
        # indexes := 1 <= i < j < n. Using 1-based notation to make it equivalent to matrix math notation
        indexes = [(i, j) for i in range(1, dims + 1) for j in range(i + 1, dims + 1)]

        embed_index = []
        for i, j in indexes:
            row = []
            for c_i, c_j in [(i, i), (i, j), (j, i), (j, j)]:  # 4 combinations that we care for each (i, j) pair
                flatten_index = dims * (c_i - 1) + c_j - 1
                row.append(flatten_index)
            embed_index.append(row)
        return torch.LongTensor(embed_index).unsqueeze(0)

    def build_relation_isometry_matrices(self, isom_params):
        """
        :param isom_params: r x m x 4
        :return: r x n x n
        """
        # isom_params = self.compute_rotation_params(self.rot_params)  # r x m x 4
        embeded_rotations = self.embed_params(isom_params, self.emb_dim)  # r x m x n x n
        isom_rot = productory(embeded_rotations)  # r x n x n
        return isom_rot

    def embed_params(self, iso_params: torch.Tensor, dims: int) -> torch.Tensor:
        """        
        :param iso_params: b x m x 4, where m = dims * (dims - 1) / 2, which is the amount of isometries
        :param dims: (also called n) dimension of output identities, with params embedded
        :return: b x m x n x n
        """
        bs, m, _ = iso_params.size()
        target = torch.eye(dims, requires_grad=True, device=iso_params.device)
        target = target.reshape(1, 1, dims * dims).repeat(bs, m, 1)  # b x m x n * n
        scatter_index = self.embed_index.repeat(bs, 1, 1)  # b x m x 4
        embed_isometries = target.scatter(dim=-1, index=scatter_index, src=iso_params)  # b x m x n * n
        embed_isometries = embed_isometries.reshape(bs, m, dims, dims)  # b x m x n x n
        return embed_isometries 

    def get_isometry_params(self):
        """
        :return: tensor of r x m x 4
        """
        return self.compute_reflection_params(self.ref_params)

    def compute_reflection_params(self, params):
        """        
        :param params: b x m
        :return: b x m x 4
        """
        cos_x = torch.cos(params)
        sin_x = torch.sin(params)
        res = torch.stack([cos_x, sin_x, sin_x, -cos_x], dim=-1)
        return res  

    def addition_tangents(self, x):        
        n_tokens = x.size(dim=1)
        y = lalg.sym_funcm(x[:,n_tokens-1,:,:], torch.exp)
        for i in range(n_tokens-1):
            sqrt_x = lalg.sym_funcm(x[:,n_tokens-i-2,:,:], lambda t: torch.sqrt(torch.exp(t)))
            y = sqrt_x @ y @ sqrt_x

        return y

    def addition_point_tangent(self, x, t):
        y = lalg.sym_funcm(t, torch.exp)
        sqrt_x = lalg.sym_funcm(x, torch.sqrt)
        return sqrt_x @ y @ sqrt_x

    def scaling_tangent_point(self, t, x):
        log_x = lalg.sym_funcm(x, torch.log)
        return lalg.sym_funcm(t*log_x, torch.exp) 

    def dist_spd(self, a, b):
        inv_sqrt_a = lalg.sym_inv_sqrtm1(a)                                    
        m = inv_sqrt_a @ b @ inv_sqrt_a
        m = lalg.sym(m)         

        try:
            eigvals, _ = torch.symeig(m, eigenvectors=True)                 
        except RuntimeError as e:
            log = get_logging()
            log.info(f"ERROR: torch.symeig in SPD dist did not converge. m = {m}")
            raise e

        log_eigvals = torch.log(eigvals)                                       
        res = self.metric.compute_metric(log_eigvals, keepdim=True)
        return res

    def addition_tangents_gr(self, x):        
        y     = cayley_map(x)
        n_tokens = y.size(dim=1)
        z     = y[:,0,:,:]        
        for i in range(1, n_tokens):
            z     = torch.matmul(z, y[:,i,:,:])

        inv_z = torch.inverse(z) 
        
        return torch.matmul( z, torch.matmul(self.base_point, inv_z) )

    def addition_tangent_point_gr(self, t, x):
        y = cayley_map(t)
        inv_y = torch.inverse(y)    
        return torch.matmul(y, torch.matmul(x, inv_y) )

    def dist_gr(self, x, y):
        return x - y

    def forward(self, x):        
        question_ids = x.sentence_1
        answer_ids   = x.sentence_2
       
        question_bias_spd       = lalg.sym(self.question_bias_spd)

        question_token_emb_spd = lalg.sym(self.question_embedding_spd[question_ids])                    
        question_emb_spd = self.addition_tangents(question_token_emb_spd)         
                                        
        isometry_params = self.get_isometry_params()
        all_relation_isometries = self.build_relation_isometry_matrices(isometry_params)  
        rel_isometries = all_relation_isometries[0]

        question_emb_spd = rel_isometries @ question_emb_spd @ rel_isometries.transpose(-1,-2)
        question_emb_spd = self.addition_point_tangent(question_emb_spd, question_bias_spd)

        answer_token_emb_spd   = lalg.sym(self.answer_embedding_spd[answer_ids])
        answer_emb_spd   = self.addition_tangents(answer_token_emb_spd)           

        dist_spd = self.dist_spd(question_emb_spd, answer_emb_spd)
       
        question_bias_gr       = tangent_vector(self.question_bias_gr)        

        question_token_emb_gr = self.question_transforms_gr * self.question_embedding_gr[question_ids]      
        question_token_emb_gr = tangent_vector_batch(question_token_emb_gr)                        

        question_emb_gr = self.addition_tangents_gr(question_token_emb_gr)                        
        question_emb_gr = self.addition_tangent_point_gr(question_bias_gr, question_emb_gr)
                                   
        answer_token_emb_gr   = tangent_vector_batch(self.answer_embedding_gr[answer_ids])     
        answer_emb_gr   = self.addition_tangents_gr(answer_token_emb_gr)                       

        d_log_gr = self.dist_gr(question_emb_gr, answer_emb_gr)                                      

        dist_gr = torch.norm(d_log_gr.view(d_log_gr.size(0),-1), p=2, dim=1)
        
        dist_spd = torch.squeeze(dist_spd)

        dist = self.dist_factor * dist_spd + dist_gr

        sim_scores = -self.wf * dist + self.wb

        return sim_scores
# End of SpdRefGr class
