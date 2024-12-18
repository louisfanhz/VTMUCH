import torch

def calc_pairwise_cos_sim_batch(a, b, eps=1e-08):
    an = 1 / torch.norm(a, dim=-1).clamp(min=eps).unsqueeze(-1)
    bn = 1 / torch.norm(b, dim=-1).clamp(min=eps).unsqueeze(-1)

    denom = torch.bmm(an, bn.transpose(-1, -2))
    numer = torch.bmm(a, b.transpose(-1, -2))
    result = torch.multiply(numer, denom)

    return result

def calc_pairwise_cos_sim(a, b, eps=1e-08):
    """ return the cosine similarity between tensors in batch a and b. """
    an = torch.norm(a, dim=-1).clamp(min=eps).unsqueeze(1)
    bn = torch.norm(b, dim=-1).clamp(min=eps).unsqueeze(1)
    result = torch.div(a.mm(b.T), (an.mm(bn.T)))

    return result

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def calc_mAP_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    mAP = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        # gnd > 0 if element of query and element of retrieval have common label
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        # rank the hamming dist of every query data with all retrieval data
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))   # total # of pairs that have common label
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        mAP += torch.mean(count / tindex)

    mAP = mAP / num_query

    return mAP