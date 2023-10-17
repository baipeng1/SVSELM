import torch
import torch.nn.functional as F
import numpy as np
def frame2phdur(mel2ph):
    maxvalue1 = mel2ph.max()
    # print(maxvalue1)
    ph = torch.zeros(mel2ph.shape[0], maxvalue1)
    for index in range(mel2ph.shape[0]):
        maxvalue2 = mel2ph[index].max()
        for i in range(1,maxvalue2+1):
            N = (mel2ph[index]==i).sum()
            ph[index, i - 1] = N
    # print(ph)
    phnozeroindex =torch.nonzero(ph)
    # print(phnozeroindex)
    # print(phnozeroindex.shape[0])
    # print(phnozeroindex[0])
    phnew = torch.index_select(ph, dim=1, index=phnozeroindex[0])
    # print(phnew)
    ph2 = torch.zeros([mel2ph.shape[0], maxvalue1],dtype=torch.int32)
    for i in range(ph.shape[0]):
        # print(ph[i])
        phnozeroindex2=torch.nonzero(ph[i])
        # print(phnozeroindex2)
        phnew = torch.index_select(ph[i], dim=0, index=phnozeroindex2.squeeze())
        # print(phnew)
        jidong = torch.zeros(maxvalue1-phnew.shape[0])
        # print(jidong)
        c = torch.cat((phnew, jidong), axis=0)
        # print(c)
        ph2[i] = c
    # print(ph2.type())
    return ph2


        # print(torch.cat((phnew,jidong),0))
        # phnew.resize(1,maxvalue1)
        # print(phnew)
    #     F.pad(phnew,0)
    #     print(phnew)
    #     ph2[i]=phnew
    # print("________")
    # print(ph2)
    # torch.index_select(ph,0,tensor([[0, 0]))
    # print(phnozeroindex.shape)
    # for location, yinsuvalue in enumerate(phnozeroindex[index].tolist()):
    # for index in range(phnozeroindex.shape[0]):
    #     phnew = torch.index_select(ph[index], dim=0, index=phnozeroindex[1])
    #     print(phnew)
    # for index in range(ph.shape[0]):
    #     print(index)
    #     phnozeroindex=torch.nonzero(ph[index])
    #     print(phnozeroindex)
        # phnew = torch.index_select(ph[index], dim=0, index=phnozeroindex)
        # print(phnew)

    # phnew = torch.index_select(ph,dim=0, index=phnozeroindex)
    # print(phnew)
    # return ph


        # a=torch.not_equal(mel2ph[index],0)
        # print(a)
    # out = torch.unique(mel2ph[index],return_counts=True)
    #     print(out)
    #     print(out[1])


    # out = torch.unique(mel2ph,return_counts=True,dim=1)
    # print(out)
    # print(out[1])

        # print(maxvalue)


    # for index in range(mel2ph.shape[0]):
    #     # maxvalue = mel2ph[index].max()
    #     # print(maxvalue.item())
    #     # numpyvalue=np.linspace(1,maxvalue.item(),num=maxvalue.item())
    # # print(numpyvalue)
    # # print(torch.from_numpy(numpyvalue))
    # #     yinsuzhi = torch.from_numpy(numpyvalue)
    #     ph = []
    #     for i in range(len(mel2ph[index])):
    #         if mel2ph[index][i] == mel2ph[index][i+1]
    #
    #         (mel2ph[index][i] == a).sum()
    #         print((mel2ph[index][i] == a).sum())
    #     #     ph.append(a)
    #     print(ph)
    #         print((mel2ph[index][i]==a).sum())

            # print((a==i).sum())
        # for location, yinsuvalue in enumerate(mel2ph[index].tolist()):
        #     if yinsuvalue != mel2ph[index].tolist()[location - 1]:
        #         jubulocation.append(location)
        # yinsulocation[index] = torch.tensor(jubulocation[:-1])

    # for list in mel2ph.tolist():
    #     for i in range(list.size):
    #         print(i)
        # print(list)
def compute_w(self, _feat_out, sample, wd=1, alpha=2):
    w_loss = F.l1_loss(_feat_out, sample["target"], reduction='none')
    bsz, seq_len, dim = _feat_out.size()
    durations = sample["durations"]
    _, src_l = durations.size()
    out_lens = durations.sum(dim=1)
    out = torch.zeros(bsz, src_l).to(_feat_out.device)
    for b in range(bsz):
        l = 0
        indices = []
        for t in range(src_l):
            step = durations[b, t]
            if step == 0:
                continue
            a = w_loss[b, l:l + durations[b, t], :]
            ph_ave = torch.mean(a)
            l = l + step
            out[b, t] = ph_ave

    mask = out == 0
    out2 = out.masked_fill(mask, float("-inf"))
    ph_w = F.softmax(out2 / wd, dim=1)
    # zhi shu fang da
    # ph_w = ph_w**alpha
    # ph_w = F.softmax(out2, dim=1)

    exp_ph_w, ll = self.expand(ph_w, durations)
    N = torch.numel(_feat_out)
    l1_loss = (w_loss * exp_ph_w.unsqueeze(-1)).sum() / N
    return l1_loss


def expand(self, x, durations):
    # x: B x T x C
    out_lens = durations.sum(dim=1)
    max_len = out_lens.max()
    bsz, seq_len = x.size()
    out = x.new_zeros((bsz, max_len))

    for b in range(bsz):
        indices = []
        for t in range(seq_len):
            indices.extend([t] * utils.item(durations[b, t]))
        indices = torch.tensor(indices, dtype=torch.long).to(x.device)
        out_len = utils.item(out_lens[b])
        out[b, :out_len] = x[b].index_select(0, indices)

    return out, out_lens
if __name__ == '__main__':
    # a=torch.tensor([[ 1, 1,1, 2,2,2,3,3,3, 4,4,0],
    #                 [ 1, 1,1, 2,2,2,3,3,3, 4,0,0]])
    a = torch.tensor([[1, 1, 1, 2, 2, 2, 4, 4, 5, 5, 0],
                      [1, 1, 1, 3, 3, 3, 3, 3, 3, 5, 5]])
    print(a.shape)
    frame2phdur(a)