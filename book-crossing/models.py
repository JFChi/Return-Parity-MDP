import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainRNNModel(nn.Module):
    def __init__(self, action_embeddings, reward_dim,
        rnn_input_dim, rnn_output_dim,
        statistic_dim, item_num,
        device,
    ):
        super(PretrainRNNModel, self).__init__()
        

        self.statistic_dim = statistic_dim
        self.item_num = item_num
        self.reward_dim = reward_dim
        self.rnn_input_dim = rnn_input_dim
        self.rnn_output_dim = rnn_output_dim
        self.device = device

        # see https://www.oreilly.com/library/view/deep-learning-with/9781788624336/fa953b7d-daac-4c4e-be74-66a7e94de98e.xhtml
        # for fix embedding when training
        # self.action_embeddings.require_grad=False
        self.action_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(action_embeddings))
        self.action_embeddings.requires_grad = False
        self.rnn = nn.LSTMCell(self.rnn_input_dim, self.rnn_output_dim)
        self.linear = nn.Linear(self.rnn_output_dim + self.statistic_dim, self.item_num)

    def forward(self, input_states, rnn_initial_states):

        # embed actions and reward
        pre_actions, pre_rewards, pre_statistic = input_states
        # pre_actions.shape = (None, self.batch_size)
        # pre_rewards.shape = (None, self.batch_size)
        # pre_statistic.shape = (None, self.batch_size, user_stats_dim)

        pre_action_embeds = self.action_embeddings(pre_actions)
        # pre_action_embeds.shape = (None, self.batch_size, action_dim)
        one_hot_rewards = F.one_hot(
            torch.floor(self.reward_dim * (2.0-pre_rewards) / 4.0).to(torch.int64),
            num_classes=self.reward_dim,
        )
        # one_hot_rewards.shape = (None, self.batch_size, reward_dim)

        pre_ars = torch.cat((pre_action_embeds, one_hot_rewards, pre_statistic), dim=-1)
        
        # input to rnn
        rnn_outputs = []
        hx, cx = rnn_initial_states
        for i in range(pre_ars.shape[0]): # iterate over
            hx, cx = self.rnn(pre_ars[i], (hx, cx))
            # print(pre_ars[i].shape)
            rnn_outputs.append(hx)

        pn_outputs = []
        for i in range(len(rnn_outputs)):
            linear_input = torch.cat((rnn_outputs[i], pre_statistic[i]), dim=-1)
            linear_output = self.linear(linear_input)
            pn_outputs.append(linear_output)

        # add l2 regularization on linear layers
        l2_norm = torch.tensor(0.0).to(self.device)
        for name, param in self.named_parameters():
            if name.startswith('linear'):
                l2_norm += torch.norm(param)
        return pn_outputs, l2_norm

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, units=[256, 256]):
        super().__init__()

        self.nets = nn.ModuleList([
            nn.Linear(input_dim, units[0]),
            *[nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(units[i], units[i+1])
            ) for i in range(len(units) - 1)],
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(units[-1], output_dim)
            )
        ])

    def forward(self, x):
        for net in self.nets:
            x = net(x)
        return x

# MMD unbiasd distance
# code adapted from 
# https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/statistics_diff.py
class MMDStatistic():
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.."""

    def __init__(self, alphas, kernel_name="gaussian"):
        self.alphas = alphas
        self.kernel_name = kernel_name
        assert kernel_name in ["gaussian", "laplacian"]

    def __call__(self, sample_1, sample_2, ret_matrix=False):
        r"""
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""

        self.n_1 = sample_1.shape[0]
        self.n_2 = sample_2.shape[0]

        # The three constants used in the test.
        self.a00 = 1. / (self.n_1 * (self.n_1 - 1))
        self.a11 = 1. / (self.n_2 * (self.n_2 - 1))
        self.a01 = - 1. / (self.n_1 * self.n_2)


        sample_12 = torch.cat((sample_1, sample_2), 0)
        if self.kernel_name == "gaussian":
            distances = pdist(sample_12, sample_12, norm=2)
        elif self.kernel_name == "laplacian":
            distances = pdist(sample_12, sample_12, norm=1)
        else:
            raise NotImplementedError

        kernels = None
        for alpha in self.alphas:
            # For single kernel
            if self.kernel_name == "gaussian":
                kernels_a = torch.exp(- alpha * distances ** 2)
            elif self.kernel_name == "laplacian":
                kernels_a = torch.exp(- alpha * distances)
            else:
                raise NotImplementedError
            # For multiple kernel, append kernel
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd

def pdist(sample_1, sample_2, norm=2, eps=1e-9):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

if __name__ == "__main__":
    # test MMD
    # x = torch.FloatTensor([[3, 3], [4, 3], [5, 3]])
    x = torch.FloatTensor([[1, 3], [2, 3], [5, 3]])
    y = torch.FloatTensor([[1, 3], [2, 3], [5, 3]])
    #### for guassian kernel ####
    print("test guassian kernels")
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 1, 5, 10] # coeiffient of rbf kernel 
    print(x)
    print(y)
    print(x.shape, y.shape)
    n1, n2 = x.shape[0], y.shape[0]
    print("n1, n2", n1, n2)
    mmd_dist = MMDStatistic(alphas, kernel_name="gaussian") # MMDStatistic(alphas, kernel_name="guassian")
    mmd, dist_matrix  = mmd_dist(x, y, ret_matrix=True)
    print("mmd", mmd)