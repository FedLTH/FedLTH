import numpy as np
import math

from typing import List, Tuple, Union
from scipy import special



########################
# LOG-SPACE ARITHMETIC #
########################


def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.

    Args:
        logx: First term in log space.
        logy: Second term in log space.

    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.

    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.

    Returns:
        Difference of numbers in log space.

    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _compute_log_a_for_int_alpha(q: float, sigma: float, alpha: int) -> float:
    r"""Computes :math:`log(A_\alpha)` for integer ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """

    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.binom(alpha, i))
            + i * math.log(q)
            + (alpha - i) * math.log(1 - q)
        )

        s = log_coef_i + (i * i - i) / (2 * (sigma ** 2))
        log_a = _log_add(log_a, s)

    return float(log_a)


def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for any positive finite ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf
        for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in the paper mentioned above.
    """
    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)


def _log_erfc(x: float) -> float:
    r"""Computes :math:`log(erfc(x))` with high accuracy for large ``x``.

    Helper function used in computation of :math:`log(A_\alpha)`
    for a fractional alpha.

    Args:
        x: The input to the function

    Returns:
        :math:`log(erfc(x))`
    """
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)


def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    r"""Computes RDP of the Sampled Gaussian Mechanism at order ``alpha``.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        RDP at order ``alpha``; can be np.inf.
    """
    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    if q == 1.0:
        return alpha / (2 * sigma ** 2)

    if np.isinf(alpha):
        return np.inf

    return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(
    q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float]
) -> Union[List[float], float]:
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled Gaussian Mechanism (SGM) iterated ``steps`` times.

    Args:
        q: Sampling rate of SGM.
        noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added. Note that this is same as the standard
            deviation of the additive Gaussian noise when the L2-sensitivity
            of the function is 1.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.

    Returns:
        The RDP guarantees at all orders; can be ``np.inf``.
    """
    if isinstance(orders, float):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps


def get_privacy_spent(
    orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    return eps[idx_opt], orders_vec[idx_opt]


def compute_epsilon_alpha(noise_multiplier, num_steps, q, delta):

    alpha_list = np.arange(1.01, 100.0, 0.05)

    # noise_multiplier  = noise_multiplier*num_users*q

    temp_alpha, temp_epsilon = None, None
    # if num_processes <= 1:
    """
    def compute_rdp(q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float]
        ) -> Union[List[float], float]:
    """
    rdp_list = compute_rdp(q=q, noise_multiplier=noise_multiplier,
                            steps=num_steps, orders=alpha_list)

    """
    def get_privacy_spent(
    orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
    ) -> Tuple[float, float]:
    """


    temp_epsilon, temp_alpha = get_privacy_spent(orders=alpha_list, rdp=rdp_list, delta=delta)
    temp_rdp = rdp_list[list(alpha_list).index(temp_alpha)]

    return temp_epsilon, temp_alpha, temp_rdp


def MA(noise_multiplier, num_steps, q, delta):

  temp_epsilon = 2*q*np.sqrt(num_steps*np.log2(1/delta))/noise_multiplier


  return temp_epsilon






def cartesian_to_polar(x):
    r = np.linalg.norm(x)
    theta = np.arccos(x[0] / r)
    phi = [1. for i in range(len(x) - 1)]
    for i in range(len(phi)):
        phi[i] = np.arctan2(x[i + 1], x[0])
    return np.concatenate(([r, theta], phi))


def polar_to_cartesian(p):
    r = p[0]
    theta = p[1]
    phi = p[2:]
    x = [1. for i in range(len(phi) + 1)]
    x[0] = r * np.cos(theta)
    for i in range(len(phi)):
        x[i + 1] = x[0] * np.tan(phi[i])
    for j in range(len(x)):
        x[j] = round(x[j], 4)
    return x


def vector_to_matrix(vector, shape):
    shape = tuple(shape)
    if len(shape) == 0 or np.prod(shape) != len(vector):
        raise ValueError("Invalid input dimensions")
    matrix = np.zeros(shape)
    strides = [np.prod(shape[i + 1:]) for i in range(len(shape) - 1)] + [1]
    for i in range(len(vector)):
        index = [0] * len(shape)
        for j in range(len(shape)):
            index[j] = (i // strides[j]) % shape[j]
        matrix[tuple(index)] = vector[i]
    return matrix



def cartesian_add_noise(p, sigma1, C1, sigma2):

    r = p[0]
    r += C1 * sigma1 * np.random.normal(0, 1)

    theta = p[1:]
    theta += 2 * math.pi * sigma2 * np.random.normal(0, 1)

    return np.concatenate(([r], theta))


def devide_epslion(sigma, q, n):
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    eps, opt_order = apply_dp_sgd_analysis(q, sigma, 1, orders, 10 ** (-5))

    eps_sum = n * eps

    eps1 = eps_sum * 0.000001

    eps2 = eps_sum - eps1

    sigma1 = get_noise_multiplier(target_epsilon=eps1, target_delta=1e-5, sample_rate=512 / 60000, steps=1,
                                  alphas=orders)
    sigma2 = get_noise_multiplier(target_epsilon=eps2, target_delta=1e-5, sample_rate=512 / 60000, steps=1,
                                  alphas=orders)
    return sigma1, sigma2


def get_noise_multiplier(
    noise_multiplier_list,
    steps_list,
    conf,
    n,
    rounds,
    target_epsilon,
    epsilon_tolerance: float = 0.1,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        steps: number of steps to run
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
        # 确保 noise_multiplier_list 的长度等于 num_steps
    assert len(noise_multiplier_list) == len(steps_list), "噪声乘数列表的长度必须等于步骤数"

    # 二分查找noise_multiplier
    alphas = np.arange(1.01, 100.0, 0.05)
    sigma_low, sigma_high = 0, 5
    delta = conf['delta']
    batch_size=conf['batch_size']
    epochs=conf['local_epoch']
    q=batch_size/n
    num_steps=int(epochs*rounds*conf['k']/q)
    steps_list.append(num_steps)
    noise_multiplier_list.append(sigma_high)
    eps_high, best_alpha = compute_eps_noise_list(noise_multiplier_list, steps_list, q, delta)


    if eps_high > target_epsilon:
        raise ValueError("The target privacy budget is too low. 当前可供搜索的最大的sigma只到10")
    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        noise_multiplier_list[-1]=sigma
        eps, best_alpha = compute_eps_noise_list(noise_multiplier_list, steps_list, q, delta)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma




    return round(sigma_high, 2)



def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    """Compute and print results of DP-SGD analysis."""

    # compute_rdp requires that sigma be the ratio of the standard deviation of
    # the Gaussian noise to the l2-sensitivity of the function to which it is
    # added. Hence, sigma here corresponds to the `noise_multiplier` parameter   sigma=noise_multiplier
    # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
    rdp = compute_rdp(q, sigma, steps, orders)

    eps, opt_order = compute_eps(orders, rdp, delta)

    return eps, opt_order




def compute_eps(orders, rdp, delta):
    """Compute epsilon given a list of RDP values and target delta.
    Args:
      orders: An array (or a scalar) of orders.
      rdp: A list (or a scalar) of RDP guarantees.
      delta: The target delta.
    Returns:
      Pair of (eps, optimal_order).
    Raises:
      ValueError: If input is malformed.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if delta <= 0:
        raise ValueError("Privacy failure probability bound delta must be >0.")
    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")

    eps_vec = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1:
            raise ValueError("Renyi divergence order must be >=1.")
        if r < 0:
            raise ValueError("Renyi divergence must be >=0.")

        if delta ** 2 + math.expm1(-r) >= 0:  # delta的约束条件
            # In this case, we can simply bound via KL divergence:
            # delta <= sqrt(1-exp(-KL)).
            eps = 0  # No need to try further computation if we have eps = 0.
        elif a > 1.01:
            # This bound is not numerically stable as alpha->1.Thus we have a min value of alpha.
            eps = (r - (np.log(delta) + np.log(a)) / (a - 1) + np.log((a - 1) / a))
        else:
            # In this case we can't do anything. E.g., asking for delta = 0.
            eps = np.inf
        eps_vec.append(eps)

    idx_opt = np.argmin(eps_vec)
    return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def compute_eps_noise_list(noise_multiplier_list, steps_list, q, delta):
    """
    计算总的隐私预算（epsilon）对于不同的噪声乘数。

    参数：
    noise_multiplier_list (list): 每个步骤的噪声乘数列表。
    num_steps (int): 步骤数列表。
    q (float): 采样率。
    delta (float): 隐私参数 delta。

    返回：
    float: 总的隐私消耗 epsilon。
    """
    # 确保 noise_multiplier_list 的长度等于 num_steps
    assert len(noise_multiplier_list) == len(steps_list), "噪声乘数列表的长度必须等于步骤数"

    # 初始化每个步骤的 RDP 记录
    rdp_list = []

    # 对每个步骤单独计算 RDP
    for i in range(len(steps_list)):
        noise_multiplier = noise_multiplier_list[i]
        orders = np.arange(1.01, 100.0, 0.05)
        # orders = (list(range(2, 64)) + [128, 256, 512])
        # 计算rdp
        rdp = compute_rdp(q=q, noise_multiplier=noise_multiplier, steps=steps_list[i], orders=orders)
        rdp_list.append(rdp)

    # 将所有步骤的 RDP 进行累积
    accumulated_rdp = np.sum(rdp_list, axis=0)

    # 计算总的 epsilon 和 alpha
    epsilon, alpha = get_privacy_spent(orders=orders, rdp=accumulated_rdp, delta=delta)

    return epsilon, alpha



if __name__ == '__main__':
    #客户端级别差分隐私
    import conf
    conf=conf.conf
    noise_multiplier, delta = 0.8, 1e-5
    batch_size = conf['batch_size']
    g_round = 10

    n_equiv = batch_size / conf['k']

    noise = get_noise_multiplier([noise_multiplier], [0], conf, 60000, g_round, target_epsilon=10)
    print(noise)

    # 交叉验证：用返回的noise，按真正的client-level语义重新算一次eps，看是否≈10
    eps_final, _ = compute_eps_noise_list([noise], [g_round], conf['k'], conf['delta'])
    print(f'验证: noise={noise} 对应 eps={eps_final}, 目标=10')

    # 样本级别差分隐私
    # conf=conf.conf
    # noise_multiplier, delta = 0.7,  1e-5
    # n=600 #客户端训练集数量

    # batch_size=32
    # epochs=1
    # g_round=100
    # q=batch_size/n
    # num_steps=int(epochs*g_round*0.2/q)
    # noise=get_noise_multiplier([noise_multiplier],[0],conf,n,g_round,target_epsilon=10)
    # print(noise)
# if __name__=='__main__':
#     # group : p_round,total_round
#     params=[[14,105],[5,92],[6,112],[9,97]]
#     from conf import conf
#     n=400
#     q=conf['batch_size']/n
#     init_noise=get_noise_multiplier([1],[0],conf,n,500,target_epsilon=5)
#     print(f'init noise:{init_noise}')
    # for group_id in range(len(params)):
    #     train_round=params[group_id][0]-5
    #     train_steps=int(conf['local_epoch']*train_round/q)
    #     for i in range(train_round):
    #         num_steps=int(conf['local_epoch']*i/q)+1
    #         eps,_=compute_eps_noise_list([init_noise],[num_steps],q,0.001)
    #         print(f'group:{group_id};eps_used:{eps}; ')

    #     p_steps=int(conf['local_epoch']*5/q)
    #     p_noise=get_noise_multiplier([init_noise],[train_steps],conf,n,25,target_epsilon=1.3)
    #     print(f'prune_noise:{p_noise}')
    #     for i in range(5):
    #         num_steps=int(conf['local_epoch']*i/q)+1
    #         eps,_=compute_eps_noise_list([init_noise,p_noise],[train_steps,num_steps],q,delta=0.001)
    #         print(f'group:{group_id};eps_used:{eps}; ')

    #     remain_rounds=params[group_id][1]-params[group_id][0]
    #     remain_steps=int(conf['local_epoch']*remain_rounds/q)
    #     remain_noise=get_noise_multiplier([init_noise,p_noise],[train_steps,p_steps],conf,n,500-35,target_epsilon=5)
    #     print(f'next_noise:{remain_noise}')
    #     for i in range(remain_rounds):
    #         num_steps=int(conf['local_epoch']*i/q)+1
    #         eps,_=compute_eps_noise_list([init_noise,p_noise,remain_noise],[train_steps,p_steps,num_steps],q,delta=0.001)
    #         print(f'group:{group_id};eps_used:{eps}; ')
    #     print('finish')
    # eps,_=compute_eps_noise_list([2.08],[num_steps],q,delta=0.001)
    # noise_multiplier=get_noise_multiplier([0.2],[0],conf,n,500,target_epsilon=10)
    # print(eps)
    # for i in range(100):
    #     num_steps=int(conf['local_epoch']*i/q)+1
    #     print(compute_eps_noise_list([init_noise],[num_steps],q,0.001))


    # 1.48,1.56,1.72,1.88


    # noise_multiplier=get_noise_multiplier([2.08,p_noise],[train_steps,p_steps],conf,n,500-35,target_epsilon=5)
    # # print(noise_multiplier)
    # print(f'next noise:{noise_multiplier}')
    # for i in range():
    #     new_num_steps=int(conf['local_epoch']*i/q)+1
    #     eps,_=compute_eps_noise_list([2.08,p_noise,noise_multiplier],[train_steps,p_steps,new_num_steps],q,0.001)
    #     print(eps)

    # for i in range(100):
    #     new_num_steps=int(conf['local_epoch']*i/q)+1
    #     print(f'eps_used:{compute_eps_noise_list([2.08],[new_num_steps],q,conf["delta"])}; ')
