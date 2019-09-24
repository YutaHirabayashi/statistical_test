import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_normal_sample(n, mu, sigma):
    x = np.random.normal(mu, sigma, n)
    return x

def unbiased_variance(x_i):
    avg = np.average(x_i)
    v = np.sum((avg-x_i)**2)/(x_i.shape[0]-1)
    return v

def exp_variance(m, n):
    sigma = 1
    x_m = create_normal_sample(n=m, mu=5, sigma=sigma)
    x_n = create_normal_sample(n=n, mu=5, sigma=sigma)

    #それぞれの不偏分散を計算
    v_m = unbiased_variance(x_m)
    v_n = unbiased_variance(x_n)

    #プールされた分散を計算
    v_p = ((x_m.shape[0]-1)*v_m + (x_n.shape[0]-1)*v_n)/(x_m.shape[0]+x_n.shape[0]-2)

    #普通に不偏分散を計算
    x_s = np.concatenate([x_m, x_n])
    v_s = unbiased_variance(x_s)
    return [v_p, v_s]

def main():
    exp_num = 10000
    m = 5
    n = 5

    v_list = [exp_variance(m, n) for i in range(0, exp_num)]
    v_p_list = [v[0] for v in v_list]
    v_s_list = [v[1] for v in v_list]

    plt.figure()
    sns.distplot(v_p_list) ##自由度m+n-2->8のカイ二乗分布/3 -> 分散の理論値は0.25
    sns.distplot(v_s_list) ##自由度m+n-1->9のカイ二乗分布/2 -> 分散の理論値は0.22
    plt.savefig("test.png")

    vp_v = np.var(v_p_list)
    vs_v = np.var(v_s_list)

    ##　参考：https://blog.rinotc.com/entry/2017/07/30/標本(不偏)分散の期待値,_分散%5B正規分布%5D

    return

if __name__ == "__main__":
    main()
    pass