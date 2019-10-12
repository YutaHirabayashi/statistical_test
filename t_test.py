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

def exp_variance(m, n, m_mu, n_mu, sigma):
    x_m = create_normal_sample(n=m, mu=m_mu, sigma=sigma)
    x_n = create_normal_sample(n=n, mu=n_mu, sigma=sigma)

    #それぞれの不偏分散を計算
    v_m = unbiased_variance(x_m)
    v_n = unbiased_variance(x_n)

    #それぞれのサンプル平均の差を計算
    e = np.average(x_m) - np.average(x_n)


    #プールされた分散を計算
    v_p = ((x_m.shape[0]-1)*v_m + (x_n.shape[0]-1)*v_n)/(x_m.shape[0]+x_n.shape[0]-2)

    #普通に不偏分散を計算
    #x_s = np.concatenate([x_m, x_n])
    #v_s = unbiased_variance(x_s)
    return [e, v_p]

def check_prob(exp_num, m_mu, n_mu, sigma, m_size, n_size):

    a = [exp_variance(m_size, n_size, m_mu, n_mu, sigma) for i in range(0, exp_num)]
    dif_mu = np.array([v[0] for v in a])
    dif_var = np.array([v[1] for v in a])

    #t検定量の計算
    t_val = dif_mu / np.sqrt(dif_var) * np.sqrt((m_size*n_size)) / np.sqrt(m_size+n_size)
    
    #有意確率の検証（t検定は近似を一切使ってないので厳密に一致）
    significant = np.where(
        (t_val <= scipy.stats.t.ppf(0.025, m_size+n_size-2)) | (t_val >= scipy.stats.t.ppf(0.975, m_size+n_size-2)), 
        1,0
    )
    prob = np.average(significant)
    return prob



def main():
    #危険率の計算
    size_list = [10, 100, 1000, 10000]
    alpha = [check_prob(exp_num=10000, m_mu=5, n_mu=5, sigma=1, m_size=size, n_size=size) for size in size_list]

    #効果量を固定(0.1)した時のサンプルサイズと検出力の関係
    size_list = [10, 100, 1000, 10000]
    beta = [check_prob(exp_num=10000, m_mu=5, n_mu=5.1, sigma=1, m_size=size, n_size=size) for size in size_list]
    
    plt.figure()
    sns.pointplot(x=size_list, y=alpha, label="alpha", color='blue')
    sns.pointplot(x=size_list, y=beta, label="beta", color='red')
    plt.savefig("サンプルサイズと危険率_検出力.png")
    plt.close()
    return

if __name__ == "__main__":
    main()
    pass