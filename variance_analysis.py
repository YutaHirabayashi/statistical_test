import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 共分散分析

def observe_1d_table(sample_size, sigma, alpha_list = [0, 0]):
    
    #beta = 3
    beta = 0

    list_df = []
    for series, alpha in enumerate(alpha_list):
        series_name = "series- {:0=2}".format(series)
        mulcol = pd.MultiIndex.from_arrays(
            [[series_name, series_name], ["x_data", "y_data"]]
        )
        x_data = np.random.rand(sample_size)*20
        y_data = np.random.normal(loc=x_data*beta+alpha, scale=sigma)
        
        _df = pd.DataFrame(np.array([x_data, y_data]).T, columns=mulcol) 
        list_df.append(_df)
    
    df = pd.concat(list_df, axis = 1)
    return df

def get_S_for_anova(df):
    #y_dataだけ抽出
    idx = pd.IndexSlice
    data = df.loc[:, idx[:, "y_data"]].values
    mean_array = np.average(data, axis = 0)
    mean = np.average(mean_array)

    S_all = np.sum((data - mean)**2)#全変動
    #残差
    S_e = np.sum((data - mean_array)**2) ##平均からの変動
    
    S_avg = data.shape[0] * np.sum((mean_array - mean)**2) ##平均の変動
    return S_all, S_e, S_avg

def plot_S_and_chi(fig_name, S, sigma, degree, bins = 30):
    plt.figure()
    plt.hist(S/(sigma**2), density=True, bins = bins)
    x_arr = np.arange(0, (S/(sigma**2)).max(), 0.1)
    chi_arr = stats.chi2.pdf(x_arr, degree)
    plt.plot(x_arr, chi_arr, label = "degree : {}".format(degree))
    plt.legend()
    plt.savefig(fig_name)
    plt.close()
    return

def plot_F(fig_name, F, dfn, dfd, bins = 30):
    plt.figure()
    plt.hist(F, density = True, bins = bins)
    x_arr = np.arange(0, F.max(), 0.1)
    F_arr = stats.f.pdf(x_arr, dfn, dfd)
    plt.plot(x_arr, F_arr, label = "dfn:{} dfd:{}".format(dfn, dfd))
    plt.legend()
    plt.savefig(fig_name)
    plt.close()
    return

def do_F_test(F_list, dfn, dfd):
    a = stats.f.ppf(0.95, dfn, dfd)
    sig_list = [1 if F > a else 0 for F in F_list]
    return np.average(np.array(sig_list))

def exp_F_test(exp_num, sample_size, alpha_list, sigma):
    series_num = len(alpha_list)
    df_list = [observe_1d_table(sample_size = sample_size, sigma = sigma, alpha_list=alpha_list) for _ in range(exp_num)]
    s = [get_S_for_anova(df) for df in df_list]

    S_all = np.array([a[0] for a in s])
    S_e = np.array([a[1] for a in s])
    S_avg = np.array([a[2] for a in s])

    #plot_S_and_chi("S_all.png", S_all, sigma, series_num*sample_size-1)
    #plot_S_and_chi("S_e.png", S_e, sigma, series_num*(sample_size-1))
    #plot_S_and_chi("S_avg.png", S_avg, sigma, series_num-1)

    #F検定量1
    F_1 = (S_avg/(series_num-1)) / (S_e/(series_num*(sample_size-1)))
    #plot_F("F.png", F, series_num-1, series_num*(sample_size-1))
    p_1 = do_F_test(F_1, series_num-1, series_num*(sample_size-1))

    F_2 = (S_avg/(series_num-1)/(S_all/(series_num*sample_size-1)))
    p_2 = do_F_test(F_2, series_num-1, series_num*sample_size-1)

    return p_1, p_2

def main():

    exp_num = 1000
    sample_size = 50
    sigma = 5

    #p_1, p_2 = exp_F_test(exp_num, sample_size, [0, 0], sigma)

    #F統計量、どっちを使ったほうが精度（検出力）が良いかチェック
    delta_list = [0.5, 1, 2, 4]
    beta_table = [exp_F_test(exp_num, sample_size, [0, delta], sigma) for delta in delta_list]

    df_list=[]
    for i, hue in enumerate(["F-1", "F-2"]):
        _df = pd.DataFrame([b[i] for b in beta_table], columns = ["beta"])
        _df["delta"] = delta_list
        _df["hue"] = hue
        df_list.append(_df)
    df = pd.concat(df_list, axis = 0)

    plt.figure()
    sns.pointplot(x = "delta", y = "beta", data = df, hue = "hue")
    plt.savefig("compare_F_how.png")
    plt.show()

    return

if __name__ == "__main__":
    main()

    pass