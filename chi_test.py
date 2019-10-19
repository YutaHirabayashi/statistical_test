#分割表
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def observe(N):

    #タバコを吸ってる人の割合
    p_tabaco = 0.3

    #がんの人の割合
    p_cancer = 0.1

    #タバコを吸ってることとがんであることは独立である場合
    p_tabaco_cancer = p_tabaco*p_cancer
    p_no_tabaco_cancer = (1-p_tabaco)*p_cancer
    p_tabaco_no_cancer = p_tabaco*(1-p_cancer)
    p_no_tabaco_no_cancer = (1-p_tabaco)*(1-p_cancer)

    #上記の4項分布にしたがって、観測値を得る
    res = np.random.multinomial(N, [p_tabaco_cancer, p_no_tabaco_cancer, p_tabaco_no_cancer, p_no_tabaco_no_cancer], 1)
    dict_res = {}
    dict_res["tabaco_cancer"] = res[0][0]
    dict_res["tabaco_no_cancer"] = res[0][1]
    dict_res["no_tabaco_cancer"] = res[0][2]
    dict_res["no_tabaco_no_cancer"] = res[0][3]
    return dict_res

def do_chi_test(dict_res):
    N = dict_res["tabaco_cancer"] + dict_res["tabaco_no_cancer"] + dict_res["no_tabaco_cancer"] + dict_res["no_tabaco_no_cancer"]

    #周辺度数の計算
    e_tabaco = dict_res["tabaco_cancer"] + dict_res["tabaco_no_cancer"]
    e_cancer = dict_res["tabaco_cancer"] + dict_res["no_tabaco_cancer"]
    
    #期待度数を計算
    e_tabaco_cancer = (e_tabaco)*(e_cancer)/N
    e_tabaco_no_cancer = (e_tabaco)*(N-e_cancer)/N
    e_no_tabaco_cancer = (N-e_tabaco)*(e_cancer)/N
    e_no_tabaco_no_cancer = (N-e_tabaco)*(N-e_cancer)/N

    #カイ二乗統計量を計算
    chi = ((dict_res["tabaco_cancer"] - e_tabaco_cancer)**2)/e_tabaco_cancer +\
        + ((dict_res["tabaco_no_cancer"] - e_tabaco_no_cancer)**2)/e_tabaco_no_cancer+\
        + ((dict_res["no_tabaco_cancer"] - e_no_tabaco_cancer)**2)/e_no_tabaco_cancer+\
        + ((dict_res["no_tabaco_no_cancer"] - e_no_tabaco_no_cancer)**2)/e_no_tabaco_no_cancer
    
    #カイ二乗検定を実施
    a = stats.chi2.ppf(0.95,1)
    sig = 1 if chi > a else 0
    return sig

def do_fisher_test(dict_res):
    _, p_value = stats.fisher_exact(
        [[dict_res["tabaco_cancer"], dict_res["tabaco_no_cancer"]],[dict_res["no_tabaco_cancer"], dict_res["no_tabaco_no_cancer"]]]
    )
    sig = 1 if p_value < 0.05 else 0
    return sig

def main():

    N_list = [10, 50, 100, 500] #実験ごとのサンプルサイズ
    exp_num = 10000 #各サンプルサイズでの実験回数
    
    chi_alpha_list = []
    fisher_alpha_list = []

    for N in N_list:
        print("*****N = {} : start*****".format(N))
        
        dict_res_list = [observe(N = N) for _ in range(exp_num)]

        chi_sig_list = [do_chi_test(dict_res) for dict_res in dict_res_list]
        fisher_sig_list = [do_fisher_test(dict_res) for dict_res in dict_res_list]

        chi_alpha_list.append(np.average(np.array(chi_sig_list)))
        fisher_alpha_list.append(np.average(np.array(fisher_sig_list)))

    dict_alpha_every_test = {}
    dict_alpha_every_test = {"chi_test":chi_alpha_list, "fisher_test":fisher_alpha_list}
    
    df_list = []
    for test_name in dict_alpha_every_test:
        df = pd.DataFrame()
        df["N"] = N_list
        df["alpha"] = dict_alpha_every_test[test_name]
        df["kind"] = test_name
        df_list.append(df)
    df = pd.concat(df_list, axis = 0)

    plt.figure()
    sns.pointplot(x = "N", y = "alpha", data = df, hue = "kind")
    plt.savefig("chi_vs_fisher.png")
    plt.show()
    
    

    return



if __name__ == "__main__":
    main()

    pass