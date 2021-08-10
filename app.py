import numpy as np
import pandas as pd
from collections import defaultdict

res=pd.read_csv('YOUR_FILE')
res.columns=["resid", "userid", "star",'comm','es']
res=res.drop(res.columns[3:], axis=1)

def find_common_res(user1,user2):
    s1 = set((res.loc[res["userid"]==user1,"resid"].values))
    s2 = set((res.loc[res["userid"]==user2,"resid"].values))
    return s1.intersection(s2)
    
def cosine_similarity(vec1, vec2):

    vec1 = np.mat(vec1)
    vec2 = np.mat(vec2)
    num = float(vec1 * vec2.T)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
   
def cal_user_similarity_with_movie_rating(user1,user2,res_id):
 
    u1 = res[res["userid"]==user1]
    u2 = res[res["userid"]==user2]
    vec1 = u1[u1.resid.isin([res_id])].sort_values(by="resid")["star"].values
    vec2 = u2[u2.resid.isin([res_id])].sort_values(by="resid")["star"].values
    #return vec1,vec2
    return cosine_similarity(vec1, vec2)




def recommend(user,num=10):
    #find similarity between user and other uesr
    user_similarity = [] 
    for other_user in res['userid'].unique():
        if other_user == user:
            continue
        print ("other user :",other_user)
        common_res = find_common_res(user,other_user)
        sim = cal_user_similarity_with_movie_rating(user,other_user,common_res)
        user_similarity.append([other_user,sim])
   

    user_similarity = np.array(user_similarity)
    sorted_index = np.argsort(user_similarity, axis=0)[:,1][::-1][:10]
    top10_similar_user = user_similarity[:,0][sorted_index]
    

    vis_res=res.loc[res["userid"]==user,"resid"].values
    not_vis_res = defaultdict(list) 
    for similar_user in top10_similar_user:
        _ress = res.loc[res.userid==similar_user,["resid","star"]].values.tolist()
        if isinstance(_ress[0], list):
            for _res in _ress:
                if _res[0] in vis_res:
                    continue
                not_vis_res[_res[0]].append(_res[1])
                
        elif _res[0] not in vis_res:
            not_vis_res[_res[0]].append(_res[1])
    

    for _res in not_vis_res:
        not_vis_res[_res] = np.mean(not_vis_res[_res])
    

    top10_res = sorted(not_vis_res.items(), key=lambda x: x[1], reverse=True)
    return[_res for _res,rating in top10_res][:num]
    
    print(recommend('USER_NAME',num=5))
