import sqlalchemy
import sqlite3
import os

def get_connection():

    uri = os.getcwd()+'/app/reco.db'
    creator = lambda: sqlite3.connect(uri, uri=True)
    db = sqlalchemy.create_engine('sqlite:////', creator=creator)
    conn = db.raw_connection()
    
    return conn

# 최근에 조회한 순서대로 조회한 item 가져오기
def get_viewed_list(user_id: int):
    conn = get_connection()

    sql = '''
        select item_title
        from user_history
        where user_id = {}
        order by timestamp desc
    '''.format(user_id)

    cursor = conn.cursor()
    cursor.execute(sql)

    row = cursor.fetchall()

    view_list = []

    for obj in row :
        data_dic = {
            'item_title' : obj[0]
        }
        view_list.append(data_dic)
    
    conn.close

    return view_list

# Popular item 가져오기
def get_popular_list():
    conn = get_connection()

    sql = '''
        select item_title
        from popular_items
    '''

    cursor = conn.cursor()
    cursor.execute(sql)

    row = cursor.fetchall()

    pop_list = []

    for obj in row :
        data_dic = {
            'item_title' : obj[0]
        }
        pop_list.append(data_dic)
    
    conn.close

    return pop_list



# sims reco set
def get_reco_list(user_id: int,
                 reco_db: str):
    
    conn = get_connection()

    sql = '''
        select item_title
        from {}
        where user_id = {}
        order by priority 
    '''.format(reco_db, user_id)

    cursor = conn.cursor()
    cursor.execute(sql)

    row = cursor.fetchall()

    reco_list = []

    for obj in row :
        data_dic = {
            'item_title' : obj[0]
        }
        reco_list.append(data_dic)
    
    conn.close

    return reco_list

def get_compare_list(user_id: int,
                    reco_db_1: str, 
                    reco_db_2: str):

    reco_1=get_reco_list(user_id=user_id,
                        reco_db=reco_db_1)
    
    reco_2=get_reco_list(user_id=user_id,
                        reco_db=reco_db_2)
    
    reco_items_1=[list(reco_1[i].values())[0] for i in range(len(reco_1))]
    reco_items_2=[list(reco_2[i].values())[0] for i in range(len(reco_2))]
    
    
    ind_dict_reco_1 = dict((k,i) for i,k in enumerate(reco_items_1))
    ind_dict_reco_2 = dict((k,i) for i,k in enumerate(reco_items_2))

    inter_reco_1 = set(ind_dict_reco_1).intersection(reco_items_2)
    inter_reco_2 = set(ind_dict_reco_2).intersection(reco_items_1)

    indices_reco_1= [ind_dict_reco_1[x] for x in inter_reco_1]
    indices_reco_2= [ind_dict_reco_2[x] for x in inter_reco_2]
    
    
    for i in range(len(reco_1)):
        if i in indices_reco_1:
            reco_1[i]["is_intersection"]=1
        else:
            reco_1[i]["is_intersection"]=0
        reco_1[i]["reco_db"]=reco_db_1
        
    for i in range(len(reco_2)):
        if i in indices_reco_2:
            reco_2[i]["is_intersection"]=1
        else:
            reco_2[i]["is_intersection"]=0
        reco_2[i]["reco_db"]=reco_db_2
        
        
    return reco_1, reco_2