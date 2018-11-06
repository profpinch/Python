"""
A Markov model is a stochastic approach used to describe randomly changing systems (credit ratings) where it
is assumed that future states depend only on the current state and not on events occuring before.

"""
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg as linalg
from itertools import product
from itertools import chain

def convert_data(df, string_date = 'period_ending_dt', string_rating ='system_risk_grade', string_ID = 'ID', time_conv = 365, assumption=False):
    '''
    Converts simple dataframe to what is needed for transition matrix (generator matrix calc)
    
    http://past.rinfinance.com/agenda/2015/talk/AlexanderMcNeil.pdf
    '''

    #want to have id, startime, endtime, startrating, endrating and time in a brand new dataframe
    #group by unique id
    unique_group = df[string_ID].unique()
    Data=[]
    for un in unique_group:
        data = df[df[string_ID]==un]
        data_ = data.reset_index(drop=True)
        Data.append(data_)
        
    #now create a startrating and an endrating: assumption on the first one: start = end rating
    complete_data = []
    for j in range(len(Data)):
        start_rate = []
        end_rate = []
        start_date = []
        end_date = []
        for i in range(len(Data[j])):
            if i == 0:
                start_rate.append(Data[j][string_rating][i])
                end_rate.append(Data[j][string_rating][i])
                start_date.append(Data[j][string_date][i])
                end_date.append(Data[j][string_date][i])
            else:
                start_rate.append(Data[j][string_rating][i-1])
                end_rate.append(Data[j][string_rating][i])
                start_date.append(Data[j][string_date][i-1])
                end_date.append(Data[j][string_date][i])
        
        Data[j]["start_rate"] = start_rate
        Data[j]["end_rate"] = end_rate
        Data[j]["start_time"] = pd.to_datetime(start_date)
        Data[j]["end_time"] = pd.to_datetime(end_date)
        Data[j]['time'] = ((Data[j]["end_time"] - Data[j]["start_time"]).dt.days)/time_conv #annual/monthly whatever the user inputs
        
        if assumption == False:
            pass
        if assumption == True:
            if len(Data[j])>0:
                Data[j]['time'] = Data[j]['time'].replace(to_replace=0, method='bfill') #is user wants assumption fill the zero (first observation) with an assumption of forward period time.
                
        complete_data.append(Data[j])
    
    #now stack the data on top of each other: make it one full dataset
    df_final = pd.concat(complete_data)

    return df_final


def Transition_Matrix_Markov(df_final,n=1):
    #Replicate the R-code in Python
    Njktable = pd.crosstab(index=df_final['start_rate'], columns = df_final['end_rate']) #VERIFIED to the Njktable
    
    RiskSet = df_final.groupby(['start_rate'])['time'].sum() #this is verified with the R function
    Jlevels = df_final['start_rate'].unique().tolist() #verified to R: levels(df$start_rate)
    Klevels = df_final['end_rate'].unique().tolist() # verified to R: levels(df$end_rate)
    
    #now create the Njmatrix in R
    Njmatrix = pd.DataFrame(index = Jlevels, columns=Klevels)
    
    i = 0
    for risk in RiskSet:
        Njmatrix.iloc[i] = risk #the Njmatrix is now verified relative to the R-function
        i = i+1
        
       
    #next create the Lambda.hat <- Njktable/Njmatrix
    Lambda_hat = Njktable/Njmatrix #this is verified!
    
    '''
    R-code:
    D <- rep(0,dim(Lambda.hat)[2]) just zero rows
    Lambda.hat <- rbind(Lambda.hat,D) add it to the lambda matrix
    diag(Lambda.hat) <- D make the diagonals equal to the D (zeros)
    rowsums <- apply(Lambda.hat,1,sum) #calculate rowsums
    diag(Lambda.hat) <- -rowsums #insert diagonals as the negative rowsums
    
    '''
    #complete matrix by adding a default row
    D = np.zeros(len(Klevels))
    Lambda_hat2 = Lambda_hat
    Lambda_hat2.loc[len(Lambda_hat)] = D
    Lambda_hat2 = Lambda_hat2.rename({(len(Lambda_hat)-1):'{0}'.format(Klevels[-1])})
    
    #next make sure the diagonal is set to zero (or "D")
    def pd_fill_diagonal(df_matrix, value=0): 
        mat = df_matrix.values
        n = mat.shape[0]
        mat[range(n), range(n)] = value
        return pd.DataFrame(mat,index=df_matrix.index, columns=df_matrix.columns)
    
    Lambda_hat3 = pd_fill_diagonal(Lambda_hat2) #this is verified with the R-function
    
    '''
    now calculate the row_sums
    rowsums <- apply(Lambda.hat,1,sum)    
    '''
    #Row sums
    row_sums = -1*(Lambda_hat3.sum(axis=1).values)
    #final make the negative rowsums the diagonal instead of zeros
    Lambda_hat4 = pd_fill_diagonal(Lambda_hat3, value = row_sums) #verified as the last lambda hat before estimating the transition matrix
    Lambda_hat4 = Lambda_hat4.apply(pd.to_numeric) #important to make expm work
    
    #now create the estimated transition probabilities
    Matrix = np.matrix(Lambda_hat4.as_matrix())
    Prob_matrix = linalg.expm(Matrix) #Verified!!
    
    #this one will make it an annual, 2-year, 3-year, 5-year etc use n=1 in function
    Prob_matrix_final = np.linalg.matrix_power(Prob_matrix,n)
    
    #now make visual improvements on the Prob_matrix
    Prob_matrix_df = pd.DataFrame(Prob_matrix_final,index=Klevels,columns=Klevels)
    Prob_matrix_df[Prob_matrix_df==0.] = 0. #just fix the -0 to 0
    Prob_matrix_df = np.round((Prob_matrix_df*100),2).astype(str) + "%"
    
    return Prob_matrix_df, Prob_matrix_final
    
'''
Historical Transition Matrix Calculation
'''
def cond_probs_np(sequences, prob = True):
    """Calculate 1-step transitional probabilities from sequences.
    Return dictionary mapping transitions to probabilities.
    """
    distinct = set(chain.from_iterable(sequences))
    n = len(distinct)
    coding = {j:i for i, j in enumerate(distinct)}
    counts = np.zeros((n, n))
    for seq in sequences:
        print(seq) #just to visualize each sequency for each unique ID
        coded_seq = np.fromiter((coding[i] for i in seq), dtype=np.int)
        pairs = coded_seq[:-1] + n * coded_seq[1:]
        counts += np.bincount(pairs, minlength=n*n).reshape(n, n)
    totals = counts.sum(axis=0)
    totals[totals == 0] = 1     # avoid division by zero
    probs = counts / totals
    
    if prob == True:        
        return {(a, b): p for a, b in product(distinct, repeat=2) 
                for p in (probs[coding[b], coding[a]],) if p}, probs, distinct
                
    if prob == False:        
        return {(a, b): p for a, b in product(distinct, repeat=2) 
                for p in (counts[coding[b], coding[a]],) if p}, probs, distinct
            
            
def historical_transition_matrix(data,string_id = 'group', string_group = 'event', string_seq = 'sequence_num' ,seq = False, prob=True):
    
    '''
    First Approach -  VERY IMPORTANT need to sort by DATE to have the correct order in place for each group, make sure date is indexed
    
    data = data which contains the event, unique id and where date is the index (very important)
    string_id = name in the dataframe where the column represents unique ID's. 
    string_group = name in the dataframe where the column represents the selection of events (important that it is an string for this code)
    '''
    #convert dates to datetime 
    data.index = pd.to_datetime(data.index)
    unique_group = data[string_id].unique()
        
    if seq == True: 
        unique_ID = []
        for un in unique_group:
            data_ = data[data[string_id]==un]
            data_fin = data_.sort_index(ascending=True) # his part is very important for the sequence for each ID
            unique_ID.append(data_fin)
            
        unique_ID_seq = []
        for i in range(len(unique_ID)):
            for unique in unique_ID[i][string_seq].unique():
                data_ = unique_ID[i][unique_ID[i][string_seq] == unique]
                data_fin = data_.sort_index(ascending=True)
                unique_ID_seq.append(data_fin)
                
        events_seq = []
        for i in range(len(unique_ID_seq)):
            event_int = unique_ID_seq[i][string_group].tolist()
            event_ =' '.join(event_int)
            events_seq.append(event_.replace(" ",""))
            
        #generate the probability matrix and make sure the set_events are in order to implement it in a dataframe
        pairs, probs, set_events = cond_probs_np(events_seq)
        set_events = sorted(list(set_events))
        
        #make some visual improvements
        prob_mat = pd.DataFrame(index = set_events,columns = set_events)
        for x in pairs:
            prob_mat.loc[x[0],x[1]] = pairs[x]
        
        prob_mat = prob_mat.fillna(0)  
        prob_mat = np.round((prob_mat*100),2).astype(str) + "%"
        
    if seq == False:
        unique_ID = []
        for un in unique_group:
            data_ = data[data[string_id]==un]
            data_fin = data_.sort_index(ascending=True) # his part is very important for the sequence for each ID
            unique_ID.append(data_fin)
        
          
        events_seq = []
        for i in range(len(unique_ID)):
            event_int = unique_ID[i][string_group].tolist()
            event_ =' '.join(event_int)
            events_seq.append(event_.replace(" ",""))
            
        #generate the probability matrix and make sure the set_events are in order to implement it in a dataframe
        if prob == True:
            pairs, probs, set_events = cond_probs_np(events_seq, prob=True)
            set_events = sorted(list(set_events))
            
            #make some visual improvements
            prob_mat = pd.DataFrame(index = set_events,columns = set_events)
            for x in pairs:
                prob_mat.loc[x[0],x[1]] = pairs[x]
            
            prob_mat = prob_mat.fillna(0)  
            prob_mat = np.round((prob_mat*100),2).astype(str) + "%"
            
        if prob == False:
            pairs, probs, set_events = cond_probs_np(events_seq, prob=False)
            set_events = sorted(list(set_events))
            
            #make some visual improvements
            prob_mat = pd.DataFrame(index = set_events,columns = set_events)
            for x in pairs:
                prob_mat.loc[x[0],x[1]] = pairs[x]
            
            prob_mat = prob_mat.fillna(0)  
                  
    return pairs, prob_mat, set_events, events_seq


   

'''
Only grab the dataset with date, id and rating (nothing else needed and will reduce computation time)

Markov Matrix Calcuation

'''
#Use the test data set created
df = pd.read_csv('test_data_set.csv',encoding='utf-8')
df_final = convert_data(df,string_date = 'period_ending_dt', string_rating ='system_risk_grade', string_ID = 'ID', time_conv = 360, assumption=True)

#create the annual probability matrix n=1



'''
Historical Transition Matrix Calculation

'''
#make df_final compatible 
data = df_final.set_index('period_ending_dt')
data['system_risk_grade'] = data['system_risk_grade'].map({'A1':'A','A2':'B','B1':'C','B2':'D','C1':'E','C2':'F','D':'G'})


pairs, prob_mat, set_events, events_seq = historical_transition_matrix(data,string_id = 'ID', string_group = 'system_risk_grade', string_seq = 'sequence_num' ,seq = False)
prob_mat.iloc[-1][-1] = "100%"
final_historical_prob = pd.DataFrame(prob_mat)
final_historical_prob = final_historical_prob.set_index(Prob_matrix_df.index)
final_historical_prob.columns = Prob_matrix_df.columns

#compare the two:
print("\n\n")
print("Markov Annual Transition Matrix\n", Prob_matrix_df)
print("\n\n")
print("Historical Transitions\n", final_historical_prob)


#testing to get counts out in a matrix (Historical)
pairs2, prob_mat2, set_events2, events_seq2 = historical_transition_matrix(data,string_id = 'ID', string_group = 'system_risk_grade', string_seq = 'sequence_num' ,seq = False, prob=False)
prob_mat2.iloc[-1][-1] = "0.0"
final_historical_prob2 = pd.DataFrame(prob_mat2)
final_historical_prob2 = final_historical_prob2.set_index(Prob_matrix_df.index)
final_historical_prob2.columns = Prob_matrix_df.columns