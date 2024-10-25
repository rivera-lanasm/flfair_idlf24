## IDL Project 
# Pseudo code for Fair Fed Aggregration Method
# Pseudo code developed from Fair fed paper



'''
NOTATION: 

k: one client; K = all clients

w_t_k: weight for client k at time step t

n_k: size of the dataset for client k

beta :  a param that controls the fairness budget for each update, thus impacting 
        the trade-off between model accuracy and fairness. Higher values of beta
        result in fairness metrics having a higher impact on the model optimization.
        Note that at beta = 0, FairFed is equivalent to FedAvg

'''


## Initialize

# Initilize theta 0 (global model paramteter)
theta_0 = ?
# Initialize w_0_k
w_0_k = n_k / sum(n_k)

# SERVER SIDE:
# Server aggregates this statistic and shares w/ client ONCE at the start of training
S = {P(Y =1, A=0),P(Y =1, A=1)}

time_steps = 10

for t in range(time_steps):
    
    ''' STEP 1: Calculate Global Fairness Metric and Global Accuracy '''

    # Substep 1.a: CLIENT SIDE
    # Each client calculate their EOD metric and sends it to sever
    EOD_k = #long formula - check paper


    # Substep 1.b: SERVER SIDE
    # Server aggregates EOD metrics from clients to compute F_global
    # COMPUTE F_global such that:
    F_global = sum(EOD_k) # sum across all clients

    # SERVER SIDE --> computes global accuracy
    acc_global = sum(acc_local)

    ''' STEP 2: SERVER sends F_global and acc_global BACK to clients '''

    ''' STEP 3: CLIENT SIDE **IMPORTANT** '''
    # Each client computes its own metric gap
    for k in K:
        if F_local is None: # if local fairness metric is undefined, use accuracy as proxy
            delta_k = acc_local - acc_global
        else:
            delta_k = F_local - F_global
    
    ''' # STEP 4: SERVER SIDE: '''
    # SERVER can aggregate all the deltas across all clients
    avg_deltas = mean(delta_k) # for all K clients

        
    ''' STEP 5: CLIENT SIDE - Updating weights! '''
    # From the metric gap (delta_k), each client can compute its weight updates
    for k in k:
        w_t_k = w_t_k - beta * (delta_k - avg_deltas)

    ''' STEP 6: SEVER SIDE: Aggregate the new weights '''
    avg_w_t_k = mean(w_t_k)

    avg_w_theta = mean(w_t_k * theta_t_k)

    ## NOTE: might be missing one calculation here - I got confused by the paper

    # NEW THETA
    theta_t_plus1 = avg_w_theta / avg_w_t_k

    ''' STEP 7: SERVER SIDE --> BROADCAST NEW MODEL PARAMS TO CLIENT'''

 