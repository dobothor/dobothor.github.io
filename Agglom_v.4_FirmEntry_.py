#create data structures based on operations that must be done
#focus on the back-end

#-------------------------------------------------------------#
#Current

#draw to png

# -> firm entry
#   seems stable...
# add:  follow profits
#       niche filling
#       entry type tracker
#clean up
#   -make own package?
#-------------------------------------------------------------#
# SPEED GAINS
# check speed for
#   -initialize
#   -entry
#   -comp
#   -price_index
# make wage set only 10 options, propo to each firm's current
#   ditto price set
# improve price index update
#   -calc for 'regions' not for every position
#   -combine entry z-score calculations [test if speed gain]
# improve comp
#   -compare dema inside wage to outside wage
#   -compare prof_max_find inside wage to outside wage


# Function  |   Time (sec)  |   Notes
# wage      |   .2          | for 15 checks, goes up by .07 for every +5 checks
# comp      |   15-40       | doesn't vary with # firms in ~predictable way
# entry     |   12          | sample 60 locations, up to 5 entries, run for each sector 
# entry     |   8           | with only 4 firms entering using z_method  
#-------------------------------------------------------------#

# build roads

# spatial distribution statistic

#-------------------------------------------------------------#

#HEURISTIC STRATEGIES
#  -update price index 'every so often' [every 5 price&wage updates]
#  -form 'location index'               [randomly(?) weight price index, wage, demand]
#

#-------------------------------------------------------------#

#KNOWN PROBLEMS

#  with 'profitable entry'
#   wages ascend, very high profit
#   number of firms drops
#   firms don't live very long
#   eventually no firms left
#SUGGESTIONS
#travel not costly enough
#look at unemployment rate
#lower productivity
#
#-------------------------------------------------------------#
###
# firms don't consider own impact on price index
# firms don't consider changing wage effect on demand [matters most for local]
#KNOWN DIFFERENCES
# transpo cost additive

#-------------------------------------------------------------#

#0. Initialize 
#   -locations
#   -road
#   -distance matrix [least cost]

#1. vector of consumers
#   -choose who to work for [based on wages and where]
#   -choose amount to purchase [based on prices and where
# [[x,y],f,w]
# firm id? depends on data-type

#2. vector of firms
#   -choose what price to set
#   -choose what wage to set
# [[x,y],p,w,π]

#-------------------------------------------------------------#

from tkinter import *
import random
from math import *
import numpy    #note have to use numpy._____ for commands
import time
from itertools import *
from igraph import *
import copy
import pickle
from PIL import Image

def to_i(x,y):
    return((x-1)*100+y-1)

def to_c(i):
    return([i//100+1,i%100+1])

width = 100
height = 100
colors = ['black','red','blue','green']
List = []
cons=[]
firms=[]

for x in range(1,width+1):
    for y in range(1,height+1):
        if x==50:
            List.append([x,y,[0,0,0],0,0,3])
        else:
            List.append([x,y,[0,0,0],0,0,0])
#[x,y,[P0,P1,P2],av_w,dem,type]
        
###########Form Edge List
##g = Graph()
##g.add_vertices(len(List))
##edg=[]
##dire = list(product([0,1,-1],[0,1,-1]))
##for i in range(len(List)):
##    for y_dif, x_dif in dire:
##        if List[i][0]+x_dif < 1 or List[i][1]+y_dif < 1 or List[i][1]+y_dif > 100 or List[i][0]+x_dif > 100 :
##            continue
##        #g.add_edge(i,to_i(List[i][0]+x_dif,List[i][1]+y_dif),weight=1)
##        if List[i][5]==3:
##            edg.append( (i, to_i(List[i][0]+x_dif,List[i][1]+y_dif), .2) )
##        else:
##            edg.append( (i, to_i(List[i][0]+x_dif,List[i][1]+y_dif), 1) )
##        
##
##G = Graph.TupleList(edg, weights=True)
##na = []
##for i, n in enumerate(G.vs['name']):
##	na.append([n,i])
##na.sort()
##na = [i[1] for i in na]
###use na[i] to obtain vertex id of List[i]
###input("Pause1")
##
##t0 = time.time()
###dist = G.shortest_paths()
##weight = G.es['weight']     #check if shortest_paths produces smaller pkl
##dist = G.shortest_paths_dijkstra(weights=weight)
##print(time.time()-t0)
###dist[na[0][1]][na[303][1]]    #reference example
###G.shortest_paths_dijkstra(0,100,weights='weight')
##t0=time.time()
##func = lambda x: round(x,2)   #check speed
##dist = [list(map(func,i)) for i in dist]
##print(time.time()-t0)
###[[np.round(float(i), 2) for i in nested] for nested in outerlist]
##
            ##could make the dist matrix lower triangular to reduce memory
##            
##dist_out = open(r"C:\Users\somebody\Dropbox\paper\new papers\Clustering\dist3.pkl", 'wb')
##pickle.dump(dist,dist_out)
##dist_out.close()
##na_out = open(r"C:\Users\somebody\Dropbox\paper\new papers\Clustering\na3.pkl", 'wb')
##pickle.dump(na,na_out)
##na_out.close()
###########################

# Load in distance matrix to save time on simulation runs
# Distance matrix formation above in comments
t0=time.time()
dist_file = open(r"C:\Users\Thor\Dropbox\paper\new papers\Clustering\dist3.pkl", 'rb')
dist = pickle.load(dist_file)
dist_file.close()
na_file = open(r"C:\Users\Thor\Dropbox\paper\new papers\Clustering\na3.pkl", 'rb')
na = pickle.load(na_file)
na_file.close()
print(time.time()-t0)
#input("Pause2")
sig = 4
u = [.3,.3,.4] #u0 = .3 ; u1 = .3 ; u2 = .4
b = [.25,.25,.25]
##b = [[0,0,0],
##     [.25,.25,.25],
##     [1,0,0]] 
tw = .15
tg = .1
g = .6   #return to labor      {!}
A = 6  #productivity scalar  {!}
E = 2 #extra income  [active]

wi = 5
pi = 5
# [i,f,w]
for h in random.sample(list(range(1,10000)),300):    #w/o replacement
    cons.append([h,0,0])
for i in range(len(u)):
    for f in random.sample(list(range(1,10000)),10):
        firms.append([f,pi,wi,i,[1,1,1],0,0,0,0])
# [i,p,w,j,[Cj],π,labo,dema,age]



def work(h):
    #form list of wages would recieve from each firm
    #assign household to best wage firm [update cons[1] and cons[2]
    #construct a permanent vector for each house for dist to firms
    wages = [i[2]+1-exp(tw*dist[na[h[0]]][na[i[0]]]) for i in firms]
    choose =max([[n,i] for i,n in enumerate(wages)])
    h[1] = choose[1]
    h[2] = choose[0]
    if h[2]<0:
        h[2]=0
        h[1]=999
    return(h)

#[x,y,[P0,P1,P2],av_w,d]
def update_price_index(L):
    u_c = [i[3] for i in firms]
    for j in range(len(u)):
        if u_c.count(j)!=0:
            fu = [firms[i] for i,n in enumerate(firms) if n[3]==j]
            prices = [i[1] -1+exp(tg*dist[na[to_i(L[0],L[1])]][na[i[0]]]) for i in fu]
            L[2][j] = sum([i**(1-sig) for i in prices])**(1/(1-sig))
    #wage of nearest customer, [[largest wage of nearby firm]]  [for now choose 2nd]
    #adding L[3] and L[4] took run time from .6 to 1.5
    wages = [i[2] +1- exp(tw*dist[na[to_i(L[0],L[1])]][na[i[0]]]) for i in firms]
    L[3] = max(wages)
    dem = [(dist[na[to_i(L[0],L[1])]][na[i[0]]])**2 for i in cons]  #doesn't actually need to be updated...
    L[4] = sum(dem)
    return(L)

def demand(p,f,cons):
    #calculate demand for a given price
    #        w * (distance factor firm to customer * price)**-sig / price index from firm_j for customer
    cust = [ u[firms[f][3]]*(i[2]+E)* (exp(tg*dist[na[firms[f][0]]][na[i[0]]]) + p-1)**-sig / List[i[0]][2][firms[f][3]]**(1-sig) for i in cons]
    cust = [(abs(i)+i)/2 for i in cust]
    #print(p,max(cust), sum(cust)/len(cust))
    #input("pause")
    ##!## if issue with list index out of range persists then write an error catch here
    #cust = [0 for i,n in enumerate(cust) if n<0]
    fam = [ j[4][firms[f][3]] * (p-1+exp(tg*dist[na[firms[f][0]]][na[j[0]]]))**-sig / List[j[0]][2][firms[f][3]]**-sig for j in firms]
    fam = [(abs(i)+i)/2 for i in fam]
##    print(p)
##    print(cust)
##    print(sum(cust))
##    print(fam)
##    print(sum(fam))
##    input("Pause")        
    return(sum(cust)+sum(fam))

def bundle(Q,L,f,j):
    ps = 1
    for k in range(len(u)):
        ps = ps* (b[k]/List[firms[f][0]][2][k])**b[k] 
    C = ( Q/ (A*L**g *ps) )**(1/sum(b)) * b[j] / List[firms[f][0]][2][j]
    return(C)

def wage(w,f):  ##wage a bit slow...can map the loops?
    #construct hypothetical for labor supply and demand
    global firms, cons
    cans = copy.deepcopy(cons) 
    wold = firms[f][2]
    firms[f][2]=w
    cans = list(map(work,cans))
    dema=list(map(demand,pri,repeat(f),repeat(cans)))
    workers = [i[1] for i in cans]
    if f==-1:
        labo = workers.count(len(firms)-1)
    else:
        labo = workers.count(f)
    #return firms and cons to original settings
    firms[f][2] = wold
    if labo==0:
        return([0,[0,0,0]])
    Cj = []
    for j in range(len(u)):    #find optimal bundle for each Q
        fu = [firms[i] for i,n in enumerate(firms) if n[3]==j]       ##!##
        Cj.append(list(map(bundle, dema, repeat(labo),repeat(f),repeat(j))))
    cost=[i for i in Cj]
    for j in range(len(u)):    #calc cost of composite for each Q
        cost[j] = [i*List[firms[f][0]][2][j] for i in cost[j]]
    tc = []
    for j in range(len(Cj[0])):  #calc the total cost of bundle + labor
        tc.append(sum([i[j] for i in cost])+w*labo)
    tr = [a*b for a,b in zip(pri,dema)]
    profit = [a-b for a,b in zip(tr,tc)]
    try:
        pr = [i for i,n in enumerate(profit) if n==max(profit)]
    except:
        print(tr,'\n',tc)
        input("Pause")
    bun = [i[pr[0]] for i in Cj]
    #Q_check = labo**g *Cj[0][0]**b[0] *Cj[1][0]**b[1] *Cj[2][0]**b[2]
##    print(w,labo)##    #print("TR",tr)##    print("TR", tr)##    #print("bundle",Cj)##    print("TC",tc)##    print(profit)##    input("pause")        #calculate profit from each        #return only best profit FOR THIS L  [later take best among L]    #return [profit maximizing price, C1, C2, C3]
    return([max(profit),w,labo,pri[pr[0]],list(map(round,bun,repeat(3))),round(dema[pr[0]],3)])

def update_firm(f,prof):
    best = [j for j,n in enumerate(prof) if n==max(prof)]
    firms[f][2]=prof[best[0]][1]  #wage     
    firms[f][1]=prof[best[0]][3]  #price
    firms[f][4]=prof[best[0]][4]  #bundle
    firms[f][5]=prof[best[0]][0]  #profit
    firms[f][6]=prof[best[0]][2]  #labo
    firms[f][7]=prof[best[0]][5]  #dema
    firms[f][8]+=1
    return

def draw(name):
    Lost = [i[5] for i in List]
    for h in cons:
        Lost[h[0]] = 1
    for f in firms:
        Lost[f[0]] = 2
    color = [(0,0,0),(255,0,0),(0,0,255),(0,255,0)]
    pixels = [color[i] for i in Lost]
    array = numpy.array(pixels,dtype=numpy.uint8)
    array = numpy.reshape(array,(100,100,3))
    new_image = Image.fromarray(array)  #image is mirrored diagonally -> transform nparray 1st
    new_image.save(name)
    return

################
# Initializing #
################
t0=time.time()
cons = list(map(work,cons))  
print(time.time()-t0)
t0=time.time()
List = list(map(update_price_index,List))  #takes about 5 seconds
print(time.time()-t0)

# cons  [i,f,w]
# firms [i,p,w,j,[Cj],π,dema,labo] 
pri = [i for i in range(1,10)]
wa = [i for i in range(4,15)]
t0=time.time()
print("Initializing")
for i in range(len(firms)):
##    t0=time.time()
##    dema=list(map(demand,pri,repeat(i)))
##    print(time.time()-t0)
    t0=time.time()
    wami = firms[i][2]-10
    if wami<2:
        wami=2
    wama = firms[i][2]+10
    step = wama//10
    wa = [i for i in range(wami,wama+1,2)]
    prof=list(map(wage,wa,repeat(i)))
    #print("Price Index Update --",time.time()-t0)
    if max(prof)[0] <= 0:
        firms[i][5]=-999
    else:    
        update_firm(i,prof)
        #op.append([firms[i][0],prof[best[0]]])
        cons = list(map(work,cons))
        #firms[i][1] = prof[0][0][3]
        #firms[i][2] = prof[0][0][1]
    #update price and wage for the firm
    #print(op[i],'\n',dema)
    E = sum([ (abs(i[5])+i[5])/2 for i in firms])/len(cons)
    if E<2:
        E=2
print(time.time()-t0)

##for i in op:
##    print(i)
for i in firms:
    print(i)
# like this I can take a given firm
# and calc demand by
# mapping over a set of prices
# given all the other prices and wages

#Remove dead firms...come up with a better way
#numbers[:] = [n for n in numbers if not odd(n)]  # ahh, the beauty of it all
#make this a function
flag=0
c=0
while flag==0:
    if firms[c][5]==-999:
       del firms[c]
       c-=1
    c+=1
    if c>=len(firms):
        flag=1
#input("Pause, dead check")
        
List = list(map(update_price_index,List))

####################
# Firm Competition #
####################


def comp():
    global List, cons,E,firms
    for r in range(3):

        t0=time.time()
        List = list(map(update_price_index,List))
        #print("Price Index Update --",time.time()-t0)
        farms = copy.deepcopy(firms)
        orde = random.sample(list(range(0,len(firms))),len(firms))
        #for k in cons:
        #    print(k[2])
        # firms [i,p,w,j,[Cj],π]
        for i in orde:
            #profits = [j[5] for j in firms]
            #low = [j for j,n in enumerate(profits) if n==min(profits)]  ##!## no good
            #dema=list(map(demand,pri,repeat(i)))
            #put dema inside of wage function ~ doubled run time but now firms consider wage effect on demand
            wami = firms[i][2]-10
            if wami<2:
                wami=2
            wama = firms[i][2]+10
            step = wama//10
            wa = [i for i in range(wami,wama+1,2)]
            prof=list(map(wage,wa,repeat(i)))
            if max(prof)[0] <= 0:
                #del firms[i]
                firms[i][5]=-999
                firms[i][2]=-999
                #print(firms[i][0],'dead')
            else:    
                update_firm(i,prof)

            cons = list(map(work,cons))
                #print(firms[i][0],prof[best[0]])
            #print(farms[i])
            #print(firms[i],'\n')
        #input("Pause: ")
        flag=0
        c=0
        while flag==0:
            if firms[c][5]==-999:
               del firms[c]
               c-=1
            c+=1
            if c>=len(firms):
                flag=1
        E = sum([ (abs(i[5])+i[5])/2 for i in firms])/len(cons)
        wav = sum([i[2] for i in firms])/len(firms)     #average wage firm pays
        wavc = sum([i[2] for i in cons])/len(cons)      #average wage customer receives
        pav = sum([i[1] for i in firms])/len(firms)     #average price firm receives
        print("Round",r,"-- #firms:",len(firms), "-- E:", E,"-- π:",E*len(cons)/len(firms))
        print("wav:",wav,"-- wavc:",wavc,"-- pav:",pav)
        print("time:",time.time()-t0,"-- age:",sum([i[8] for i in firms])/len(firms))   #check if prices and wages have stabilized


        cons = list(map(work,cons))
        #print(set([i[1] for i in cons]))
        if E<2:
            E=2
comp()
    
#firms.append([f,pi,wi,i,[1,1,1],0,0,0])
# [i,p,w,j,[Cj],π,labo,dema]

def objective(place,k):
    global firms,cons
    firms.append([to_i(place[0],place[1]),5,5,k,[0,0,0],0,0,0,0])
    wami = min([i[2] for i in firms]) - 10
    wama = max([i[2] for i in firms]) + 10
    if wami<2:
        wami=2
    step = wama//10
    wa = [i for i in range(wami,wama+1,step)]
    prof=list(map(wage,wa,repeat(-1)))
    #print(prof)
    del firms[-1]
    if max(prof)[0] <= 0:
        #print('none')
        return([[-999]])
    else:
        best = [j for j,n in enumerate(prof) if n==max(prof)]
        #print(prof[best[0]][0])
        return([prof[best[0]],[-999,place[0],place[1]]])

def entry(k):
    global firms
    spac = list([[i,j] for i in range(1,101) for j in range(1,101)])
    new = []
    prof_comp = [i[5] for i in firms]
    prof_comp.sort(reverse=True)
    for i in random.sample(spac,60):
        prof = objective(i,k)
        #print(i,prof)
        #if profit > 90 percentile, enter
    ##    if prof[0][0] > prof_comp[int(len(firms)*.1)]:
    ##        c+=1
    ##        firms.append([to_i(i[0],i[1]),0,0,0,[0,0,0],0,0,0])
    ##        update_firm(-1,prof)
    ##        if c>3:
    ##            break
        #take top 5 percentile of sample and enter
        if prof[0][0] > prof_comp[int(len(firms)*.5)]:
            new.append(prof)
                   
    new.sort(reverse=True)
    for i in range(min(5,len(new))):
        firms.append([to_i(new[i][1][1],new[i][1][2]),0,0,k,[0,0,0],0,0,0,0])
        #print("\n new:",new[i])
        update_firm(-1,new[i])
        #adjust sample space based on number of succesfull entries
    #print('\n')
    return(min(5,len(new)))

for i in range(50):
    t0=time.time()
    c=0
    for k in range(len(u)):
        c+=entry(k)
    name = "D:\\thor\'s folder\\grad stuff\\Transpo\\Transport Econ\\data\\simulation\\"+str(i)+"entry.png"
    draw(name)
    print("\nEntry ",i," -- ","#new: ",c," time: ", time.time()-t0,sep="")
    print("----------------------------------------------------------------")
    
    comp()
    name = "D:\\thor\'s folder\\grad stuff\\Transpo\\Transport Econ\\data\\simulation\\"+str(i)+"pcomp.png"
    draw(name)

####pn = Image.new(png.mode, png.size)
####pn.putdata(l)
####pn.save('D:\\thor\'s folder\\grad stuff\\Transpo\\1950 ~\\metro_red.png')

#space = hp.choice('pl',spac)
#tpe_algo = tpe.suggest
#tpe_trials = Trials()
#tpe_best = fmin(fn=objective, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=50)
#print(tpe_best)
##profs = [i[5] for i in firms]
##bost = [i for i,n in enumerate(profs) if n==max(profs)][0]
##print(bost)
##print(objective( to_c(firms[bost][0]) ))
input("Pause")






















##############
# Firm Entry #          # add weight for profit from each firm
##############              # enter where firms known to be profitable

#List Descript
#[x,y,[P0,P1,P2],av_w,dem,type]
# 2-P  3-w  4-dem
z_hist = []

for a in range(10):
    print("-----------------------------------------------\n")
    print("Firms Entry --",a)
    t0=time.time()
    p0 = max([i[2][0] for i in List])
    p0_= min([i[2][0] for i in List])
    p1 = max([i[2][1] for i in List])           #should care about own price index differently
    p1_= min([i[2][1] for i in List])
    p2 = max([i[2][2] for i in List])
    p2_= min([i[2][2] for i in List])
                                                #subtract the minimum for each to better map to [0,1]?
    #av_wage
    y_ = min([i[3] for i in List])
    y =  max([i[3] for i in List])              # [ (i-min(x)) / (max(x)-min(x)) for i in x]

    #distance ~ dem
    x = max([i[4] for i in List])
    x_= min([i[4] for i in List])

    z = [ .3*sum([ (i[2][0]-p0_)/(p0-p0_), (i[2][1]-p1_)/(p1-p1_), (i[2][2]-p2_)/(p2-p2_) ]) /3 +
          .3*(i[3]-y_)/(y-y_) +
          .3*(i[4]-x_)/(x-x_) for i in List]
    z2 = [ .1*sum([ (i[2][0]-p0_)/(p0-p0_), (i[2][1]-p1_)/(p1-p1_), (i[2][2]-p2_)/(p2-p2_) ]) /3 +
          .45*(i[3]-y_)/(y-y_) +
          .45*(i[4]-x_)/(x-x_) for i in List]
    z3 = [ .1*sum([ (i[2][0]-p0_)/(p0-p0_), (i[2][1]-p1_)/(p1-p1_), (i[2][2]-p2_)/(p2-p2_) ]) /3 +
          .1*(i[3]-y_)/(y-y_) +
          .8*(i[4]-x_)/(x-x_) for i in List]
    z4 = [ .1*sum([ (i[2][0]-p0_)/(p0-p0_), (i[2][1]-p1_)/(p1-p1_), (i[2][2]-p2_)/(p2-p2_) ]) /3 +
          .8*(i[3]-y_)/(y-y_) +
          .1*(i[4]-x_)/(x-x_) for i in List]
    #### !      ! ####
    #Firm should enter at min(z)  # List[ [i for i,n in enumerate(z) if n==min(z)][0] ]
    num = [i[3] for i in firms]
    cou = []
    for i in range(len(u)):
        cou.append(num.count(i))

    z_ = [i for i,n in enumerate(z) if n==min(z)][0]
    z_hist.append(z_)
    z_2 = [i for i,n in enumerate(z2) if n==min(z2)][0]
    z_hist.append(z_2)
    z_3 = [i for i,n in enumerate(z3) if n==min(z3)][0]
    z_hist.append(z_3)
    z_4 = [i for i,n in enumerate(z4) if n==min(z4)][0]
    z_hist.append(z_4)

    u_ = [i for i,n in enumerate(cou) if n==min(cou)][0]
        
    pav = sum([i[1] for i in firms])/len(firms)
    wav = sum([i[2] for i in firms])/len(firms)
    #firms [i,p,w,j,[Cj],π]
    firms.append([ z_,pav,int(wav),u_,[1,1,1],0,0,0])
    firms.append([ z_2,pav,int(wav),u_,[1,1,1],0,0,0])
    firms.append([ z_3,pav,int(wav),u_,[1,1,1],0,0,0])
    firms.append([ z_4,pav,int(wav),u_,[1,1,1],0,0,0])
    randos=[]
    for i in range(len(u)):
        for f in random.sample(list(range(1,10000)),2):
            firms.append([f,pav,int(wav),i,[1,1,1],0,0,0])
            randos.append(f)

    ######################################
    #find most profitable location# (...only 4 million choices...)
    #two ways
            #use existing functions (takes .3 sec for each check)
            #do it by scratch (pretend I'm an existing firm at a location with price and wage...4 million choices later...)

    ######################################      ###!HERE!###
    #1. objective function
        #pass in x and y
        # run wage, then
        #best = [j for j,n in enumerate(prof) if n==max(prof)]
        #prof[best[0]][0]  #profit
    #2. domain space
    #3. optimization algo
    #4. Results
#List Descript
#[x,y,[P0,P1,P2],av_w,dem,type]
#firms [i,p,w,j,[Cj],π]
#print(space)


    ######################################      ###!HERE!###
    
    #enter where other firms are profitable
    #####

    ####

    #send in a firm of each industry where price index is high
            
    print("New firms:", z_, z_2,z_3,z_4,u_,time.time()-t0,'\n')
    t1=time.time()
    comp()
    print("Comp time:",time.time()-t1,"Number of firms:",len(firms))
    print("Randos --", [i in [j[0] for j in firms] for i in randos])
    print("Z_spot --", [i in [j[0] for j in firms] for i in z_hist])
    #for i in range(len(firms)):
    #    print(firms[i])
        
##for i in range(len(firms)):
##    print(firms[i])
print(z_hist)
print([i in [j[0] for j in firms] for i in z_hist])
##z = [ .3*sum([i[2][0]/max(p0),i[2][1]/max(p1),i[2][2]/max(p2)])/3 +
##      .3*(i[3]+abs(min(y)))/max(y) +
##      .3*(i[4]/max(x)) for i in List]
     
##for i in range(len(u)):
##    for f in random.sample(list(range(1,10000)),10):
##        firms.append([f,pi,wi,i,[1,1,1],0])

#[(i-min(x))/(max(x)-min(x)) for i in x]



#calculate average price index across locations
#sum([sum(i[2])/len(i[2]) for i in List])/len(List)
#Average wage per worker
#sum([i[2] for i in cons])/len(cons)
#[i[2] for i in cons].count(0)
#workers = [i[1] for i in cons]
#[workers.count(i) for i in set(workers)]


#calculate profit of firms
#for i in firms:
#    print(i[1]*i[7] - i[2]*i[6] - sum([a*b for a,b in zip( List[i[0]][2],i[4])]))



for h in cons:
    List[h[0]][5] = 1
for f in firms:
    List[f[0]][5] = 2

root=Tk()
w = Canvas(root,width=width, height=height)
w.pack()

w.create_rectangle(0,0,width,height,fill=colors[0])

for i in List:
    w.create_rectangle((i[0],i[1])*2,outline="",fill=colors[i[5]])

#draw image to png
    
#normalize
##def normalize(edge):
##    n1,n2,n3 = edge
##    if n1>n2:
##        n1,n2 = n2,n1
##    return(n1,n2,n3)
