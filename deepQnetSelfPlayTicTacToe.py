# Projet personnel : apprentissage par renforcement pour le Tic-Tac-Toe (morpion en français).
# Ce script entraîne un agent Deep Q-Network (DQN) en self-play
# (apprentissage par auto-jeu, c'est-à-dire jeu contre soi-même).

# L'état du plateau est représenté par 18 entrées :
# 9 cases pour le joueur actuel et 9 cases pour l'adversaire.
# Le réseau renvoie 9 Q-valeurs, une pour chaque action possible.

# L'agent privilégie l'action ayant la plus grande Q-valeur,
# c'est-à-dire le coup dont la récompense espérée est la plus élevée.

# Grâce aux principes de l'apprentissage par renforcement,
# le DQN (réseau de neurones) apprend progressivement à estimer
# les Q-valeurs optimales afin d'améliorer ses performances
# et de maximiser ses chances de victoire.

# Ce fichier contient à la fois la définition du réseau,
# l'environnement, la boucle d'entraînement,
# ainsi que l'affichage en direct de l'évolution
# de la récompense moyenne.


import numpy as cp #remplacer numpy par cupy pour les calculs matriciels par GPU
import numpy as np
import time


# Fonctions d'activation utilisées dans le réseau.
# Le paramètre s sert ici de facteur d'échelle

def sigmoid(x,s):
    y = cp.maximum(x,-70)
    return s/(1+cp.exp(-y))
    
def dsigmoid(x,s):
    sig = sigmoid(x,1)
    return s * sig * (1-sig)

def tanh(x,s):
    y = cp.clip(x,-70,70)
    ne = cp.exp(-y)
    pe = cp.exp(y)
    return s*(pe-ne)/(pe+ne)
def dtanh(x,s):
    return s*(1 - cp.power(tanh(x,1),2)) 


def silu(x,s):
    return x*sigmoid(x,1)
def dsilu(x,s):
    sig = sigmoid(x,1)
    return sig*(1+x*(1-sig))

def relu(x,s):
    return cp.where(x > 0, x, 0)
def drelu(x,s):
    return cp.where(x > 0, 1, 0)

def identity(x,s):
    return x
def didentity(x,s):
    return x*0+1


# Fonctions de coût.
# Le code permet d'utiliser soit une MSE(Mean Squared Error=Erreur Quadratique Moyenne) classique, soit une perte de Huber

def MSE(A,Y,delta):
    return cp.power(A-Y,2)/2
def dMSE(A,Y,delta):
    return A-Y

def Huber(A,Y,delta):
    e = cp.abs(A-Y)
    return cp.where(e <= delta, cp.power(e,2)/2, delta*(e - delta/2))
def dHuber(A,Y,delta):
    e = cp.abs(A-Y)
    return cp.where(e <= delta, (A-Y), delta*cp.where((A-Y) >= 0, 1, -1))
    

# Initialisation aléatoire des poids.
# Ici, l'initialisation de He est utilisée pour garder des valeurs de variance raisonnables au début de l'entraînement.

def He_init_normal(fanin,fanout):
    return (cp.random.randn(1,fanout,fanin).astype(cp.float32)*cp.sqrt(2/fanin)).astype(cp.float32)
def He_init_uniform(fanin,fanout):
    limit = cp.sqrt(6/fanin)
    return cp.random.uniform(-limit,limit,(1,fanout,fanin)).astype(cp.float32)


# Classe représentant le réseau de neurones utilisé comme Q-network.
# Elle stocke la structure du réseau, les poids, les biais, les gradients de la rétropropagation,
# ainsi que les paramètres d'optimisation et d'exploration.

class NET():
    def __init__(self,sh,opti,bs,rb_size,lr,gamma,alpha,eps_start,eps_end,eps_decay):
        self.opti = opti
        self.sh = sh
        self.nh = sh.shape[0]-1
        self.batch_size = bs
        self.rb_size = rb_size
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.f = silu
        self.df = dsilu
        self.lf = identity
        self.ldf = didentity
        self.lfscale = 1

        self.lossf = Huber
        self.dlossf = dHuber

        self.inti_distrib = He_init_normal

        self.bloss = cp.zeros(bs)

        self.X = None
        self.Y = None
        self.Z = [None for i in range(self.nh+1)]
        self.A = [None for i in range(self.nh+1)]
        self.W = [self.inti_distrib(sh[i,0],sh[i,1]) for i in range(self.nh+1)]
        self.B = [0*cp.random.randn(1,sh[i,1],1).astype(cp.float32) for i in range(self.nh+1)]

        self.dZ = [None for i in range(self.nh+1)]
        self.dW = [cp.zeros((1,sh[i,1],sh[i,0]),dtype=cp.float32) for i in range(self.nh+1)]
        self.dB = [cp.zeros((1,sh[i,1],1),dtype=cp.float32) for i in range(self.nh+1)]

        self.MW = [cp.zeros((1,sh[i,1],sh[i,0]),dtype=cp.float32) for i in range(self.nh+1)]
        self.VW = [cp.zeros((1,sh[i,1],sh[i,0]),dtype=cp.float32) for i in range(self.nh+1)]
        self.MB = [cp.zeros((1,sh[i,1],1),dtype=cp.float32) for i in range(self.nh+1)]
        self.VB = [cp.zeros((1,sh[i,1],1),dtype=cp.float32) for i in range(self.nh+1)]

        self.t = 1

# Propagation avant : calcule les activations couche par couche à partir de l'entrée courante n.X.

def forward_propagation(n):
    for i in range(n.nh+1):
        if i == 0:
            AP = n.X
        else:
            AP = n.A[i-1]
        W = n.W[i]
        B = n.B[i]
        Z = W @ AP + B
        if i == n.nh:
            A = n.lf(Z,n.lfscale)
        else:
            A = n.f(Z,1)
        n.Z[i] = Z
        n.A[i] = A
        

# Rétropropagation : calcule les gradients des poids et des biais à partir de la sortie attendue n.Y.

def back_propagation(n):
    for i in range(n.nh,-1,-1):
        if i == n.nh:
            Y = n.Y
            A = n.A[n.nh]
            n.bloss = cp.sum(n.lossf(A,Y,1),axis=1)
            dA = n.dlossf(A,Y,1)
            Z = n.Z[i]
            dZ = n.ldf(Z,n.lfscale) * dA
        else:
            A = n.A[i]
            dA = cp.transpose(n.W[i+1], (0, 2, 1)) @ n.dZ[i+1]
            Z = n.Z[i]
            dZ = n.df(Z,1) * dA

        n.dZ[i] = dZ   

        B = n.B[i]
        W = n.W[i]
        if i > 0:
            AP = n.A[i-1]
        else:
            AP = n.X

        dB = cp.expand_dims(cp.sum(dZ,axis=0),axis=0)
        dW = cp.expand_dims(cp.sum(dZ @ cp.transpose(AP, (0, 2, 1)),axis=0),axis=0)
        n.dB[i] = dB
        n.dW[i] = dW



# Mise à jour des paramètres.
# Le code supporte SGD ou Adam selon la valeur de n.opti.

def update(n,lr):
    gW = [n.dW[i]/n.batch_size for i in range(n.nh+1)]
    gB = [n.dB[i]/n.batch_size for i in range(n.nh+1)]
    #SGD
    if n.opti == 0:
        for i in range(n.nh+1):
            n.W[i] = n.W[i] - lr*gW[i]
            n.B[i] = n.B[i] - lr*gB[i]

    #ADAM
    if n.opti == 1:
        beta1 = 0.9
        beta2 = 0.999
        eps = 10**(-8)
        for i in range(n.nh+1):
            n.MW[i] = beta1*n.MW[i] + (1 - beta1)*gW[i]
            n.VW[i] = beta2*n.VW[i] + (1 - beta2)*cp.power(gW[i],2)
            MWhat = n.MW[i]/(1-cp.power(beta1,n.t))
            VWhat = n.VW[i]/(1-cp.power(beta2,n.t))

            n.W[i] = n.W[i] - lr*(MWhat)/(cp.sqrt(VWhat) + eps)

            n.MB[i] = beta1*n.MB[i] + (1 - beta1)*gB[i]
            n.VB[i] = beta2*n.VB[i] + (1 - beta2)*cp.power(gB[i],2)
            MBhat = n.MB[i]/(1-cp.power(beta1,n.t))
            VBhat = n.VB[i]/(1-cp.power(beta2,n.t))

            n.B[i] = n.B[i] - lr*(MBhat)/(cp.sqrt(VBhat) + eps)
        n.t += 1

# Politique epsilon-greedy :
# - avec probabilité eps, on choisit un coup valide aléatoire ;
# - sinon on choisit l'action valide de meilleure Q-valeur prédite.

def policy(QO,state,inv_mask,eps):
    if np.random.rand() < eps:
        args = np.argwhere(inv_mask[:,0] == 0)
        return int(np.random.choice(args[:,0]))
    
    else:
        QO.X = cp.expand_dims(state,axis=0) 
        forward_propagation(QO)
        return int(cp.argmax(QO.A[-1][0,:,0] + inv_mask[:,0]*(-10**8),axis = 0))


# Fonctions liées à l'environnement Tic-Tac-Toe.

def reset_env():
    return np.zeros((3,3),cp.int32)


def mask_invalid_actions(env):
    inv_actions = cp.where(env == 0, 0, 1)
    inv_mask = cp.zeros((9,1),dtype=cp.int32)
    
    for i in range(3):
        for j in range(3):
            inv_mask[i*3+j,0] = cp.where(inv_actions[i,j] == 1, 1, 0)
            
    return inv_mask

# Conversion du plateau en représentation réseau :
# les 9 premières cases indiquent les positions du joueur 1,
# les 9 suivantes celles du joueur -1.

def state_from_env(env):
    state = cp.zeros((18,1),dtype=cp.float32)
    for i in range(3):
        for j in range(3):
            state[i*3+j,0] = cp.where(env[i,j] == 1, 1, 0)
            state[9+i*3+j,0] = cp.where(env[i,j] == -1, 1, 0) 
    
    return state


# Détection de fin de partie et attribution d'une récompense.
# Le code renvoie un booléen de terminaison et la récompense associée.

def terminal_rewards_from_env(env):
    
    for i in range(3):
        if env[i,0] != 0 and (env[i,0] == env[i,1] == env[i,2]):
            return 1 , env[i,0]
        if env[0,i] != 0 and (env[0,i] == env[1,i] == env[2,i]):
            return 1 , env[0,i]
        
    if env[0,0] != 0 and (env[0,0] == env[1,1] == env[2,2]):
        return 1 , env[0,0]
    if env[2,0] != 0 and (env[2,0] == env[1,1] == env[0,2]):
        return 1 , env[2,0]
    
    
    if not cp.any(env.ravel() == 0):
        return 1,0
    
    return 0,0.01


# Joue un coup depuis l'état courant puis renvoie l'état précédent,
# l'action choisie et le nouvel environnement.

def play_state(QO,eps,env):
    state = state_from_env(env)
    inv_mask = mask_invalid_actions(env)
    temp = QO.batch_size
    QO.batch_size = 1
    action = policy(QO,state,inv_mask,eps)
    QO.batch_size = temp
    statelast = cp.copy(state)

    for i in range(9):
        if i == action:
            env[int(np.floor(i/3)),i%3] = 1
            
    return statelast,action,env




# Joue un épisode complet en self-play et remplit le replay buffer.
# Le changement de signe env = env * (-1) permet de réexprimer le plateau
# du point de vue du joueur qui doit jouer.

def do_episode(QO,eps,rb,index_rb,test_phase):
    env = reset_env()
    if np.random.rand() < 0.5:
        if test_phase:
            statelast,action,env = play_state(QO,1,env)
        else:
            if np.random.rand() < 0.5:
                statelast,action,env = play_state(QO,0,env)
            else:
                statelast,action,env = play_state(QO,0.5,env)

        env = env*(-1)
        if test_phase:
            print(env)
            print()
    
    for t in range(9):
        if test_phase:
            if t == 0:
                statelast,action,env = play_state(QO,1,env)
            else:
                statelast,action,env = play_state(QO,0,env)
        else:
            statelast,action,env = play_state(QO,eps,env)
        
        if test_phase:
            print(env)
            print()
        done,reward = terminal_rewards_from_env(env)
        if done == 1:
            statenew = state_from_env(env)
            inv_mask_next = mask_invalid_actions(env)
            replay = [statelast,action,reward,statenew,done,inv_mask_next]
            for i in range(6):
                rb[i][index_rb] = replay[i]
            index_rb = (index_rb + 1) % QO.rb_size
            break
        
        if (np.random.rand() < 0.5) or test_phase:
            newstatelast,newaction,env = play_state(QO,0,env*(-1))
        else:
            newstatelast,newaction,env = play_state(QO,0.5,env*(-1))

        env = env*(-1)
        if test_phase:
            print(env)
            print()

        statenew = state_from_env(env)
        inv_mask_next = mask_invalid_actions(env)
        done,reward = terminal_rewards_from_env(env)

        replay = [statelast,action,reward,statenew,done,inv_mask_next]
        for i in range(6):
            rb[i][index_rb] = replay[i]
        index_rb = (index_rb + 1) % QO.rb_size

        if done == 1:
            break


    return index_rb,reward




# Étape d'entraînement DQN.
# La cible utilise le réseau cible QT et le choix d'action du réseau en ligne QO
# avec un masque pour interdire les coups invalides.

def update_QNET(QO,QT,rb):
    r = cp.random.randint(0, QO.rb_size, QO.batch_size)
    states = rb[0][r]
    actions = rb[1][r]
    rewards = rb[2][r]
    stateps = rb[3][r]
    dones = rb[4][r]
    mask = rb[5][r]

    QO.X = stateps
    QT.X = stateps
    forward_propagation(QO)
    forward_propagation(QT)
    
    qt = QT.A[-1]

    qastar = QO.A[-1] + mask*(-10**8)

    y = QO.alpha*(rewards + QO.gamma*(1-dones)*qt[cp.arange(QO.batch_size),cp.argmax(qastar[:,:,0],axis=1),0])
    
    QO.X = states
    forward_propagation(QO)
    q = QO.A[-1]

    QO.Y = cp.copy(q)
    QO.Y[cp.arange(QO.batch_size),actions,0] = y

    back_propagation(QO)
    update(QO,QO.lr)


# Boucle principale d'entraînement :
# remplissage initial du replay buffer, épisodes de self-play,
# mises à jour du réseau et synchronisation périodique du réseau cible.

def train(QO,QT,QPREV,rep):
    rb_s = cp.zeros((QO.rb_size,QO.sh[0,0],1),dtype=cp.float32)
    rb_a = cp.zeros(QO.rb_size,dtype=cp.int32)
    rb_r = cp.zeros(QO.rb_size,dtype=cp.float32)
    rb_sp = cp.zeros((QO.rb_size,QO.sh[0,0],1),dtype=cp.float32)
    rb_d = cp.zeros(QO.rb_size,dtype=cp.float32)
    rb_im = cp.zeros((QO.rb_size,QO.sh[-1,1],1),dtype=cp.float32)
    rb = [rb_s,rb_a,rb_r,rb_sp,rb_d,rb_im]
    index_rb = 0
    eps = 1

    for i in range(QO.rb_size):
        index_rb,empty = do_episode(QO,eps,rb,index_rb,0)
        
    
    
    REWARDS = []
    REWARDSav = []
    for i in range(rep):
        index_rb,empty = do_episode(QO,eps,rb,index_rb,0)
        
        rew = rb_r[(index_rb - 1)%QO.rb_size]
        REWARDS.append(rew)


        update_QNET(QO,QT,rb)
        
        if (i % 500) == 499:
            for j in range(QO.nh+1):
                QT.W[j] = cp.copy(QO.W[j])
                QT.B[j] = cp.copy(QO.B[j])


        eps = QO.eps_end + (QO.eps_start - QO.eps_end) * np.exp(-i/QO.eps_decay)
        if (i%10000) == 0:
            print(str(i)+"/"+str(rep)+" episode") 
        if i == (rep-1):
            print(str(rep)+"/"+str(rep)+ " epsiode") 

    return REWARDS


cp.random.seed(0)

# Hyperparamètres et architecture du réseau.

rep = 100000
batch_size = 32
rb_size = 50000
lr = 0.001
gamma = 0.95
alpha = 1
eps_start = 1
eps_end = 0.1
eps_decay = rep/5


netshape = np.array([
    [18,32],
    [32,32],
    [32,9],
])
QO = NET(netshape,1,batch_size,rb_size,lr,gamma,alpha,eps_start,eps_end,eps_decay)
QT = NET(netshape,1,batch_size,rb_size,lr,gamma,alpha,eps_start,eps_end,eps_decay)
QPREV = NET(netshape,1,batch_size,rb_size,lr,gamma,alpha,eps_start,eps_end,eps_decay)
for j in range(QO.nh+1):
    QT.W[j] = cp.copy(QO.W[j])
    QT.B[j] = cp.copy(QO.B[j])

temp = [[None,None,None,None,None] for i in range(20)]

print("s")
t0 = time.perf_counter()
REWARDS = train(QO,QT,QPREV,rep)
dt = time.perf_counter() - t0
print(dt)
for i in range(20):
    print("episode "+str(i))
    do_episode(QO,0,temp,0,1)
    print("----------------------------------")
