# Simulation 3D simple d'un guidage.
# Ce script affiche deux poursuivants et une cible dans une scène 3D.

#
# Raccourcis clavier :
# - ZQSD / WASD : déplacer la caméra
# - Shift / Espace : descendre / monter
# - Souris : orienter la vue
# - Q : libérer / reprendre la souris
# - E : accélérer le déplacement de caméra
# - P : pause / reprise
# - R : réinitialiser la scène
# - Ctrl gauche : passer au mode contrôle de la cible
# - Molette : zoom quand Ctrl est maintenu

import pygame
import copy
import math

# pygame setup
pygame.init()
font = pygame.font.Font('freesansbold.ttf', 18)
font2 = pygame.font.Font('freesansbold.ttf', 30)
WIDTH = 2560
HEIGHT = 1440
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True
time = 0
keyQ = 0
keyE = 0
keyP = 0
keyC = 0
keyCtrl = 0
keyR = 0
Reset = [0,0]
dis = 100
mouse_scroll = 0
PAUSE = [1,2]
keyWASDShSp = [0,0,0,0,0,0]
pygame.mouse.set_visible(0)

cam_pos = [0,10,0]
FOVX = 90/180*math.pi 
FOVY = FOVX*9/16
lookazi = 0
lookele = 0
pygame.mouse.set_pos(WIDTH/2,HEIGHT/2)



# Structure simple pour chaque objet de la scène.
class BODY():
    ty = None
    p = None
    v = None
    vertex = None
    fragment = None
    connection = None
    vmax = None
    trail = None
    prevR = []


def matrix_rotation(p,th):
    xp = p[0]*math.cos(th) + p[1]*math.sin(th)
    yp = -p[0]*math.sin(th) + p[1]*math.cos(th)
    return [xp,yp]

def dcalc(p):
    add = 0
    for i in range(len(p)):
        add += p[i]**2
    return math.sqrt(add)

def norm(p):
    s = 0
    for i in range(len(p)):
        s += p[i]**2

    return s**0.5

def vecdot(a,b):
    s = 0
    for i in range(len(a)):
        s += a[i]*b[i]
    return s

def crossprod(a,b):
    acb = []
    acb.append(a[1]*b[2] - a[2]*b[1])
    acb.append(a[2]*b[0] - a[0]*b[2])
    acb.append(a[0]*b[1] - a[1]*b[0])
    return acb

def make_vec_unit(a):
    b = []
    norma = norm(a)
    for i in range(len(a)):
        b.append(a[i]/norma)
    return b

def clamp(a,mini,maxi):
    b = max(min(a,maxi),mini)
    return b

# Met à jour l'orientation de la caméra avec la souris.
def get_angle(q):
    x,y = pygame.mouse.get_pos()
    rx,ry = x - WIDTH/2, y - HEIGHT/2
    global lookazi,lookele
    if q == 0:
        lookazi = lookazi + rx/170
        lookele = lookele - ry/220
        if lookele > math.pi/2:
            lookele = math.pi/2
        if lookele < -math.pi/2:
            lookele = -math.pi/2
        pygame.mouse.set_pos(WIDTH/2,HEIGHT/2)

def cam_move(e):
    multi = e*20
    if multi == 0:
        multi = 1

    global cam_pos
    lp = [0,0,0]
    if keyWASDShSp[0]:
        lp = [0,0,0]
        lp[2],lp[1] = matrix_rotation([1,0],-lookele)
        lp[2],lp[0] = matrix_rotation([lp[2],lp[0]],-lookazi)
        cam_pos[0] += lp[0]*multi
        cam_pos[1] += lp[1]*multi
        cam_pos[2] += lp[2]*multi
    if keyWASDShSp[2]:
        lp = [0,0,0]
        lp[2],lp[1] = matrix_rotation([-1,0],-lookele)
        lp[2],lp[0] = matrix_rotation([lp[2],lp[0]],-lookazi)
        cam_pos[0] += lp[0]*multi
        cam_pos[1] += lp[1]*multi
        cam_pos[2] += lp[2]*multi
    if keyWASDShSp[1]:
        lp = [0,0,0]
        lp[2],lp[0] = matrix_rotation([0,-1],-lookazi)
        cam_pos[0] += lp[0]*multi
        cam_pos[1] += lp[1]*multi
        cam_pos[2] += lp[2]*multi
    if keyWASDShSp[3]:
        lp = [0,0,0]
        lp[2],lp[0] = matrix_rotation([0,1],-lookazi)
        cam_pos[0] += lp[0]*multi
        cam_pos[1] += lp[1]*multi
        cam_pos[2] += lp[2]*multi
    if keyWASDShSp[4]:
        cam_pos[1] -= 1*multi
    if keyWASDShSp[5]:
        cam_pos[1] += 1*multi

def pauser(keyP,PAUSE):
    if (keyP == 1) and (PAUSE[1] == 0):
        PAUSE = [1,1]
    if ((keyP == 0) and (PAUSE[1] == 1)):
        PAUSE[1] = 2
    if (keyP == 1) and (PAUSE[1] == 2):
        PAUSE = [0,3]
    if ((keyP == 0) and (PAUSE[1] == 3)):
        PAUSE[1] = 0
    return PAUSE




def local_to_screen(lp):
    if(lp[2] <= 0):
        return None
    sp = [(lp[0]/lp[2]/math.tan(FOVX/2) + 1)*WIDTH/2,(-lp[1]/lp[2]/math.tan(FOVY/2) + 1)*HEIGHT/2]
    return sp

def point_tranform(p,off_pos):
    lp = [p[0] - cam_pos[0] + off_pos[0],p[1] - cam_pos[1] + off_pos[1],p[2] - cam_pos[2] + off_pos[2]]

    lp[2],lp[0] = matrix_rotation([lp[2],lp[0]],lookazi)
    lp[2],lp[1] = matrix_rotation([lp[2],lp[1]],lookele)

    sp = local_to_screen(lp)
    return sp




# Fait avancer un objet selon sa vitesse.
def update_body(a):
    a.trail.append(copy.copy(a.p))
    a.p[0] += a.v[0]
    a.p[1] += a.v[1]
    a.p[2] += a.v[2]


def render_body(a):
    a.fragment.clear()
    for i in range(len(a.vertex)):
        a.fragment.append(point_tranform(a.vertex[i],a.p))
        if a.fragment[i] != None:
            pygame.draw.line(screen,"white",a.fragment[i],a.fragment[i])

    for i in range(len(a.connection)):
        if (a.fragment[a.connection[i][0]] != None) and (a.fragment[a.connection[i][1]] != None):
            pygame.draw.line(screen,"white",a.fragment[a.connection[i][0]],a.fragment[a.connection[i][1]])


def render_trail(a,time):
    for i in range(time):
        res = point_tranform(a.trail[i],[0,0,0])
        if res != None:
            
            if a.ty == 0:
                pygame.draw.line(screen,(255,0,0),res,res)
            if a.ty == 1:
                pygame.draw.line(screen,(0,0,255),res,res)
            if a.ty == 2:
                pygame.draw.line(screen,(0,255,255),res,res)

def render_direction(a):
    if dcalc(a.v) == 0:
        return
    res = point_tranform([200*a.v[0]/dcalc(a.v),200*a.v[1]/dcalc(a.v),200*a.v[2]/dcalc(a.v)],a.p)
    res2 = point_tranform([0,0,0],a.p)
    if (res != None) and (res2 != None):
        pygame.draw.line(screen,(255,255,0),res,res2)


def render_range(a,b):
    res = point_tranform(a.p,[0,0,0])
    if res != None:
        res2 = point_tranform(b.p,[0,0,0])
        if res2 != None:
            pygame.draw.line(screen,(0,255,0),res,res2)
            mid = [(res[0]+res2[0])/2, (res[1]+res2[1])/2]
            ran = dcalc([a.p[0]-b.p[0],a.p[1]-b.p[1],a.p[2]-b.p[2]])
            text = font.render(str(math.ceil(ran)), True, (0,255,0))
            screen.blit(text, [mid[0]-10,mid[1]-20,30,30])


#EQUINAV: Loi de guidage développé par ma propre personne, n'a pas d'équivalent réel.
def equinav3d(a,b,time,modifier):
    VP = 3 - time/100*0.04
    if VP < 0:
        VP = 0

    R = [b.p[0] - a.p[0],b.p[1] - a.p[1],b.p[2] - a.p[2]]
    if modifier == 0:
        thalpha = math.atan2(R[0],R[2])
        thbeta = math.atan2(R[1],dcalc([R[0],R[2]]))

        vbR = copy.copy(b.v)
        vbR[2],vbR[0] = matrix_rotation([vbR[2],vbR[0]],thalpha)
        vbR[2],vbR[1] = matrix_rotation([vbR[2],vbR[1]],thbeta)

        if (VP**2 - vbR[0]**2 - vbR[1]**2) < 0:
            a.v = [0,0,0]
            return
        
        vaRp = [vbR[0], vbR[1], math.sqrt(VP**2 - vbR[0]**2 - vbR[1]**2)]
        a.v = [0,0,0]
        a.v[2],a.v[1] = copy.copy(matrix_rotation([vaRp[2],vaRp[1]],-thbeta))
        a.v[2],a.v[0] = copy.copy(matrix_rotation([vaRp[2],vaRp[0]],-thalpha))
    
    else:
        
        thgamma = modifier/180*math.pi  # with modifier in degrees

        thalpha = math.atan2(R[0],R[2])
        thbeta = math.atan2(R[1],dcalc([R[0],R[2]]))
        thphi = math.atan2(R[0],R[1])

        vbR = copy.copy(b.v)
        vbR[2],vbR[0] = matrix_rotation([vbR[2],vbR[0]],thalpha)
        vbR[2],vbR[1] = matrix_rotation([vbR[2],vbR[1]],thbeta)
        vbR[0],vbR[1] = matrix_rotation([vbR[0],vbR[1]],thphi)

        vaR = copy.copy(a.v)
        vaR[2],vaR[0] = matrix_rotation([vaR[2],vaR[0]],thalpha)
        vaR[2],vaR[1] = matrix_rotation([vaR[2],vaR[1]],thbeta)
        vaR[0],vaR[1] = matrix_rotation([vaR[0],vaR[1]],thphi)

        if (VP**2 - vbR[0]**2 - vbR[1]**2) < 0:
            a.v = [0,0,0]   
            return  # use return a.v to conserve heading if pursuer speed too low
        
        vaRp = [vbR[0], vbR[1], math.sqrt(VP**2 - vbR[0]**2 - vbR[1]**2)]
        
        dvaR = [vaRp[0] - vaR[0],vaRp[1] - vaR[1],vaRp[2] - vaR[2]]
        aaR = copy.copy(dvaR)
        aaR[2] = 0

        thlambda = math.atan2(aaR[1],aaR[0])
        
        vaRp2 = copy.copy(vaRp)
        vaRp2[0],vaRp2[1] = matrix_rotation([vaRp2[0],vaRp2[1]],thlambda)

        thgamma = min(math.asin(dcalc(dvaR)/2/VP)*4,thgamma)
        vaRp3 = [vaRp2[2]*math.sin(thgamma), vaRp2[1], vaRp2[2]*math.cos(thgamma)]


        vaRp3[0],vaRp3[1] = matrix_rotation([vaRp3[0],vaRp3[1]],-thlambda)
        
        vaRp3[0],vaRp3[1] = matrix_rotation([vaRp3[0],vaRp3[1]],-thphi)
        vaRp3[2],vaRp3[1] = matrix_rotation([vaRp3[2],vaRp3[1]],-thbeta)
        vaRp3[2],vaRp3[0] = matrix_rotation([vaRp3[2],vaRp3[0]],-thalpha)

        aaRx = [aaR[0],0,0]
        aaRy = [0,aaR[1],0]
        aaRx[0],aaRx[1] = matrix_rotation([aaRx[0],aaRx[1]],-thphi)
        aaRx[2],aaRx[1] = matrix_rotation([aaRx[2],aaRx[1]],-thbeta)
        aaRx[2],aaRx[0] = matrix_rotation([aaRx[2],aaRx[0]],-thalpha)

        aaRy[0],aaRy[1] = matrix_rotation([aaRy[0],aaRy[1]],-thphi)
        aaRy[2],aaRy[1] = matrix_rotation([aaRy[2],aaRy[1]],-thbeta)
        aaRy[2],aaRy[0] = matrix_rotation([aaRy[2],aaRy[0]],-thalpha)


        a.v = copy.copy(vaRp3)

        return aaRx,aaRy



# Loi de guidage par navigation proportionnelle.
def purepronav(a,b,N,dim,time,align):
    VP = 4 #- time/100*0.04
    
    R = []
    for i in range(dim):
        R.append(b.p[i] - a.p[i])

    if len(a.prevR) == 0:
        for i in range(dim):
            a.prevR.append(R[i])
            

    u0 = []
    u1 = []
    du = []
    for i in range(dim):
        u0.append(R[i]/norm(R))
    for i in range(dim):
        u1.append(a.prevR[i]/norm(a.prevR))
    for i in range(dim):
        du.append(u0[i] - u1[i])

    alos = []
    Vc = -(norm(R) - norm(a.prevR))
    
    if Vc > (VP*math.sin(align/180*math.pi)):
        print(a.ty,"pronav")
        for i in range(dim):
            alos.append(N * Vc * (du[i] - u0[i] * vecdot(u0,du)))
        
        normalos = norm(alos)
        for i in range(dim):
            if alos[i] != 0:
                alos[i] = alos[i]
    else:
        print(a.ty,"heading align")
        alos = make_vec_unit(u0)








    alat = []
    
    vhat = make_vec_unit(a.v)

    for i in range(dim):
        alat.append((alos[i] - vhat[i]*vecdot(alos,vhat)))
    if Vc > (VP*math.sin(align/180*math.pi)):
        alathat = make_vec_unit(alat)
        alat = alathat
    for i in range(dim):
        alat[i] = clamp(alat[i],-0.04,0.04)
        

    tmp = BODY()
    tmp.p = a.p
    tmp.v = alos

    tmp2 = BODY()
    tmp2.p = a.p
    tmp2.v = alat
    
    for i in range(dim):
        a.v[i] += alat[i]

    vhat = make_vec_unit(a.v)
    for i in range(dim):
        a.v[i] = vhat[i]*VP
    a.prevR = copy.copy(R)

    return tmp,tmp2







g = BODY()
g.ty = 2
g.p = [0,0,0]
g.fragment = []
g.vertex = []
gx = 20
gy = 20
for i in range(int(-gx/2),int(gx/2)):
    for j in range(int(-gy/2),int(gy/2)):
        g.vertex.append([i*1000,0,j*1000])

g.connection = []
for i in range(gx-1):
    for j in range(int(gy-1)):
        g.connection.append([i*gy+j,i*gy+j+1])
        g.connection.append([i*gy+j,(i+1)*gy+j])


m = BODY()
m.ty = 1
m.p = [0,10,0]
m.v = [0,0,-3]
m.trail = []
m.fragment = []
m.vertex = [
    [-5,-5,-5],
    [5,-5,-5],
    [-5,-5,5],
    [5,-5,5],

    [-5,5,-5],
    [5,5,-5],
    [-5,5,5],
    [5,5,5],
    
]

m.connection = [
    [0,1],
    [0,2],
    [1,3],
    [2,3],

    [4,5],
    [4,6],
    [5,7],
    [6,7],

    [0,4],
    [1,5],
    [2,6],
    [3,7],
]


m2 = BODY()
m2.ty = 2
m2.p = [0,10,0]
m2.v = [0,0,-3]
m2.trail = []
m2.fragment = []
m2.vertex = [
    [-5,-5,-5],
    [5,-5,-5],
    [-5,-5,5],
    [5,-5,5],

    [-5,5,-5],
    [5,5,-5],
    [-5,5,5],
    [5,5,5],
    
]

m2.connection = [
    [0,1],
    [0,2],
    [1,3],
    [2,3],

    [4,5],
    [4,6],
    [5,7],
    [6,7],

    [0,4],
    [1,5],
    [2,6],
    [3,7],
]

t = BODY()
t.ty = 0
t.p = [0,1000,4000]
t.v = [30/100,-10/100,40/100]
t.trail = []
t.fragment = []
t.vertex = [
    [-5,-5,-5],
    [5,-5,-5],
    [-5,-5,5],
    [5,-5,5],

    [-5,5,-5],
    [5,5,-5],
    [-5,5,5],
    [5,5,5],
    
]

t.connection = [
    [0,1],
    [0,2],
    [1,3],
    [2,3],

    [4,5],
    [4,6],
    [5,7],
    [6,7],

    [0,4],
    [1,5],
    [2,6],
    [3,7],
]

tmp = None
tmp2 = None
# Boucle principale : entrées, calculs puis affichage.
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                keyQ = 1
                pygame.mouse.set_visible(1)
            if event.key == pygame.K_e:
                keyE = 1
            if event.key == pygame.K_p:
                keyP = 1
            if event.key == pygame.K_c:
                keyC = 1
            if event.key == pygame.K_LCTRL:
                keyCtrl = 1
            if event.key == pygame.K_r:
                keyR = 1
                
            if event.key == pygame.K_w:
                keyWASDShSp[0] = 1
            if event.key == pygame.K_a:
                keyWASDShSp[1] = 1
            if event.key == pygame.K_s:
                keyWASDShSp[2] = 1
            if event.key == pygame.K_d:
                keyWASDShSp[3] = 1
            if event.key == pygame.K_LSHIFT:
                keyWASDShSp[4] = 1
            if event.key == pygame.K_SPACE:
                keyWASDShSp[5] = 1
        mouse_scroll = 0
        if event.type == pygame.MOUSEWHEEL:
            mouse_scroll = event.y
        
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_q:
                keyQ = 0
                pygame.mouse.set_visible(0)
                pygame.mouse.set_pos(WIDTH/2,HEIGHT/2)
            if event.key == pygame.K_e:
                keyE = 0
            if event.key == pygame.K_p:
                keyP = 0
            if event.key == pygame.K_c:
                keyC = 0
            if event.key == pygame.K_LCTRL:
                keyCtrl = 0
            if event.key == pygame.K_r:
                keyR = 0
            
            
            if event.key == pygame.K_w:
                keyWASDShSp[0] = 0
            if event.key == pygame.K_a:
                keyWASDShSp[1] = 0
            if event.key == pygame.K_s:
                keyWASDShSp[2] = 0
            if event.key == pygame.K_d:
                keyWASDShSp[3] = 0
            if event.key == pygame.K_LSHIFT:
                keyWASDShSp[4] = 0
            if event.key == pygame.K_SPACE:
                keyWASDShSp[5] = 0


    # Mise à jour de la simulation.
    if keyR == 1:
        if Reset[1] == 0:
            Reset = [1,1]
            keyP = 0
            PAUSE = [1,2]
            time = 0
            m.p = [0,10,0]
            m.v = [0,0,0]
            m.trail = []
            m.fragment = []

            m2.p = [0,10,0]
            m2.v = [0,0,0]
            m2.trail = []
            m2.fragment = []

            t.p = [0,1000,4000]
            t.v = [30/100,-10/100,40/100]
            t.trail = []
            t.fragment = []
    else:
        Reset = [0,0]


    PAUSE = pauser(keyP,PAUSE)
    aaRx = [0,0,0]
    aaRy = [0,0,0]
    if PAUSE[0] == 0:
        update_body(m)
        update_body(m2)
        update_body(t)
        thgamma = 30
        tmp,tmp2 = purepronav(m,t,4,3,time,60)#equinav3d(m,t,time,0)
        tmp,tmp2 = purepronav(m2,t,4,3,time,45)



    
    VT = 150/100
    if keyCtrl == 1:

        if keyWASDShSp[0]:
            thVTazi = math.atan2(t.v[0],t.v[2])
            thVTele = math.atan2(t.v[1],dcalc([t.v[0],t.v[2]]))
            t.v[0] = VT*math.cos(thVTele-0.02)*math.sin(thVTazi)
            t.v[1] = VT*math.sin(thVTele-0.02)
            t.v[2] = VT*math.cos(thVTele-0.02)*math.cos(thVTazi)
        if keyWASDShSp[2]:
            thVTazi = math.atan2(t.v[0],t.v[2])
            thVTele = math.atan2(t.v[1],dcalc([t.v[0],t.v[2]]))
            t.v[0] = VT*math.cos(thVTele+0.02)*math.sin(thVTazi)
            t.v[1] = VT*math.sin(thVTele+0.02)
            t.v[2] = VT*math.cos(thVTele+0.02)*math.cos(thVTazi)
        if keyWASDShSp[1]:
            thVTazi = math.atan2(t.v[0],t.v[2])
            thVTele = math.atan2(t.v[1],dcalc([t.v[0],t.v[2]]))
            t.v[0] = VT*math.cos(thVTele)*math.sin(thVTazi-0.02)
            t.v[1] = VT*math.sin(thVTele)
            t.v[2] = VT*math.cos(thVTele)*math.cos(thVTazi-0.02)
        if keyWASDShSp[3]:
            thVTazi = math.atan2(t.v[0],t.v[2])
            thVTele = math.atan2(t.v[1],dcalc([t.v[0],t.v[2]]))
            t.v[0] = VT*math.cos(thVTele)*math.sin(thVTazi+0.02)
            t.v[1] = VT*math.sin(thVTele)
            t.v[2] = VT*math.cos(thVTele)*math.cos(thVTazi+0.02)



    # Affichage de la scène.
    screen.fill((0,150,250))
    if keyCtrl == 0:
        cam_move(keyE)
    get_angle(keyQ)
    if keyCtrl == 1:
        if mouse_scroll == 1:
            dis /= 1.2
        if mouse_scroll == -1:
            dis *= 1.2
        cam_pos = copy.copy(t.p)
        cam_pos[0] -= dis*math.cos(lookele)*math.sin(lookazi)
        cam_pos[1] -= dis*math.sin(lookele)
        cam_pos[2] -= dis*math.cos(lookele)*math.cos(lookazi)

    render_body(g)

    render_trail(m,time)
    render_trail(m2,time)
    render_trail(t,time)
    render_range(m,t)
    render_range(m2,t)
    render_body(m)
    render_body(m2)
    render_body(t)
    render_direction(m)
    #render_direction(m2)
    render_direction(t)

    temp = BODY()
    temp.p = m2.p
    temp.v = aaRx
    render_direction(temp)
    temp2 = BODY()
    temp2.p = m2.p
    temp2.v = aaRy
    render_direction(temp2)
    if tmp != None:
        render_direction(tmp)
    if tmp2 != None:
        render_direction(tmp2)

    pygame.display.flip()
    if PAUSE[0] == 0:
        time+=1
    clock.tick(120)  # limits FPS to 60

pygame.quit()