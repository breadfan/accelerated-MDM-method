import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def GenerPoints(a, alpha, num_points, dim):
    return alpha * np.random.randn(num_points, dim) + a

def GetHullandPlot(points, dim):
    hull = ConvexHull(points)
    if dim == 2:
        plt.plot(points[:, 0], points[:, 1], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'b-')
    elif dim == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.plot(points[:, 0], points[:, 1], points[:, 2], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'b-')
    else:
        print('There is no way to plot graph in dim > 3, but hull has found successfully!')
    return hull


class MDM(object):
    __class__ = 'MDM'
    __doc__ = """
        This is an implementation the accelerated Mitchell-Demyanov-Malozemov method for 
        finding nearest to coordinates beginning point.
        Also plots convex-hull and optimal solution in 2- and 3-dimensional cases.
        
        #############################################################
        ######################   ARGUMENTS    ######################
        #############################################################
        _dim: int 
            Dimensionality of the problem, usually dim = 2 or 3 for plotting purposes
            
        _points: np.ndarray of shape (dim, number_of_points)
            Points which form convex hull
            
        is_accelerated: boolean
            Flag for using accelerated either not-accelerated MDM-method
            
        supp_vector: list of floats
            Support vector (i.e. {i \in 0 : dim - 1 | p[i] > 0} )
            
        max_iter: int (default value is 500)
            Maximum iterations we can evaluate on
            
        init_approx_index: int (default is 1)
            Index of element of convex hull, which forms first approximation of v vector.
            It can be changed for lowering iterations sake
            For first approximation we'll just take point from a board of hull - cause it's easy reduced
        #############################################################
        #############################################################
        #############################################################
    """

    def __init__(self, points, hull, dim, accel, max_iter = 500, init_approx_index = 1):
        self._dim = dim
        self._points = points.copy()
        self._hull = hull
        self._A_matrix = points.copy().transpose()
        self.is_accelerated = accel
        self.iterations = None
        self.delta_p = None
        self.p_vector = None
        self.vector_current = None
        self.supp_vector = None
        self.max_iter = max_iter
        self.init_approx_index = init_approx_index


    def solve(self):
        V = 0
        iterations = 0

        delta_p = 1
        p_vector = [0 for i in range(0, len(self._points))]
        supp_vector = []
        t_param_vector =[]

        MIN_set = []
        MAX_set = []
        diff_vector = []                            # for cycles finding
        P_vectors = []                              # matrix of p_vectors
        V_vectors = []
        cycle_constructed = False
        cycle_is_constructing = False
        special_upd_done = False                    #whether special update Wn = W + lambdaV is done
        cycle_current_size = 0                      #we will search actual size of cycle

        vector_current = self._points[self._hull.vertices[self.init_approx_index]].copy()    #need copy() there for non-changing _points
        supp_vector.append(self._hull.vertices[self.init_approx_index])                      # approximation => get vect_0
        p_vector[self._hull.vertices[self.init_approx_index]] = 1                            # working right concat
        #then we need to find vect_{k+1} iteratively

        while delta_p > 0.000001 and iterations < self.max_iter and len(supp_vector) != 0:
            if self.is_accelerated is True and cycle_constructed is True and special_upd_done is False:
                for i in range(cycle_size):  #constructing V as linear combination of D's that we used previously
                    V += -1 * t_param_vector[cycle_start + i - 1] * diff_vector[cycle_start + i -1]
                p_vector = P_vectors[cycle_start - 1]                   #returning to value where cycle had begun
                vector_current = V_vectors[cycle_start - 1]       #returning
                supp_vector = []                        #returning
                for i in range(len(p_vector)):          #returning
                    if p_vector[i] > 0.0000001:
                        supp_vector.append(i)

                lambda_t = -np.dot(vector_current, V) / np.linalg.norm(V) ** 2
                for i in range(cycle_size):
                    if t_param_vector[i] > 0:
                        if lambda_t > (1 - p_vector[MIN_set[i]]) / t_param_vector[i]:
                            lambda_t = (1 - p_vector[MIN_set[i]]) / t_param_vector[i]
                    elif t_param_vector[i] < 0:
                        if lambda_t > -p_vector[MAX_set[i]] / t_param_vector[i]:
                            lambda_t = -p_vector[MAX_set[i]] / t_param_vector[i]
                vector_current += lambda_t * V
                for i in range(cycle_size):
                    p_vector[MAX_set[i]] -= lambda_t * t_param_vector[i]
                    p_vector[MIN_set[i]] += lambda_t * t_param_vector[i]
                special_upd_done = True     #once it's done we're forgiving about that


            mult = np.dot(self._points[supp_vector], vector_current)
            ind_max = np.argmax(mult)           # finding max for indices in supp_vector
            ind_max = supp_vector[ind_max]      # finding max general in our mult product

            mult = np.matmul(vector_current, self._A_matrix)
            ind_min = np.argmin(mult)                                                 # i''_k
            if self.is_accelerated is True and cycle_constructed is False:
                MIN_set.append(ind_min)
                MAX_set.append(ind_max)
            diff = self._points[ind_max] - self._points[ind_min]
            print('\nDifference: ' + str(diff))
            delta_p = np.dot(diff, vector_current)

            if delta_p > 0.000001:                  #if not bigger, then we've found a solution
                print('delta_p: ' + str(delta_p))
                print('p_vector[ind_max] = ' + str(p_vector[ind_max]) + '\nnp.linalg.norm(diff)): '
                      + str(np.linalg.norm(diff)))
                t_param = delta_p /(p_vector[ind_max] * (np.linalg.norm(diff)) ** 2)  # recounting all variables
                if t_param >= 1:
                    t_param = 1

                ###################
                #Block for cycle finding. We must remember that MDM can improve wrong choices for updating vectors
                #It could be labeled as "false cycle construction"
                #for one cycles
                #####################
                if self.is_accelerated is True:          #if using accelerated MDM-method
                    if iterations > 0 and cycle_is_constructing is False:  #constructing cycle(active finding cycle, i mean, active-active)
                        contains = np.where(np.all(diff_vector == diff, axis = 1))[0]    #finds if diff_vector contains diff
                        print('np.where1: ' + str(contains))
                        if len(contains) != 0:      #found first element of cycle
                            cycle_is_constructing = True        #cycle is constructing now
                            cycle_start = contains[len(contains)-1]                     #index of first element of cycle; not changing
                            cycle_size = iterations - cycle_start         #not changing
                            cycle_current_size += 1         #this var for checking if all variables actually are cycle
                    elif cycle_is_constructing is True and cycle_constructed is False:
                        if cycle_current_size < cycle_size:
                            indices_match = np.where(np.all(diff_vector == diff, axis=1))[0]
                            print('np.where2: ' + str(indices_match))
                            if len(indices_match) > 0 and  indices_match[len(indices_match) - 1] == (cycle_start + cycle_current_size):
                                cycle_current_size += 1     #checking if cycle exists
                                if cycle_current_size == cycle_size:
                                    cycle_constructed = True
                                    print('CYCLE HAS FOUND AND CONSTRUCTED SUCCESSFULLY!')
                            else:
                                cycle_is_constructing = False
                                cycle_current_size = 0
                                print('False cycle construction.')
                    P_vectors.append(p_vector.copy())
                    V_vectors.append(vector_current.copy())
                    t_param_vector.append(t_param)
                    diff_vector.append(diff)


                vector_current -= t_param * p_vector[ind_max] * diff
                supp_vector = []                #recounting
                temp1 = t_param * p_vector[ind_max]
                temp2 = (1 - t_param)
                p_vector[ind_min] += temp1
                p_vector[ind_max] *= temp2

                for i in range(len(p_vector)):
                    if p_vector[i] > 0.0000001:
                        supp_vector.append(i)
            print('Vector current: ' + str(vector_current))
            iterations += 1
            print('Iterations: ' + str(iterations))
            print('Supp_vector: ' + str(supp_vector))

        return vector_current




#old 2-dim default example, works fine
points =np.array([[ -73.337555  ,   -4.82192605],
       [   9.36299101,   14.79378288],
       [  33.74875017,   10.02043701],
       [ 133.04981839,   92.18760616],
       [-105.00396348,  -69.46640213],
       [  32.54560694,   43.96449265],
       [ -78.01174375,   61.08025333],
       [  92.03366094,  -51.6208306 ],
       [  17.22114877,   54.92524147],
       [ -87.14266467,  128.5875058 ],
       [ -35.76597696, -161.63324815],
       [ 156.36709765,  -55.60266369],
       [  41.00897625,  -54.92133061],
       [ 129.50005618,  -39.14660553],
       [ 101.99767049,    5.91893179],
       [ 120.62635591,   39.32842524],
       [  58.91037616,  -29.52086718],
       [-116.99548555,  -35.64041842],
       [ -49.26778003,   18.11377985],
       [  91.22017504,   26.95527778],
       [   5.98350205,  -29.65544224],
       [  73.8606758 ,  -67.33527561],
       [ -57.11269196,  -23.38066312],
       [  10.29413585,   19.91249178],
       [ -76.57980277,   36.15112039],
       [  40.91217006,  -17.81387299],
       [  51.88700332,  -69.65988091],
       [  57.41048001, -119.28130887],
       [ -66.49323658,  -92.43371661],
       [  10.46455101,  -80.23934518]])   ##


# points = np.array([[ -75.17801366,    7.83288242,   -5.01730323],
#        [  34.64493481,   17.08160686,   -4.12464737],
#        [  69.25185714,   46.30248975,  -50.02067627],
#        [  38.10853617,  -14.51657311,   16.07997019],
#        [ -11.14528316,  -23.9139584 ,  -48.72752102],
#        [  47.71039686,  -67.59707025,  -71.9969375 ],
#        [  28.54312988, -127.60998881, -114.56196853],
#        [ 205.10854027,    0.32344749,  -45.66628019],
#        [   2.92497051,  -46.89594309,   53.06491769],
#        [  41.97712067, -170.65770544,   18.60333687],
#        [ -51.55472502,   60.64006628,   67.86351622],
#        [  44.59497019,   27.57009689,   95.74464091],
#        [   3.35649724,  -12.96848999,  -67.41464425],
#        [ -47.15574201,   33.67097469,  -48.27886268],
#        [   7.66095838,   59.77078387, -162.02028844],
#        [ -69.59850196,  -23.95314366,  -58.41426815],
#        [  27.61525677,  -47.26875507,   -4.70457163],
#        [ -74.79462539,    7.96213077,   -9.01206742],
#        [ -66.99167694,   51.87158456,  -31.53336444],
#        [-116.95434661,  -95.79658366,  -22.96431562],
#        [  54.39636852,   25.06735714,   30.18068819],
#        [ -91.04888825,   58.20995254,  -24.17908215],
#        [ -12.8309548 ,  -16.78814715,  -41.21037765],
#        [ -78.25185047,   63.99086455,  -47.92050365],
#        [ -79.99177441,  -60.06975075,   29.71067655],
#        [  74.3836634 ,  -28.15199117,  -29.7868583 ],
#        [ -65.77372521,   79.95508642,    0.29089113],
#        [   5.23160557,  151.42554231,   60.96004149],
#        [ -49.1255094 ,   23.05475224,  181.98787966],
#        [ -46.1174773 ,  -31.61406696,   60.10877562]])


dim = 2; number_of_points = 30

is_manual_enter = False
is_accelerated = True

inp = input('Use manual enter or use default parameters? M/D.')
if inp == 'M':
    is_manual_enter = True
if is_manual_enter is True:
    gener = input('Use generator or manual input of points? G/M')
    if gener == 'M':
        print('Enter the data (points values) in the \"data.txt\" file.')
        points = []
        number_of_points = 0
        with open("data.txt") as f:
            for line in f:
                temp = [float(x) for x in line.split()]
                points.append(temp)
                number_of_points += 1
        points = np.array(points)
        dim = len(points[0])
    elif gener == 'G':
        dim = int(input('Enter number of dimensions: '))
        number_of_points = int(input('Enter number of points: '))
        points = GenerPoints(3, 68, number_of_points, dim)

    temp = input('Use classic or accelerated MDM-method? C/A')      #by default we're using accelerated method
    if temp == 'C':
        is_accelerated = False

elif is_manual_enter is False:
    print('Our DEFAULT values: \nNumber of dimensions is ' \
          + str(dim) + '\nNumber of points is ' + str(number_of_points))


hull = GetHullandPlot(points, dim)
mdm = MDM(points, hull, dim, is_accelerated)
result = mdm.solve()                                    #returns a point in R^dim

if dim == 2 :
    plt.plot([result[0], 0], [result[1], 0], 'ro')
elif dim == 3 :
    plt.plot([result[0], 0], [result[1], 0], [result[2], 0], 'ro')
plt.show()
print('Result is: ' + str(result))