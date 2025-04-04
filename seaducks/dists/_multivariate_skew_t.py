from scipy.special import gammaln, gamma,hyp2f1
import numpy as np
import matplotlib.pyplot as plt

class Multivariate_skew_t:

    def __init__(self,shape,loc,disp_mat,df):
        self.dim=len(shape)
        self.shape = shape
        self.loc = loc
        self.disp_mat = disp_mat
        self.A = np.transpose(np.linalg.cholesky(self.disp_mat))
        self.df = df

    def logpdf(self,x):
        # values
        detA = np.diag(self.A).prod()
        sqrt_maha_dist = np.matmul(self.A,x-self.loc)
        maha_dist = np.dot(sqrt_maha_dist,sqrt_maha_dist)
        omega_inv = np.divide(1,np.diag(self.disp_mat))
        # constants
        c1 = np.log(2*detA*(np.pi*self.df)**(-self.dim/2))
        c2 = gammaln((self.df + self.dim)/2)
        c3 = -gammaln(self.df/2)

        # y_dependent
        fy1 = -(self.df + self.dim)*np.log(1+maha_dist/self.df)/2
        hy = np.dot(np.matmul(omega_inv,x-self.loc),self.shape)*np.sqrt(
            (self.dim+self.df)/(maha_dist+self.df)
        )
        hyp_geom = hyp2f1(1/2,(self.df+self.dim+1)/2,3/2,-hy**2/(self.dim+self.df))

        # t CDF multipliers
        m1 = gamma((
                self.df + self.dim + 1
                )/2)/(
                    np.sqrt(
                        (np.pi*(self.df + self.dim))
                    )*gamma((self.df + self.dim)/2)
                    )
        
        return c1+c2+c3+fy1+np.log(1/2 + hy*m1*hyp_geom)
        
    def pdf(self,x):
        return np.exp(self.logpdf(x))

shape = np.array([-50,200])
loc = np.array([1,0])
disp_mat = np.array([[1,0],[0,1]])
df = 1

test_t = Multivariate_skew_t(shape,loc,disp_mat,df)

# Create grid and multivariate normal
x = np.linspace(-1, 4, 200) # Create a mesh of 200 x-points
y = np.linspace(-2, 2, 200) # Create a mesh of 200 y-points


z = np.zeros((200,200))
for ii in range(200):
    for jj in range(200):
        z[ii,jj] = np.linalg.norm(np.exp(test_t.logpdf(np.array([x[ii],y[jj]]))))
X, Y = np.meshgrid(x, y)
print(z)
fig, ax = plt.subplots(figsize=(10, 8),subplot_kw={"projection": "3d"})
cs = ax.contour(X,Y,z, antialiased=False,levels=200)
plt.show()