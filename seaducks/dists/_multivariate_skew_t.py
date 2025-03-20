from scipy.special import gammaln, gamma,hyp2f1
import numpy as np

class Multivariate_skew_t:

    def __init__(self,shape,loc,disp_mat,df):
        self.dim=len(shape)
        self.shape = shape
        self.loc = loc
        self.disp_mat = disp_mat
        self.A = None
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
            

