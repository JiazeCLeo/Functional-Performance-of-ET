import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import distance

from scipy.stats import pearsonr
### Just to remove warnings to prettify the notebook. 
import warnings
warnings.filterwarnings("ignore")
def clean_data(data):
    data = np.array(data)
    if np.isnan(data).any() or np.isinf(data).any():
        # Calculate the mean of finite numbers only
        finite_mean = np.nanmean(np.where(np.isfinite(data), data, np.nan))
        # Replace inf and -inf with NaN first, then fill NaNs with the calculated mean
        data = np.where(np.isfinite(data), data, np.nan)
        data = np.nan_to_num(data, nan=finite_mean)
    return data
class ccm:
    def rmse(self, obs, pred):
        return np.sqrt(np.mean((np.array(obs) - np.array(pred)) ** 2))

    def nse(self, obs, pred):
        obs = np.array(obs)
        pred = np.array(pred)
        return 1 - (np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2))

    def __init__(self, X, Y, tau=1, E=2, L=500):
        '''
        X: timeseries for variable X that could cause Y
        Y: timeseries for variable Y that could be caused by X
        tau: time lag
        E: shadow manifold embedding dimension
        L: time period/duration to consider (longer = more data)
        We're checking for X -> Y
        '''
        self.X = X
        self.Y = Y
        self.tau = tau
        self.E = E
        self.L = L        
        self.My = self.shadow_manifold(Y)  # shadow manifold for Y (we want to know if info from X is in Y)
        self.t_steps, self.dists = self.get_distances(self.My)  # for distances between points in manifold    

    def shadow_manifold(self, X):
        """
        Given
            X: some time series vector
            tau: lag step
            E: shadow manifold embedding dimension
            L: max time step to consider - 1 (starts from 0)
        Returns
            {t:[t, t-tau, t-2*tau ... t-(E-1)*tau]} = Shadow attractor manifold, dictionary of vectors
        """
        M = {t: [] for t in range((self.E-1) * self.tau, self.L)}  # shadow manifold
        for t in range((self.E-1) * self.tau, self.L):
            x_lag = []  # lagged values
            for t2 in range(0, self.E-1 + 1):  # get lags, we add 1 to E-1 because we want to include E
                x_lag.append(X[t-t2*self.tau])            
            M[t] = x_lag
        return M

    def get_distances(self, Mx):
        """
        Args
            Mx: The shadow manifold from X
        Returns
            t_steps: timesteps
            dists: n x n matrix showing distances of each vector at t_step (rows) from other vectors (columns)
        """
        t_vec = [(k, v) for k, v in Mx.items()]
        t_steps = np.array([i[0] for i in t_vec])
        vecs = np.array([i[1] for i in t_vec])
        dists = distance.cdist(vecs, vecs)    
        return t_steps, dists

    def get_nearest_distances(self, t, t_steps, dists):
        t_ind = np.where(t_steps == t)  # get the index of time t
        if t_ind[0].size == 0:
            raise ValueError(f"No matching timestep found for t={t}")

        dist_t = dists[t_ind].squeeze()  # distances from vector at time t

        # 确保 dist_t 至少是一维数组
        if dist_t.ndim == 0:
            dist_t = np.array([dist_t])

        nearest_inds = np.argsort(dist_t)[1:self.E+1 + 1]  # get indices sorted, exclude the zero distance from itself
        if nearest_inds.size < self.E+1:
            raise ValueError("Not enough points to find nearest neighbors")

        nearest_timesteps = t_steps[nearest_inds]  # index column-wise, t_steps are same column and row-wise 
        nearest_distances = dist_t[nearest_inds]

        return nearest_timesteps, nearest_distances
    

    def predict(self, t):
        """
        Args
            t: timestep at Mx to predict Y at same time step
        Returns
            Y_true: the true value of Y at time t
            Y_hat: the predicted value of Y at time t using Mx
        """
        eps = 0.000001  # epsilon minimum distance possible
        t_ind = np.where(self.t_steps == t)  # get the index of time t
        dist_t = self.dists[t_ind].squeeze()  # distances from vector at time t (this is one row)    
        nearest_timesteps, nearest_distances = self.get_nearest_distances(t, self.t_steps, self.dists)    
        
        u = np.exp(-nearest_distances/np.max([eps, nearest_distances[0]]))  # we divide by the closest distance to scale
        w = u / np.sum(u)
        
        X_true = self.X[t]  # get corresponding true X
        X_cor = np.array(self.X)[nearest_timesteps]  # get corresponding Y to cluster in Mx
        X_hat = (w * X_cor).sum()  # get X_hat
        
        return X_true, X_hat
    
    def causality(self):
        '''
        Args:
            None
        Returns:
            correl: how much self.X causes self.Y. correlation between predicted Y and true Y
        '''

        # run over all timesteps in M
        # X causes Y, we can predict X using My
        # X puts some info into Y that we can use to reverse engineer X from Y        
        X_true_list = []
        X_hat_list = []

        for t in list(self.My.keys()): # for each time step in My
            X_true, X_hat = self.predict(t) # predict X from My
            X_true_list.append(X_true)
            X_hat_list.append(X_hat) 

        x, y = X_true_list, X_hat_list
        x = pd.Series(x)
        y = pd.Series(y)
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        x.fillna(x.mean(), inplace=True)
        y.fillna(y.mean(), inplace=True)
        x = x.tolist()
        y = y.tolist()
        r, p = pearsonr(x, y)        

        return r, p
    
    def plot_ccm_correls(self):
        M = self.shadow_manifold(self.Y)  # shadow manifold
        t_steps, dists = self.get_distances(M)  # for distances

        ccm_XY = ccm(self.X, self.Y, self.tau, self.E, self.L)  # define new ccm object # Testing for X -> Y
        ccm_YX = ccm(self.Y, self.X, self.tau, self.E, self.L)  # define new ccm object # Testing for Y -> X

        X_My_true, X_My_pred = [], []  # note pred X | My is equivalent to figuring out if X -> Y
        Y_Mx_true, Y_Mx_pred = [], []  # note pred Y | Mx is equivalent to figuring out if Y -> X

        for t in range(self.tau, self.L):
            true, pred = ccm_XY.predict(t)
            X_My_true.append(true)
            X_My_pred.append(pred)

            true, pred = ccm_YX.predict(t)
            Y_Mx_true.append(true)
            Y_Mx_pred.append(pred)
        
        rmse_XY = self.rmse(X_My_true, X_My_pred)
        nse_XY = self.nse(X_My_true, X_My_pred)

        # Calculating RMSE and NSE for Y -> X
        rmse_YX = self.rmse(Y_Mx_true, Y_Mx_pred)
        nse_YX = self.nse(Y_Mx_true, Y_Mx_pred)

        # plot
        figs, axs = plt.subplots(1, 2, figsize=(12, 5))
        X_My_true_cleaned = clean_data(X_My_true)
        X_My_pred_cleaned = clean_data(X_My_pred)
        Y_Mx_true_cleaned = clean_data(Y_Mx_true)
        Y_Mx_pred_cleaned = clean_data(Y_Mx_pred)

        # predicting X from My
        r, p = np.round(pearsonr(X_My_true_cleaned, X_My_pred_cleaned), 4)

        axs[0].scatter(X_My_true, X_My_pred, s=10)
        axs[0].set_xlabel('$X(t)$ (observed)', size=12)
        axs[0].set_ylabel('$\hat{X}(t)|M_y$ (estimated)', size=12)
        axs[0].text(0.05, 0.95, f'Correlation={r}\nRMSE={rmse_XY:.4f}\nNSE={nse_XY:.4f}', 
                    transform=axs[0].transAxes, verticalalignment='top')
        axs[0].set_title(f'tau={self.tau}, E={self.E}, L={self.L}')

        # predicting Y from Mx
        r, p = np.round(pearsonr(Y_Mx_true_cleaned, Y_Mx_pred_cleaned), 4)

        axs[1].scatter(Y_Mx_true, Y_Mx_pred, s=10)
        axs[1].set_xlabel('$Y(t)$ (observed)', size=12)
        axs[1].set_ylabel('$\hat{Y}(t)|M_x$ (estimated)', size=12)
        axs[1].text(0.05, 0.95, f'Correlation={r}\nRMSE={rmse_YX:.4f}\nNSE={nse_YX:.4f}', 
                    transform=axs[1].transAxes, verticalalignment='top')

        axs[1].set_title(f'tau={self.tau}, E={self.E}, L={self.L}')
        plt.show()
    
    def visualize_cross_mapping(self):
        import matplotlib.pyplot as plt

        f, axs = plt.subplots(1, 2, figsize=(12, 6))        

        for i, ax in zip((0, 1), axs):  # i will be used in switching Mx and My in Cross Mapping visualization
            # Shadow Manifolds Visualization
            X_lag, Y_lag = [], []
            for t in range(1, len(self.X)):
                X_lag.append(self.X[t-self.tau])
                Y_lag.append(self.Y[t-self.tau])    
            X_t, Y_t = self.X[1:], self.Y[1:]  # remove first value

            ax.scatter(X_t, X_lag, s=5, label='$M_x$')
            ax.scatter(Y_t, Y_lag, s=5, label='$M_y$', color='y')

            # Cross Mapping Visualization
            A, B = [(self.Y, self.X), (self.X, self.Y)][i]
            cm_direction = ['Mx to My', 'My to Mx'][i]

            Ma = self.shadow_manifold(A)
            Mb = self.shadow_manifold(B)

            t_steps_A, dists_A = self.get_distances(Ma)  # for distances between points in manifold
            t_steps_B, dists_B = self.get_distances(Mb)  # for distances between points in manifold

            # Plot cross mapping for different time steps
            timesteps = list(Ma.keys())
            for t in np.random.choice(timesteps, size=3, replace=False):
                Ma_t = Ma[t]
                near_t_A, near_d_A = self.get_nearest_distances(t, t_steps_A, dists_A)

                for i in range(self.E+1):
                    # points on Ma
                    A_t = Ma[near_t_A[i]][0]
                    A_lag = Ma[near_t_A[i]][1]
                    ax.scatter(A_t, A_lag, color='b', marker='s')

                    # corresponding points on Mb
                    B_t = Mb[near_t_A[i]][0]
                    B_lag = Mb[near_t_A[i]][1]
                    ax.scatter(B_t, B_lag, color='r', marker='*', s=50)  

                    # connections
                    ax.plot([A_t, B_t], [A_lag, B_lag], color='r', linestyle=':') 

            ax.set_title(f'{cm_direction} cross mapping. Time lag, tau = {self.tau}, E = {self.E}')
            ax.legend(prop={'size': 14})

            ax.set_xlabel('$X_t$, $Y_t$', size=15)
            ax.set_ylabel('$X_{t-1}$, $Y_{t-1}$', size=15)               
        plt.show()
        


