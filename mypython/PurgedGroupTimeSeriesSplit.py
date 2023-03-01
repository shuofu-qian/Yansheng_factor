import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=0,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        print('n_groups:{},n_folds:{},n_splits:{}'.format(n_groups,n_folds,n_splits))
        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * (group_test_size-2),
                                  n_groups, group_test_size-2)
        print('group_test_starts:{}'.format(group_test_starts))
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]

    # def __init__(self,
    #              n_splits=5,
    #              *,
    #              max_train_group_size=np.inf,
    #              max_test_group_size=np.inf,
    #              group_gap=0,
    #              verbose=False

    def split_2(self, X, y=None, groups=None, train_test_size_ratio=4):
        """Split dataset into n_splits with following rules:
        1.train_group_size is train_test_size_ratio times test_group_size
        2.if concatenating test_groups in different splits, they are continuous
        3.we will drop the data in the beginning instead of the end to satisfy the max_size requirements"""

        if groups is None: raise ValueError("The 'groups' parameter should not be None")

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size

        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_groups = _num_samples(unique_groups)

        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]

        print('n_groups:{},n_splits:{}'.format(n_groups, n_splits))

        group_unit_size = int(np.ceil((n_groups-group_gap) / (n_splits+train_test_size_ratio)))
        if group_unit_size <= 0: raise ValueError('The nums of groups is too small to do split')

        group_test_size = min(group_unit_size, max_test_group_size)
        group_test_starts = range(n_groups - n_splits*group_test_size, n_groups, group_test_size)
        print('group_test_starts:{}'.format(group_test_starts))

        n_groups_left = n_groups - (group_gap + n_splits*group_test_size + train_test_size_ratio*group_unit_size)
        tmp = n_groups_left+group_unit_size*train_test_size_ratio   #除去测试集和gap之后剩下的组数
        group_train_size = min(tmp, max_train_group_size)
        group_train_starts = range(tmp-group_train_size, tmp-group_train_size+n_splits*group_test_size, group_test_size)
        print('group_train_starts:{}'.format(group_train_starts))


        for i in range(n_splits):
            train_array = []
            test_array = []

            for train_group_idx in unique_groups[group_train_starts[i]:group_train_starts[i]+group_train_size]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(np.concatenate((train_array,train_array_tmp)),axis=None),axis=None)

            for test_group_idx in unique_groups[group_test_starts[i]:group_test_starts[i]+group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(np.concatenate((test_array,test_array_tmp)),axis=None),axis=None)
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]


def transform_groups(groups):
    """Transform a groups list in any types into a numerical type"""

    u = np.unique(groups)
    transform_dict = {}
    for i in range(len(u)):
        transform_dict[u[i]] = i
    for i in range(len(groups)):
        groups[i] = transform_dict[groups[i]]

    return groups


def calculate_ic(y_true,y_pred,groups):
    """Given a sorted groups list and corresponded y_true and y_pred,
    Return a list of unique values in groups and a list of ic between y_true and y_pred
    """
    
    u,ind = np.unique(groups, return_index=True)
    ind = np.append(ind,len(groups))

    ic = []
    for i in range(len(u)):
        tmp_ind = list(range(ind[i],ind[i+1]))
        rank_y_true = y_true[tmp_ind].argsort().argsort()
        rank_y_pred = y_pred[tmp_ind].argsort().argsort()
        ic.append(np.corrcoef(rank_y_true,rank_y_pred)[0][1])

    return u, ic


def plot_ic(y,test_index_list,y_pred_list,groups,ma=1,continuous=False):
    """Plot the ic in each group for each splited test dateset
    if continuous: plot all cv_test_ic in one figure, this requires using cv.split_2() when getting the index_list
    """

    if continuous:
        fig = plt.figure(figsize=(10,6))
        tmp_index = sum(test_index_list,[])
        tmp_y_true = y[tmp_index]
        tmp_y_pred = np.concatenate(y_pred_list)
        tmp_groups = groups[tmp_index]

        u,ic = calculate_ic(tmp_y_true, tmp_y_pred, tmp_groups)
        ma_ic = np.concatenate(([np.nan]*(ma-1),np.convolve(ic,np.ones(ma),'valid')/ma))

        plt.plot(u,ic)
        plt.plot(u,ma_ic,color='red')
        plt.xticks(np.arange(0,len(u),len(u)//4))
        plt.title('IC for: {} ~ {} (MA:{})'.format(tmp_groups[0],tmp_groups[-1],ma))

    else:
        n_splits = len(test_index_list)
        fig = plt.figure(figsize=(20,(n_splits//3+1)*5))

        for i in range(n_splits):
            tmp_index = test_index_list[i]
            tmp_y_true = y[tmp_index]
            tmp_y_pred = y_pred_list[i]
            tmp_groups = groups[tmp_index]

            u,ic = calculate_ic(tmp_y_true, tmp_y_pred, tmp_groups)
            ma_ic = np.concatenate(([np.nan]*(ma-1),np.convolve(ic,np.ones(ma),'valid')/ma))

            ax = fig.add_subplot((n_splits//3+1),3,i+1)
            ax.plot(u,ic)
            ax.plot(u,ma_ic,color='red')
            ax.set(xticks=np.arange(0,len(u),len(u)//4))
            ax.set_title('IC for cv_split:{}: {} ~ {} (MA:{})'.format(i+1,tmp_groups[0],tmp_groups[-1],ma))



# this is code slightly modified from the sklearn docs here:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
def plot_cv_indices(cv, X, y, groups, ax, lw=10, split_method=True):
    """Create a sample plot for indices of a cross-validation object.
    split_method: if True: use cv.split() else: use cv.split_2()
    """
    
    n_splits = cv.n_splits
    cmap_cv = plt.cm.coolwarm

    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)   # inplace
    cmap_data = ListedColormap(jet(seq))

    u = np.unique(groups)
    group = [0] * len(groups)
    transform_dict = {}
    for i in range(len(u)):
        transform_dict[u[i]] = i
    for i in range(len(groups)):
        group[i] = transform_dict[groups[i]]

    if split_method:
        split_result = cv.split(X=X, y=y, groups=groups)
    else:
        split_result = cv.split_2(X=X, y=y, groups=groups)

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(split_result):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Set3)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


def main():
    n_samples = 2000
    n_groups = 50
    assert n_samples % n_groups == 0

    idx = np.linspace(0, n_samples-1, num=n_samples)
    X_train = np.random.random(size=(n_samples, 5))
    y_train = np.random.choice([0, 1], n_samples)
    groups = np.repeat(np.linspace(0, n_groups-1, num=n_groups), n_samples/n_groups)

    fig, ax = plt.subplots()

    cv = PurgedGroupTimeSeriesSplit(
        n_splits=5,
        max_train_group_size=30,
        group_gap=2,
        max_test_group_size=5
    )
    plot_cv_indices(cv, X_train, y_train, groups, ax, lw=20)

if __name__ == '__main__':
    main()