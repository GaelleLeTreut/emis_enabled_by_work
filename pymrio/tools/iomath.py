""" Mathematical functions for input output calculations

All methods here should follow the functional programming paradigm

Note
----
To avoid namespace pollution everythin here starts with calc_

"""

import warnings

import numpy as np
import pandas as pd

import pymrio.tools.ioutil as ioutil


def calc_x(Z, Y):
    """Calculate the industry output x from the Z and Y matrix

    industry output (x) = flows (sum_columns(Z)) + final demand (sum_columns(Y))

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric input output table (flows)
    Y : pandas.DataFrame or numpy.array
        final demand with categories (1.order) for each country (2.order)

    Returns
    -------
    pandas.DataFrame or numpy.array
        Industry output x as column vector
        The type is determined by the type of Z. If DataFrame index as Z

    """
    x = np.reshape(np.sum(np.hstack((Z, Y)), 1), (-1, 1))
    if type(Z) is pd.DataFrame:
        x = pd.DataFrame(x, index=Z.index, columns=["indout"])
    if type(x) is pd.Series:
        x = pd.DataFrame(x)
    if type(x) is pd.DataFrame:
        x.columns = ["indout"]
    return x


def calc_x_from_L(L, y):
    """Calculate the industry output x from L and a y vector

    x = Ly

    The industry output x is computed from a demand vector y

    Parameters
    ----------
    L : pandas.DataFrame or numpy.array
        Symmetric input output Leontief table
    y : pandas.DataFrame or numpy.array
        a column vector of the total final demand

    Returns
    -------
    pandas.DataFrame or numpy.array
        Industry output x as column vector
        The type is determined by the type of L. If DataFrame index as L

    """
    x = L.dot(y)
    if type(x) is pd.Series:
        x = pd.DataFrame(x)
    if type(x) is pd.DataFrame:
        x.columns = ["indout"]
    return x


def calc_Z(A, x):
    """calculate the Z matrix (flows) from A and x

    A = Z / x[None, :]  =>  Z = A * x[None, :]

    By definition, the coefficient matrix A is basically the normalized flows
    So Z is just derived from A by un-normalizing using the industrial output x

    Parameters
    ----------
    A : pandas.DataFrame or numpy.array
        Symmetric input output table (coefficients)
    x : pandas.DataFrame or numpy.array
        Industry output column vector

    Returns
    -------
    pandas.DataFrame or numpy.array
        Symmetric input output table (flows) Z
        The type is determined by the type of A.
        If DataFrame index/columns as A

    """
    if (type(x) is pd.DataFrame) or (type(x) is pd.Series):
        x = x.values
    x = x.reshape((1, -1))  # use numpy broadcasting - much faster
    # (but has to ensure that x is a row vector)
    # old mathematical form:
    # return A.dot(np.diagflat(x))
    if type(A) is pd.DataFrame:
        return pd.DataFrame(A.values * x, index=A.index, columns=A.columns)
    else:
        return A * x


def calc_A(Z, x):
    """Calculate the A matrix (coefficients) from Z and x

    A is a normalized version of the industrial flows Z

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric input output table (flows)
    x : pandas.DataFrame or numpy.array
        Industry output column vector

    Returns
    -------
    pandas.DataFrame or numpy.array
        Symmetric input output table (coefficients) A
        The type is determined by the type of Z.
        If DataFrame index/columns as Z

    """
    if (type(x) is pd.DataFrame) or (type(x) is pd.Series):
        x = x.values
    if (type(x) is not np.ndarray) and (x == 0):
        recix = 0
    else:
        with warnings.catch_warnings():
            # catch the divide by zero warning
            # we deal wit that by setting to 0 afterwards
            warnings.simplefilter("ignore")
            recix = 1 / x
        recix[recix == np.inf] = 0
        recix = recix.reshape((1, -1))
    # use numpy broadcasting - factor ten faster
    # Mathematical form - slow
    # return Z.dot(np.diagflat(recix))
    if type(Z) is pd.DataFrame:
        return pd.DataFrame(Z.values * recix, index=Z.index, columns=Z.columns)
    else:
        return Z * recix


def calc_L(A):
    """Calculate the Leontief L from A

    L = inverse matrix of (I - A)

    Where I is an identity matrix of same shape as A

    Comes from:
        x = Ax + y  =>  (I-A)x = y
    Where:
        A: coefficient input () - output () table
        x: output vector
        y: final demand vector

    Hence, L allows to derive a required output vector x for a given demand y

    Parameters
    ----------
    A : pandas.DataFrame or numpy.array
        Symmetric input output table (coefficients)

    Returns
    -------
    pandas.DataFrame or numpy.array
        Leontief input output table L
        The type is determined by the type of A.
        If DataFrame index/columns as A

    """
    I = np.eye(A.shape[0])  # noqa
    if type(A) is pd.DataFrame:
        return pd.DataFrame(np.linalg.inv(I - A), index=A.index, columns=A.columns)
    else:
        return np.linalg.inv(I - A)


def calc_S(F, x):
    """Calculate extensions/factor inputs coefficients

    Parameters
    ----------
    F : pandas.DataFrame or numpy.array
        Total direct impacts
    x : pandas.DataFrame or numpy.array
        Industry output column vector

    Returns
    -------
    pandas.DataFrame or numpy.array
        Direct impact coefficients S
        The type is determined by the type of F.
        If DataFrame index/columns as F

    """
    return calc_A(F, x)


def calc_S_Y(F_Y, y):
    """Calculate extensions/factor inputs coefficients for the final demand

    Parameters
    ----------
    F_Y : pandas.DataFrame or numpy.array
         Final demand impacts
    y : pandas.DataFrame or numpy.array
        Final demand column vector (e.g. mrio.Y.sum())

    Returns
    -------
    pandas.DataFrame or numpy.array
        Direct impact coefficients of final demand S_Y
        The type is determined by the type of F.
        If DataFrame index/columns as F

    """
    return calc_A(F_Y, y)


def calc_F(S, x):
    """Calculate total direct impacts from the impact coefficients

    Parameters
    ----------
    S : pandas.DataFrame or numpy.array
        Direct impact coefficients S
    x : pandas.DataFrame or numpy.array
        Industry output column vector

    Returns
    -------
    pandas.DataFrame or numpy.array
        Total direct impacts F
        The type is determined by the type of S.
        If DataFrame index/columns as S

    """
    return calc_Z(S, x)


def calc_F_Y(S_Y, y):
    """Calc. total direct impacts from the impact coefficients of final demand

    Parameters
    ----------
    S_Y : pandas.DataFrame or numpy.array
        Direct impact coefficients of final demand
    y : pandas.DataFrame or numpy.array
        Final demand column vector (e.g. mrio.Y.sum())

    Returns
    -------
    pandas.DataFrame or numpy.array
        Total direct impacts of final demand F_Y
        The type is determined by the type of S_Y.
        If DataFrame index/columns as S_Y

    """
    return calc_Z(S_Y, y)


def calc_M(S, L):
    """Calculate multipliers of the extensions

    Parameters
    ----------
    L : pandas.DataFrame or numpy.array
        Leontief input output table L
    S : pandas.DataFrame or numpy.array
        Direct impact coefficients

    Returns
    -------
    pandas.DataFrame or numpy.array
        Multipliers M
        The type is determined by the type of D.
        If DataFrame index/columns as D

    """
    return S.dot(L)


def calc_e(M, Y):
    """Calculate total impacts (footprints of consumption Y)

    Parameters
    ----------
    M : pandas.DataFrame or numpy.array
        Multipliers
    Y : pandas.DataFrame or numpy.array
        Final consumption

    TODO - this must be completely redone (D, check for dataframe, ...)

    Returns
    -------
    pandas.DataFrame or numpy.array
        Multipliers m
        The type is determined by the type of M.
        If DataFrame index/columns as M
    The calcubased on multipliers M and finald demand Y"""

    return M.dot(Y)


def recalc_M(S, D_cba, Y, nr_sectors):
    """Calculate Multipliers based on footprints.

    Parameters
    ----------
    D_cba : pandas.DataFrame or numpy array
        Footprint per sector and country
    Y : pandas.DataFrame or numpy array
        Final demand: aggregated across categories or just one category, one
        column per country. This will be diagonalized per country block.
        The diagonolized form must be invertable for this method to work.
    nr_sectors : int
        Number of sectors in the MRIO

    Returns
    -------

    pandas.DataFrame or numpy.array
        Multipliers M
        The type is determined by the type of D_cba.
        If DataFrame index/columns as D_cba


    """

    Y_diag = ioutil.diagonalize_blocks(Y.values, blocksize=nr_sectors)
    Y_inv = np.linalg.inv(Y_diag)
    M = D_cba.dot(Y_inv)
    if type(D_cba) is pd.DataFrame:
        M.columns = D_cba.columns
        M.index = D_cba.index

    return M


def calc_accounts(S, L, Y, nr_sectors, G=None, value_added=None):
    """Calculate sector specific cba and pba based accounts, imp and exp accounts

    The total industry output x for the calculation
    is recalculated from L and y

    Parameters
    ----------
    L : pandas.DataFrame
        Leontief input output table L
    S : pandas.DataFrame
        Direct impact coefficients
    Y : pandas.DataFrame
        Final demand: aggregated across categories or just one category, one
        column per country
    nr_sectors : int
        Number of sectors in the MRIO


    Returns
    -------
    Tuple
        (D_cba, D_pba, D_imp, D_exp, D_iba)

        Format: D_row x L_col (=nr_countries*nr_sectors)

        - D_cba        Footprint per sector and country
        - D_pba      Total factur use per sector and country
        - D_imp       Total global factor use to satisfy total final demand in
                      the country per sector
        - D_exp       Total factor use in one country to satisfy final demand
                      in all other countries (per sector)
        - D_iba        Income-based footprints per sector and country
    """
    # diagonalize each sector block per country
    # this results in a disaggregated y with final demand per country per
    # sector in one column
    Y_diag = ioutil.diagonalize_blocks(Y.values, blocksize=nr_sectors)
    x_diag = L.dot(Y_diag)
    x_tot = x_diag.values.sum(1)
    del Y_diag

    D_cba = pd.DataFrame(S.values.dot(x_diag), index=S.index, columns=S.columns)
    # D_pba = S.dot(np.diagflat(x_tot))
    # faster broadcasted calculation:
    D_pba = pd.DataFrame(
        S.values * x_tot.reshape((1, -1)), index=S.index, columns=S.columns
    )

    # for the traded accounts set the domestic industry output to zero
    dom_block = np.zeros((nr_sectors, nr_sectors))
    x_trade = ioutil.set_block(x_diag.values, dom_block)
    D_imp = pd.DataFrame(S.values.dot(x_trade), index=S.index, columns=S.columns)

    x_exp = x_trade.sum(1)
    # D_exp = S.dot(np.diagflat(x_exp))
    # faster broadcasted version:
    D_exp = pd.DataFrame(
        S.values * x_exp.reshape((1, -1)), index=S.index, columns=S.columns
    )
        
    # Income-based emissions
    #Q1 = comment sait-on que D_cba -and others- sont dans io(self).satellite et non dans io? = > c'est la fonction calc_extension qui fait ca : elle calcule D_cba and co pour la liste d'extension qui par défaut n'est que 'satellite
    #Q2 = comment mettre une condition sur la présence de value_added et de G dans l'extension ?
    if (
            (value_added is not None)
            and (G is not None)
        ):
            S = S.values
            S = S.reshape(-1,1)
            S = np.transpose(S)
            iec = S.dot(G)       
            D_iba = pd.DataFrame(iec.dot(np.transpose(value_added)), index = value_added.columns, columns= value_added.index )
            del iec
            return (D_cba, D_pba, D_imp, D_exp, np.transpose(D_iba))


    return (D_cba, D_pba, D_imp, D_exp)

def calc_B(Z, x):
    """Calculate the B matrix (Ghosh) from Z and x

    B is a normalized version of the industrial flows Z

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric input output table (flows)
    x : pandas.DataFrame or numpy.array
        Industry output column vector

    Returns
    -------
    pandas.DataFrame or numpy.array
        Symmetric input output table (coefficients) B
        The type is determined by the type of Z.
        If DataFrame index/columns as Z

    """
    if type(Z) is pd.DataFrame:
        return calc_A(Z.transpose(), x)
    else:
        return calc_A(np.transpose(Z), x)


def calc_G(B):
    """Calculate the Ghosh G from B

    G = inverse matrix of (I - B)

    Where I is an identity matrix of same shape as B

    Comes from:
        x = Bx + v  =>  (I-B)x = v
    Where:
        B: Ghosh coefficient input () - output () table
        x: output vector
        v: value added vector

    Hence, G allows to derive a required output vector x for a given value added v

    Parameters
    ----------
    B : pandas.DataFrame or numpy.array
        Symmetric input output table (coefficients)

    Returns
    -------
    pandas.DataFrame or numpy.array
        Ghosh input output table G
        The type is determined by the type of B.
        If DataFrame index/columns as B

    """
    return calc_L(B)

def calc_Z_from_B(B, x):
    """calculate the Z matrix (flows) from B and x

    Z = x[None, :] * transpose(B)

    Parameters
    ----------
    B : pandas.DataFrame or numpy.array
        Symmetric input output table (coefficients)
    x : pandas.DataFrame or numpy.array
        Industry output column vector

    Returns
    -------
    pandas.DataFrame or numpy.array
        Symmetric input output table (flows) Z
        The type is determined by the type of B.
        If DataFrame index/columns as B

    """
    if (type(x) is pd.DataFrame) or (type(x) is pd.Series):
        x = x.values
    #x = x.reshape((1, -1))  # use numpy broadcasting - much faster
    # (but has to ensure that x is a column vector)
    # old mathematical form:
    # return A.dot(np.diagflat(x))
    if type(B) is pd.DataFrame:
        return pd.DataFrame(x * np.transpose(B.values), index=B.index, columns=B.columns)
    else:
        return x * np.transpose(B)
        
def calc_iec(S, G):
    """Calculate the income emission content of the VA production factor from S and G

    iec = S * G

    Parameters
    ----------
    G : pandas.DataFrame or numpy.array
        Ghosh matrix, dimension input output table (coefficient)
    S : pandas.DataFrame or numpy.array
        One type of direct impact coefficients S 
        (nb sectors, 1) dimension (eg combustion)
        #-> need to be improved

    Returns
    -------
    pandas.DataFrame or numpy.array
        Row vector Symmetric input output table (coefficients) iec
        The type is determined by the type of G.
        If DataFrame index/columns as G

    """
    if (type(S) is pd.DataFrame) or (type(S) is pd.Series):
        S = S.values
        S = S.reshape(-1,1)
        S = np.transpose(S)
    if type(G) is pd.DataFrame:
        return pd.DataFrame(S.dot(G), index= ['emission content'], columns=G.columns)
    else:
        return S.dot(G)

def calc_ibe(iec, value_added):
    """Calculate the income based emissions of VA from emission content and VA factors

    ibe = iec * VA

    Parameters
    ----------
    iec : pandas.DataFrame or numpy.array
        Emission content matrix, dimension input output table
    VA : pandas.DataFrame or numpy.array
        Value added block

    Returns
    -------
    pandas.DataFrame or numpy.array
        Income based emissions by VA factors
        The type is determined by the type of Value Added.
        If DataFrame index/columns as value_added
    """

    if (type(iec) is pd.DataFrame) or (type(iec) is pd.Series):
        iec = iec.values
        iec = np.diag(iec[0])
    if type(value_added) is pd.DataFrame:
        ibe = pd.DataFrame( iec.dot(np.transpose(value_added)), index = value_added.columns, columns= value_added.index )
        return np.transpose(ibe)
    else:
        return np.transpose(iec.dot(np.transpose(value_added)))
