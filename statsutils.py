import numpy as np
from scipy import stats

# z-score
def zscore(distribution, individual):
    return (individual-np.mean(distribution))/np.std(distribution)

# median absolute deviation  
def mad(data):
    return np.median(np.abs(data - np.median(data)))

# median z-score
def zscoremed(data, value):
    return np.abs(value - np.median(data))/(mad(data) * 1.4826)

# p value calculations
def p2z(pvals):
    return abs(stats.norm.ppf(pvals/2))

# arcsin sqrt transform
def asinsqrt(data):
    return np.arcsin(np.sqrt(data))

# fisher z transform
def fisherz(data):
    data = np.asarray(data)
    data[data == 1.] = 0.99
    return 0.5 * np.log((1+data)/(1-data))

# z score calculations
def z2p(zscore):
    return (1-stats.norm.cdf(abs(zscore)))*2
    
def z2r(zscore, n):
    return np.sqrt((zscore**2)/((zscore**2)+n))

def z2d(zscore, n):
    return (2*zscore) / np.sqrt(n)

# t-stat calculations
def t2r(tstat, df):
    return np.sqrt((tstat**2) / ((tstat**2) + df)) * np.sign(tstat)

def t2d(tstat, df):
    return (2*tstat) / np.sqrt(df)
    
# r value calculations
def r2t(rval, df):
    return np.sqrt((df * (rval**2)) / (1 - (rval**2))) * np.sign(rval)

def r2d(rval):
    return np.sqrt((4 * (rval**2)) / (1 - (rval**2)))
    
# Cohen's d calculations
def d2r(cohend):
    return np.sqrt((cohend**2) / (4 + (cohend**2)))

def zdiff(z1, z2, n1, n2):
    zdiff = z1 - z2

    # Calculate sigma given the number of trials
    se = np.sqrt((1 / (n1 - 3)) + (1 / (n2 - 3)))

    # Get z-score of the differences between correlation coefficients
    z = zdiff / se

    # Calculate p-value
    p = 1 - stats.norm.cdf(abs(z))

    return z, p
 
def corrdiff(r1, r2, n1, n2):
    # Fisher's z-transform to normalize correlation coefficients
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    zdiff = z1 - z2

    # Calculate sigma given the number of trials
    se = np.sqrt((1 / (n1 - 3)) + (1 / (n2 - 3)))

    # Get z-score of the differences between correlation coefficients
    z = zdiff / se

    # Calculate p-value
    p = 1 - stats.norm.cdf(abs(z))

    return z, p

def dprimen(phits, pfa, n):
    # Check for values of 1 or 0 and adjust them
    if phits == 1.:
        phits = (n - 1) / n

    if phits == 0.:
        phits = 1 / n

    if pfa == 1.:
        pfa = (n - 1) / n

    if pfa == 0.:
        pfa = 1 / n

    # Calculate d'
    return stats.norm.ppf(phits) - stats.norm.ppf(pfa)
