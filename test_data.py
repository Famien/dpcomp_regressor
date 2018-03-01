import scipy.stats
import matplotlib.pyplot as plt
distribution = scipy.stats.norm(loc=0,scale=10000000000000)
sample = distribution.rvs(size=10000)
plt.hist(sample)
plt.show()
