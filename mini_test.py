from sklearn.externals import joblib

models = joblib.load("models.pkl")

print models["HB"].predict([[1.96608000e+05,   4.09600000e+03,   4.53264977e-04,
         1.00800000e+03,   2.17370651e+01,   9.76562500e-03]])

print models["HB"].predict([[1.00504000e+05,   3.40000000e+01,   1.05883497e-01,
         5.91200000e+03,   1.01389964e+03,   5.88235294e-02,]])

