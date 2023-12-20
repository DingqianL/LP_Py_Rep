# import packages
import pandas as pd
import statsmodels.stats.sandwich_covariance as newey
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


df = pd.read_stata(r'AED_INTERESTRATES.DTA')

# choose impulse response horizon
hmax = 12
# set datetime
df['date'] = pd.to_datetime(df['date'])
df['daten'] = pd.to_datetime(df['daten'])

# Step 1: Generate LHS variables for the LPs 

for h in range(hmax+1):
    df[f'gs10_{h}'] = df['gs10'].shift(-h)  # levels, in stata, shift(-1) is equiavalent to f1.var
    df[f'gs10d{h}'] = df['gs10'].diff(1).shift(-h)   # differences
    df[f'gs10c{h}'] = df['gs10'].shift(-h) - df['gs10'].shift(1)  # Cumulative

for i in range(4+1):
    df[f'gs1_{i}'] = df['gs1'].shift(i)  # levels
    df[f'dgs1_{i}'] = df['dgs1'].shift(i) # differences


for i in range(1, 3+1):
    df[f'gs10_l{i}'] = df['gs10'].shift(i)  # levels
    df[f'dgs10_l{i}'] = df['dgs10'].shift(i)  # levels

# Step 2: Run the LPs 
# Levels
b = []
u = []
d = []
for h in range(hmax+1):
    reg = smf.ols(f'gs10_{h} ~ gs1_0 + gs1_1 + gs1_2 + gs1_3 + gs1_4 + gs10_l1 + gs10_l2 + gs10_l3', data = df).fit(cov_type='HAC', cov_kwds={'maxlags':h})
    b.append(reg.params['gs1_0'])
    u.append(reg.params['gs1_0'] + 1.645*reg.bse['gs1_0'])
    d.append(reg.params['gs1_0'] - 1.645*reg.bse['gs1_0'])

df_result = pd.DataFrame(data = {'Upper bund': u, 'Lower bund': d, 'IR': b})

# Impulse Response Funtion graph
ax = df_result.plot(title='Impulse response of GS10 to 1pp shock to GS1', ylim = (0, 1.5))
ax.set_xlabel("Year")
ax.set_ylabel("Percent")
plt.show()


## differences
b = []
u = []
d = []
for h in range(hmax+1):
    reg = smf.ols(f'gs10d{h} ~ dgs1_0 + dgs1_1 + dgs1_2 + dgs1_3 + dgs1_4 + dgs10_l1 + dgs10_l2 + dgs10_l3', data = df).fit(cov_type='HAC', cov_kwds={'maxlags':h})
    b.append(reg.params['dgs1_0'])
    u.append(reg.params['dgs1_0'] + 1.645*reg.bse['dgs1_0'])
    d.append(reg.params['dgs1_0'] - 1.645*reg.bse['dgs1_0'])

df_result_diff = pd.DataFrame(data = {'Upper bund': u, 'Lower bund': d, 'IR': b})

ax_diff = df_result_diff.plot(title='Impulse response of GS10 to 1pp shock to GS1', ylim = (-0.2, 0.8))
ax_diff.set_xlabel("Year")
ax_diff.set_ylabel("Percent")
plt.show()

## cumulative
b = []
u = []
d = []
for h in range(hmax+1):
    reg = smf.ols(f'gs10c{h} ~ dgs1_0 + dgs1_1 + dgs1_2 + dgs1_3 + dgs1_4 + dgs10_l1 + dgs10_l2 + dgs10_l3', data = df).fit(cov_type='HAC', cov_kwds={'maxlags':h})
    b.append(reg.params['dgs1_0'])
    u.append(reg.params['dgs1_0'] + 1.645*reg.bse['dgs1_0'])
    d.append(reg.params['dgs1_0'] - 1.645*reg.bse['dgs1_0'])

df_result_diff = pd.DataFrame(data = {'Upper bund': u, 'Lower bund': d, 'IR': b})

ax_diff = df_result_diff.plot(title='Impulse response of GS10 to 1pp shock to GS1', ylim = (0, 2))
ax_diff.set_xlabel("Year")
ax_diff.set_ylabel("Percent")
plt.show()
