from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

plt2 = plt

fh = open("KAZFLAGS43_2015-01-1_2017-01-01.csv", 'r')
#fh.readline()
rawData = np.genfromtxt(fh, delimiter = ',', usecols = (0,7), dtype=None, names = True)

print(rawData['wind_dir_degrees'][rawData['wind_dir_degrees'] > 57])
print(rawData['date'])

print(type(rawData['date'][:4][0]))
year = np.array([el[:4].decode('utf8') for el in rawData['date']])
print(year)

print(rawData['wind_dir_degrees'][year == '2015'])

print(20/360)

#code block calculates percentage of wind angle measurements within 225 and 245 degrees for 2015
wind_data_2015 = rawData['wind_dir_degrees'][year == '2015']
print(wind_data_2015)
wind_data_2015_245 = wind_data_2015[wind_data_2015[:] < 245]
print(wind_data_2015_245)
wind_data_2015_235 = wind_data_2015_245[wind_data_2015_245 > 225] #2015 data within 225-245 degrees
print(wind_data_2015_235)
print("Percentage of wind angle measurements in 2015 between 225 degrees and 245 degrees", len(wind_data_2015_235) / len(wind_data_2015))


#code block calculates percentage of wind angle measurements within 225 and 245 degrees for 2016
wind_data_2016 = rawData['wind_dir_degrees'][year == '2016'] # get data for just 2016 from large wind data sets
print(wind_data_2016)
wind_data_2016_245 = wind_data_2016[wind_data_2016 < 245]
print(wind_data_2016_245)
wind_data_2016_235 = wind_data_2016_245[wind_data_2016_245 > 225] #2015 data within 225-245 degrees
print(wind_data_2016_235)
print("Percentage of wind angle measurements in 2016 between 225 degrees and 245 degrees", len(wind_data_2016_235) / len(wind_data_2016))


plt.hist(wind_data_2015)
plt.show()
plt.hist(wind_data_2016)
plt.show()
#dto = dt.strptime(rawData['date'][0].decode('UTF-8'), '%Y-%b-%d')
#print(dto)
                  
dataPerDay, binSize = np.histogram(rawData['wind_dir_degrees'] )
print(dataPerDay, binSize)


plt2.hist(rawData['wind_dir_degrees'])
plt2.show()
plt.bar(binSize[:-1], dataPerDay, width = (binSize[-1] - binSize[0])/ (len(binSize) - 1), align = 'edge')
plt.xlim(min(binSize), max(binSize))
plt.show() 

#plt.hist(rawData, binSize)
#plt.show()

'''
for n in range(len(rawData)):
    if rawData['date'][n][0:4] == b'2015':
        rawData2015.append([rawData['date'][n], rawData['wind_dir_degrees'][n]])
'''
