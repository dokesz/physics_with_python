import numpy as np
import pandas as pd

# Load GPS data into a pandas dataframe
df = pd.read_csv('busz-1.txt')

df["tdat"]=pd.to_datetime(df['date time'], format="%Y-%m-%d %H:%M:%S")

tdat_arr=df.tdat.to_numpy()
t_arr=(tdat_arr-tdat_arr[0])/np.timedelta64(1, 's')  
# most t_arr-ban az első mérési ponttól eltelt idő van
print(t_arr[:5], "...", t_arr[-5:])

# Calculate velocity using the haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # Earth's radius in km
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.sin(dLat/2)**2 + np.sin(dLon/2)**2 * np.cos(lat1) * np.cos(lat2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

df['velocity'] = np.vectorize(haversine)(df['latitude'], df['longitude'], df['latitude'].shift(), df['longitude'].shift())

#print(df['velocity']-df['velocity'].shift())

# Calculate time interval between consecutive GPS points
#df['time_interval'] = df['tdat'] - df['tdat'].shift()

# Calculate tangential acceleration
df['tangential_acceleration'] = (df['velocity'] - df['velocity'].shift()) / t_arr #df['time_interval']

# Remove the first row, which has NaN values
#df = df.dropna()

# Print the resulting dataframe
#print(df['tangential_acceleration'])

#def radius_of_curvature(x1, y1, x2, y2, x3, y3):
    #R = 6371 # Earth's radius in km
    #a = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    #b = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
    #c = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
    #s = (a + b + c) / 2
    #area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    #print(area)
    #print(area)
    #radius = a * b * c / (4 * area)
    #cleanedList = [x for x in radius if str(x) != 'nan']
    #print(cleanedList)
    #return radius

def radius_of_curvature(lat1, lon1, lat2, lon2, lat3, lon3):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2, lat3, lon3 = map(np.radians, [lat1, lon1, lat2, lon2, lat3, lon3])
    R = 6371000 # Earth's radius in meters
    # Haversine formula
    dlat1 = lat2 - lat1
    dlon1 = lon2 - lon1
    dlat2 = lat3 - lat2
    dlon2 = lon3 - lon2
    a = np.sin(dlat1/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon1/2)**2
    c1 = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    a = np.sin(dlat2/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(dlon2/2)**2
    c2 = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = 2 * R * np.arcsin(np.sqrt(c1**2 * c2**2 - (c1*c2*np.sin((dlon1+dlon2)/2))**2))
    
    # Radius of curvature in meters
    
    return d / 2

df['radius_of_curvature'] = np.vectorize(radius_of_curvature)(df['latitude'].shift(1), df['longitude'].shift(1), df['latitude'], df['longitude'], df['latitude'].shift(-1), df['longitude'].shift(-1))

# Calculate centripetal acceleration
df['centripetal_acceleration'] = df['velocity'] ** 2 / df['radius_of_curvature']

# Calculate radial acceleration
df['radial_acceleration'] = np.sqrt(df['centripetal_acceleration'] ** 2 - df['tangential_acceleration'] ** 2)

print(df['radial_acceleration'])