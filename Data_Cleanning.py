import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import googlemaps
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Load the file we will be using:
df = pd.read_csv('TIENDAS CDMX TEC_ComeVerde.csv')

# WE see columns have spaced names. We gonna clean that
df.columns = df.columns.str.replace(r'\s+', '', regex=True)


# Looking for duplicate stores.
df = df.drop_duplicates()


# We can also note that the address (Tienda) of some observations starts with Det. We gonna eliminate that.
df['Tienda'] = df.apply(lambda row: row['Tienda'].replace(row['Det'], '', 1) if row['Tienda'].startswith(row['Det']) else row['Tienda'], axis=1)

# We have got a frequency restriction depending on the type of store;
 #we will add a column 'Frecuencia' that indicates the weekly visits a store must have.
# Note: Since we have stores that require visits only once or twice a month, we will use decimal values: 0.25 and 0.5, respectively.
def frequency(x):
  if (x == 'Chedraui') or (x == 'Walmart Mexico'):
    return 2

  elif (x == 'Benavides') or (x == 'Nutrisa'):
    return 0.25

  elif (x == 'Liverpool'):
    return 0.5

  elif (x == 'La Comer'):
    return 1

  else:
    return None

df['Frecuencia'] = df['Cadena'].apply(frequency)



# Time windows that need to be followed according to 'Formato' (Type of store).
# Note it returns a 2,1 array, where the firts value is the opening time, and the second value is the closing time.

def time_window(x):
  if (x == 'Benavides') | (x == 'Liverpool'):
    return (10, 15)

  elif (x == 'CHED SELECTO') |(x == 'CHEDRAUI') | (x =='SUPER CHE') | (x == 'SUPER CHEDRAUI') | (x == 'BODEGA AURRERA') |(x == 'SUPERCENTER')  :
    return (6, 17)

  elif (x == 'CITY MARKET') |(x == 'FRESKO') | (x =='LA COMER') | (x == 'SUMESA'):
    return (8, 14)

  elif (x == 'Nutrisa'):
    return (11, 17)

  elif (x == 'WALMART EXPRESS'):
    return (8, 12)

  else:
    return None

df[['Hora_abre', 'Hora_cierra']] = df['Formato'].apply(lambda x: pd.Series(time_window(x)))



num_tiendas_por_cat = df['Cadena'].value_counts()

plt.figure(figsize = (8, 4))
ax = num_tiendas_por_cat.plot(kind = 'bar')
plt.title('Num tiendas x cadena')
plt.xlabel('Cadenas')
plt.ylabel('Num Tiendas')

for i in ax.containers:
    ax.bar_label(i, label_type='edge')

"""

---

"""


# We are gonna need a more accurate adress: we will join Formato + Tienda.
df['Direccion'] = df['Formato'] + ' ' + df['Tienda']



df = df.drop(columns = 'Tienda')

google_maps_client = googlemaps.Client(key='AIzaSyBz43mCzC6UTV1LzrhzGoowVbaAyDcxgyY')

geolocator = Nominatim(user_agent="ComeVerde_Routes_Opt")


# We are going to get coordenates out of the adresses of the stores:
def coordenates(address):
  try:
    geocode_result = google_maps_client.geocode(address)

    if geocode_result:
      location = geocode_result[0]['geometry']['location']
      return (location['lat'], location['lng'])


  except Exception as e:
    try:
      loc = geolocator.geocode(address)
      if loc is not None:
        return (loc.latitude, loc.longitude)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
      return None

df[['Latitud', 'Longitud']] = df['Direccion'].apply(lambda x: pd.Series(coordenates(x)))

# Given the coordenates we are going to procede with the route planning: making a distance and a time matrix.
# But first, we need to revise how coordenates were saved, and if there are any missing.

ad_notfound = df[(df['Latitud'].isna()) | (df['Longitud'].isna())]

ad_notfound.shape
# We see that 74 stores do not have an address that can Lead to latitude

# We are going to plot to see if there are any patterns that can help to find a better adress:
num_tiendas_ad_notfound = ad_notfound['Cadena'].value_counts()

plt.figure(figsize = (8, 4))
ax = num_tiendas_ad_notfound.plot(kind = 'bar')
plt.title('Num tiendas notfound x cadena')
plt.xlabel('Cadenas')
plt.ylabel('Num Tiendas notfound')

for i in ax.containers:
    ax.bar_label(i, label_type='edge')

# We are going to use another excel file to see if we can find a solution to the coordenates.

# Adding files per (Cadena)
# Liverpool observations will be taken down since there is a lack of information.
Chedraui_df = pd.read_csv('Chedraui_ComeVerde.csv')
FarmaciasAhorro_df = pd.read_csv('FarmaciasAhorro_ComeVerde.csv') # We are gonna replace Benavides with these, since the info is more complete
LaComer_df = pd.read_csv('LaComer_ComeVerde.csv')
Nutrisa_df = pd.read_csv('NUTRISA_ComeVerde.csv')
Walmart_df = pd.read_csv('Walmart_ComeVerde.csv')

"""This other csv can be joined or merged by a common id, which is the df['Det'] for which we have been wondering its purpose."""

# If we take a closer look to the recent files, we see that the so-called Det we are looking for is different in all types of Cadena.
# Here is the key:
#
# Chedraui_df['Det'] ==  df['Det']
# FarmaciasAhorro['FARMA ID'] ==  df['Det']
# LaComer['(L) Código Tienda)'] ==  df['Det']
# Nutrisa['N° Tienda'] ==  df['Det']
# Walmart_df['Store Nbr'] ==  df['Det']
#
# Thus:

Chedraui_df.rename(columns={'Det': 'Det'}, inplace=True)
FarmaciasAhorro_df.rename(columns={'FARMA ID': 'Det'}, inplace=True)
LaComer_df.rename(columns={'(L) Código Tienda': 'Det'}, inplace=True)
Nutrisa_df.rename(columns={'N° tienda': 'Det'}, inplace=True)
Walmart_df.rename(columns={'Store Nbr': 'Det'}, inplace=True)


# With that done, we will eliminate all columns that are useless to us from each file:

Chedraui_df = Chedraui_df[['Det', 'Plaza', 'Tienda']] # For Chedraui it is relevant for us to keep Det, keep Plaza and Tienda. Det keeps the id, Plaza an tienda give more specific address.
FarmaciasAhorro_df = FarmaciasAhorro_df[['Calle', 'Deleg Municipio']] # Since we are not linking anyting here, we wil ignore DET, and keep Calle, Deleg Municipio
LaComer_df = LaComer_df[['Det', '(L) Formato', '(L) Dirección']] # We will keep Det, Formato and Dirección
Nutrisa_df = Nutrisa_df[['Det', 'DIRECCIÓN']] # Det and Dirección
Walmart_df = Walmart_df[['Det', 'Store Type', 'Building Address']] # We are going to keep Det, Store Type; Builidng Address

# Now we gonna be dropping all Liverpool stores
df.drop(df[df['Cadena'] == 'Liverpool'].index, inplace = True)
# Also Dropping all Benavides, cause we replacing that with Farmacias del Ahorro.
df.drop(df[df['Cadena']== 'Benavides'].index, inplace = True)

# Aight, so know we gonna keep all the names of the columns the same for an easier merge.
# On the new df we created for each Cadena, we are only interested in det, and address. We gonna make those.

Chedraui_df['Address'] = Chedraui_df['Plaza'] + ' '+ Chedraui_df['Tienda'] # Building Address
Chedraui_df = Chedraui_df.drop(columns = ['Plaza', 'Tienda']) # Dropping now useless info

FarmaciasAhorro_df['Address'] = FarmaciasAhorro_df['Calle'] +' '+ FarmaciasAhorro_df['Deleg Municipio']
FarmaciasAhorro_df = FarmaciasAhorro_df.drop(columns = ['Calle', 'Deleg Municipio'])

LaComer_df['Address'] = LaComer_df['(L) Formato'] +' '+ LaComer_df['(L) Dirección']
LaComer_df = LaComer_df.drop(columns = ['(L) Formato', '(L) Dirección'])

Nutrisa_df.rename(columns = {'DIRECCIÓN': 'Address'}, inplace = True)

Walmart_df['Address'] = Walmart_df['Store Type'] +' '+ Walmart_df['Building Address']
Walmart_df = Walmart_df.drop(columns = ['Store Type', 'Building Address'])

"""All this are, from our understanding, ALL stores in the EDOMEX and CDMX area. Not ALL the ones we provide services to. Thus, we will merge these dataframes with our main df, to keep things simple and tight. We gonna use the common denominator DET to find stores that we do provide services to."""


# We gonna add a space for df to add the new addresses
df['Address'] = np.zeros(311)


"""All left for tomorrow is checking that all addresses lead to coordenates. Then just apply the algorithm"""

Chedraui_df[Chedraui_df['Det'].apply(lambda x: isinstance(x, str))]

Chedraui_df.drop([132, 133], inplace = True)


data_frames = [Chedraui_df, LaComer_df, Nutrisa_df, Walmart_df]

Chedraui_df['Det'] = Chedraui_df['Det'].astype(int)

for element in data_frames:
  print(element['Det'].dtypes)

df['Det'] = df['Det'].astype(int)

dfs_to_merge = [Chedraui_df, LaComer_df, Nutrisa_df, Walmart_df]  # Add more DataFrames to this list as needed

# Merge each DataFrame and update 'Address' column
for i, df_0 in enumerate(dfs_to_merge):
    # Merge with the current DataFrame
    merged_df = df.merge(df_0[['Det', 'Address']], on='Det', how='left', suffixes=('', f'_df{i}'))

    # Update 'Address' column with the new addresses
    df['Address'] = merged_df[f'Address_df{i}'].combine_first(df['Address'])

df.loc[(df['Address'] == 0.0) & (df['Latitud'] == 0.0)] # Checking if there are any stores that have an empty Adress and empty coordenates.

# Since there are not any stores beyond salvation, we can get all addresses and none of them manually!
# Next thing to do is to simplify addresses: having them all in one column. Since we had the address column made by ourselves, we gonna use that

# Since All Direccion values are not empty, all values in column Address that are == 0, will be equal to its value in column Dirección.

df.loc[df['Address'] == 0, 'Address'] = df['Direccion']
df = df.drop(columns = ['Direccion'])


# Now, we are going to create another DataFrame to Join Farmacias del Ahorro stores

# We gonna be adding a column Formato to Farmacias Ahorro.
FarmaciasAhorro_df['Formato'] = 'Farmacia del Ahorro'


new_coord = df[['Formato', 'Address']]

# Now we gonna merge these two:
new_coord = pd.concat([new_coord, FarmaciasAhorro_df], ignore_index=True)

new_coord.drop_duplicates(inplace= True)

# Now, we gonna get the coordenates for these guys: get them right
new_coord[['Latitud', 'Longitud']] = new_coord['Address'].apply(lambda x: pd.Series(coordenates(x)))

new_coord[(new_coord['Latitud'].isna()) | (new_coord['Longitud'].isna())]

# For simplicity purposes, we are going to drop those addresses that could not be found.
new_coord.drop(new_coord[pd.isna(new_coord['Latitud'])].index, inplace=True)

# double checking if there are any addresses without coordenates
new_coord[(new_coord['Latitud'].isna()) | (new_coord['Longitud'].isna())].shape

# We gonna perform a K-means algorithm using coordenates. We gonna represent a 'little map' and perform clustering with the closest ones.

"""but first, lets save the file into a .csv

"""

new_coord.to_csv('ComeVerde_Coord.csv', index=False)