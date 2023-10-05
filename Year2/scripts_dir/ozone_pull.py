'''
EPA Data pull
'''
import requests
import pandas as pd

# set variables for AQS
email = 'wkeenan21@gmail.com'
key = 'carmelswift52'
bbox = "39.259770,-105.632996,40.172953,-104.237732"
bbox = bbox.split(',')
ozoneCode= '44201'

# find parameters
params = requests.get(url = "https://aqs.epa.gov/data/api/list/parametersByClass?email={}&key={}&pc=CRITERIA".format(email,key))
params = params.json()

bdateJuly = '20210101'
edateJuly = '20211231'
years = ['2021', '2022', '2023']
months = ['05', '06', '07', '08', '09']
thirties = ['06', '09']
thirty1s = ['05', '07', '08']

for year in years:
    for month in months:
        bdate = "{}{}01".format(year, month)
        if month in thirties:
            edate = "{}{}{}".format(year, month, '30')
        else:
            edate = "{}{}{}".format(year, month, '31')

        do = requests.get("https://aqs.epa.gov/data/api/sampleData/byBox?email={}&key={}&param={}&bdate={}&edate={}&minlat={}&maxlat={}&minlon={}&maxlon={}".format(email,key,ozoneCode,bdate,edate, bbox[0], bbox[2], bbox[1], bbox[3])).json()
        do = do['Data']
        do = pd.DataFrame(do)
        do.to_csv(r"D:\Will_Git\Ozone_ML\Year2\EPA_Data\year{}month{}.csv".format(year, month))
