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
ozoneCode= '42602' # actually this is NO2

# find parameters
params = requests.get(url = "https://aqs.epa.gov/data/api/list/parametersByClass?email={}&key={}&pc=CRITERIA".format(email,key))
params = params.json()

years = ['2021', '2022', '2023']
months = ['05', '06', '07', '08', '09']
thirties = ['06', '09']
thirty1s = ['05', '07', '08']

# years = ['2023']
# months = ['08', '09']
for year in years:
    for month in months:
        bdate = "{}{}01".format(year, month)
        if month in thirties:
            edate = "{}{}{}".format(year, month, '30')
        else:
            edate = "{}{}{}".format(year, month, '31')

        print(bdate, edate, bbox)
        do = requests.get("https://aqs.epa.gov/data/api/sampleData/byBox?email={}&key={}&param={}&bdate={}&edate={}&minlat={}&maxlat={}&minlon={}&maxlon={}".format(email,key,ozoneCode,bdate,edate, bbox[0], bbox[2], bbox[1], bbox[3])).json()
        #"https://aqs.epa.gov/data/api/sampleData/byBox?email=test@aqs.api&key=test&param=44201&bdate=20230701&edate=20230731&minlat=39.259770&maxlat=40.172953&minlon=-105.632996&maxlon=-104.237732"
        do = do['Data']
        do = pd.DataFrame(do)
        do.to_csv(r"D:\Will_Git\Ozone_ML\Year2\EPA_Data\no2\year{}month{}.csv".format(year, month))

df = pd.read_csv(r"D:\Will_Git\Ozone_ML\Year2\EPA_Data\no2\year2021month05.csv")

df['site'] = df['site_number'] + df['county_code']
df['site'].unique()