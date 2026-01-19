#!/usr/bin/env python
# coding: utf-8

import requests
import json

# Sample customer data (you can replace with data/customer.json)
# customer = {"MOSTYPE": 9, "MAANTHUI": 1, "MGEMOMV": 2, 
#             "MGEMLEEF": 3, "MOSHOOFD": 3, "MGODRK": 0, 
#             "MGODPR": 4, "MGODOV": 1, "MGODGE": 5, 
#             "MRELGE": 4, "MRELSA": 3, "MRELOV": 3, 
#             "MFALLEEN": 2, "MFGEKIND": 3, "MFWEKIND": 4, 
#             "MOPLHOOG": 1, "MOPLMIDD": 4, "MOPLLAAG": 5, 
#             "MBERHOOG": 1, "MBERZELF": 0, "MBERBOER": 0, 
#             "MBERMIDD": 2, "MBERARBG": 3, "MBERARBO": 4, 
#             "MSKA": 1, "MSKB1": 1, "MSKB2": 3, "MSKC": 4, 
#             "MSKD": 1, "MHHUUR": 8, "MHKOOP": 1, "MAUT1": 6, 
#             "MAUT2": 1, "MAUT0": 2, "MZFONDS": 8, "MZPART": 1, 
#             "MINKM30": 3, "MINK3045": 4, "MINK4575": 2, "MINK7512": 0, 
#             "MINK123M": 0, "MINKGEM": 3, "MKOOPKLA": 4, "PWAPART": 0, 
#             "PWABEDR": 0, "PWALAND": 0, "PPERSAUT": 6, "PBESAUT": 0, 
#             "PMOTSCO": 0, "PVRAAUT": 0, "PAANHANG": 0, "PTRACTOR": 0, 
#             "PWERKT": 0, "PBROM": 0, "PLEVEN": 0, "PPERSONG": 0, 
#             "PGEZONG": 0, "PWAOREG": 0, "PBRAND": 0, "PZEILPL": 0, 
#             "PPLEZIER": 0, "PFIETS": 0, "PINBOED": 0, "PBYSTAND": 0, 
#             "AWAPART": 0, "AWABEDR": 0, "AWALAND": 0, "APERSAUT": 1, 
#             "ABESAUT": 0, "AMOTSCO": 0, "AVRAAUT": 0, "AAANHANG": 0, 
#             "ATRACTOR": 0, "AWERKT": 0, "ABROM": 0, "ALEVEN": 0, 
#             "APERSONG": 0, "AGEZONG": 0, "AWAOREG": 0, "ABRAND": 0, 
#             "AZEILPL": 0, "APLEZIER": 0, "AFIETS": 0, "AINBOED": 0, "ABYSTAND": 0}

# Or load from your saved customer.json
with open('data/customer.json', 'r') as f_in:
    customer = json.load(f_in)

host = "localhost:9696"
url = f'http://{host}/predict'

response = requests.post(url, json=customer).json()
print(response)

if response['potential_customer']:
    print(f"Customer is a potential caravan insurance buyer!")
    print(f"Probability: {response['probability']:.2%}")
else:
    print(f"Customer is unlikely to buy caravan insurance.")
    print(f"Probability: {response['probability']:.2%}")