#!/usr/bin/env python
# coding: utf-8

import requests
import json

# Sample customer data
customer = {
    "MOSTYPE": 33,
    "MAANTHUI": 1,
    "MGEMOMV": 3,
    "MGEMLEEF": 2,
    "MOSHOOFD": 8,
    "MGODRK": 0,
    "MGODPR": 5,
    "MGODOV": 1,
    "MGODGE": 3,
    "MRELGE": 7,
    "MRELSA": 0,
    "MRELOV": 2,
    "MFALLEEN": 1,
    "MFGEKIND": 4,
    "MFWEKIND": 4,
    "MOPLHOOG": 1,
    "MOPLMIDD": 4,
    "MOPLLAAG": 4,
    "MBERHOOG": 1,
    "MBERZELF": 0,
    "MBERBOER": 1,
    "MBERMIDD": 3,
    "MBERARBG": 4,
    "MBERARBO": 0,
    "MSKA": 1,
    "MSKB1": 2,
    "MSKB2": 2,
    "MSKC": 4,
    "MSKD": 0,
    "MHHUUR": 0,
    "MHKOOP": 9,
    "MAUT1": 6,
    "MAUT2": 2,
    "MAUT0": 1,
    "MZFONDS": 4,
    "MZPART": 5,
    "MINKM30": 0,
    "MINK3045": 2,
    "MINK4575": 4,
    "MINK7512": 3,
    "MINK123M": 0,
    "MINKGEM": 5,
    "MKOOPKLA": 5,
    "PWAPART": 0,
    "PWABEDR": 0,
    "PWALAND": 0,
    "PPERSAUT": 5,
    "PBESTEFONDS": 0,
    "PMOTSCO": 0,
    "PVRATEFONDS": 0,
    "PATEFONDS": 0,
    "PSTADSFONDS": 0,
    "PWERKAUTO": 0,
    "PBROM": 0,
    "PLEVEN": 0,
    "PPERSONG": 0,
    "PGEZONG": 0,
    "PWAOREG": 0,
    "PBRAND": 4,
    "PZEILPL": 0,
    "PPLEZIER": 0,
    "PFIETS": 0,
    "PINBOED": 0,
    "PBYSTAND": 0,
    "AWAPART": 0,
    "AWABEDR": 0,
    "AWALAND": 0,
    "APERSAUT": 6,
    "ABESTEFONDS": 0,
    "AMOTSCO": 0,
    "AVRATEFONDS": 0,
    "AATEFONDS": 0,
    "ASTADSFONDS": 0,
    "AWERKAUTO": 0,
    "ABROM": 0,
    "ALEVEN": 0,
    "APERSONG": 0,
    "AGEZONG": 0,
    "AWAOREG": 0,
    "ABRAND": 1,
    "AZEILPL": 0,
    "APLEZIER": 0,
    "AFIETS": 0,
    "AINBOED": 0,
    "ABYSTAND": 0
}

# Render cloud URL
host = "caravan-insurance-capstone.onrender.com"
url = f'https://{host}/predict'

print(f"Sending request to {url}...")
print("(First request may take 30-50 seconds if instance is cold)")

response = requests.post(url, json=customer).json()
print(response)

if response['potential_customer']:
    print(f"Customer is a potential caravan insurance buyer!")
    print(f"Probability: {response['probability']:.2%}")
else:
    print(f"Customer is unlikely to buy caravan insurance.")
    print(f"Probability: {response['probability']:.2%}")