import requests
import json
 
res = requests.post("https://leaderboard42.herokuapp.com/reveal/", data={
       'username': 'Bolo.V5',
       'password': "my_password",
       'exercise_id': 0,
       'datum_id': 4  # label requested from the val dataset.
   })
 
try:
   res = json.loads(res.content)
   print(res)
except:
   print("Error")
   print(res.content)