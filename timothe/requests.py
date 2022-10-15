import json
import requests

res = requests.post("https://leaderboard42.herokuapp.com/reveal/", data={
       'username': 'my_awesome_team',
       'password': "my_password",
       'exercise_id': 0,
       'datum_id': 4,
   })
 
try:
   res = json.loads(res.content)
   print(res)
except:
   print("Error")
   print(res.content)
 
# {'exercise_id': 0, 'datum_id': 4, 'label': 0, 'previously revealed': [12]}