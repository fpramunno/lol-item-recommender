import json, urllib.request, ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
with urllib.request.urlopen("https://127.0.0.1:2999/liveclientdata/allgamedata", context=ctx) as r:
    data = json.loads(r.read())
enemy = next(p for p in data["allPlayers"] if p["team"] != data["allPlayers"][0]["team"])
print(json.dumps(enemy, indent=2))
