import json

sample_event = {
    "status": {"type": "finished"},
    "homeScore": {"display": 2},
    "awayScore": {"display": 1}
}

print(sample_event.get('homeScore', {}).get('display'))
