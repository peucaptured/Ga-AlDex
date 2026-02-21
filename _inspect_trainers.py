import sys, pandas as pd, json, ast
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel('SaveData_RPG.xlsx')
print('All columns:', df.columns.tolist())
print()

for i, row in df.iterrows():
    raw = row.get('Dados', None)
    trainer_col = row.get('Treinador', None)
    if pd.isna(raw) or not raw:
        continue
    try:
        data = json.loads(str(raw))
    except:
        try:
            data = ast.literal_eval(str(raw))
        except:
            continue

    npc = data.get('npc_user', False)
    if npc:
        continue

    name = data.get('trainer_name') or data.get('npc_name') or ''
    party = data.get('party', [])
    caught = data.get('caught', [])
    dex_uid = data.get('__dex_uid', {})
    uid_party = dex_uid.get('party', [])
    uid_caught = dex_uid.get('caught', [])

    print(f'ROW {i} | Treinador col: {repr(trainer_col)} | trainer_name in data: {repr(name)}')
    print(f'  ALL KEYS: {list(data.keys())}')
    print(f'  party:        {party}')
    print(f'  uid_party:    {uid_party}')
    print(f'  caught:       {caught}')
    print(f'  uid_caught:   {uid_caught}')
    print()
