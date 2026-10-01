"""Check one GPU inference through the official websocket client (synthetic input)."""
import json,time
from pathlib import Path
import numpy as np
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
client=WebsocketClientPolicy('127.0.0.1',15694)
start=time.monotonic()
response=client.predict_action({'examples':[{'image':[np.zeros((224,448,3),dtype=np.uint8)],'lang':'pick up the black bowl and place it on the plate','state':np.zeros((1,16),dtype=np.float32)}]})
assert response.get('ok'),response
x=np.asarray(response['data']['normalized_actions'])
assert x.shape==(1,8,8),x.shape
assert np.isfinite(x).all(),x
out=Path('results/preflight'); out.mkdir(parents=True,exist_ok=True)
np.save(out/'synthetic_normalized_actions.npy',x)
report={'status':'passed','input':'synthetic black dual-camera image and zero state','output_shape':list(x.shape),'all_finite':True,'elapsed_seconds':time.monotonic()-start}
(out/'policy.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
