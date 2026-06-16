import sys, time
sys.path.insert(0, 'app')
from model_loader import get_model, get_metadata
import agent as ag

pipeline = get_model()
meta = get_metadata()

t0 = time.time()
ag.build_agent_data(pipeline, meta)
elapsed = time.time() - t0

cache = ag._agent_cache
print(f"Done in {elapsed:.1f}s")
print(f"at_risk count:     {len(cache.get('at_risk', []))}")
print(f"at_risk_all count: {len(cache.get('at_risk_all', []))}")

listings = cache.get('at_risk', [])
if listings:
    print("\nTop 5 at-risk listings:")
    for r in listings[:5]:
        print(f"  [{r['probability']:.1%}] {r['listing_name'][:40]} | rating={r['rating']} | missing={r['missing_amenities']}")
else:
    print("WARNING: at_risk list is empty!")
