# =====================================================
# 🚀 UAV Single-City Simulation Server – Baghdad (Enhanced)
#      Ultra-Motion + High Noise + Dynamic Direction
#      Produces strong analytics & visible changes
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, MetaData, Table
from sqlalchemy.orm import sessionmaker
import random, math, asyncio

# =====================================================
# CITY CONFIG
# =====================================================
CITY_NAME   = "Baghdad"
CITY_CENTER = (33.3, 44.4)

COLLISION_THRESHOLD = 0.05
NEAR_FACTOR         = 2.0
AVOID_ALT_STEP      = 5
AVOID_SPEED_STEP    = -3

# 🔥🔥 الحركة الآن أسرع ×40
MOVE_DT = 0.08

# Zones
NO_FLY_ZONES = [
    {"cx": 33.3, "cy": 44.4, "r": 0.2},
    {"cx": 33.0, "cy": 44.0, "r": 0.15},
]

# =====================================================
# MODELS
# =====================================================
class UAV(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    speed: float
    system_case: str = "normal"

# =====================================================
# DATABASE
# =====================================================
engine = create_engine(
    "sqlite:///uav_single_city.sqlite",
    connect_args={"check_same_thread": False}
)

metadata = MetaData()
uav_table = Table(
    "uavs", metadata,
    Column("uav_id", Integer, primary_key=True),
    Column("x", Float),
    Column("y", Float),
    Column("altitude", Float),
    Column("speed", Float),
    Column("direction", Float),
    Column("system_case", String),
)
metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="UAV – Enhanced Baghdad Server (Ultra Motion)")

@app.get("/")
async def home():
    return {"server": "running", "city": CITY_NAME}

@app.get("/health")
async def health():
    return {"status": "ok"}

# =====================================================
# HELPERS
# =====================================================
def distance(u1, u2):
    return math.sqrt((u1.x - u2.x)**2 + (u1.y - u2.y)**2)

def inside_nfz_xy(x, y):
    for idx, z in enumerate(NO_FLY_ZONES):
        dx = x - z["cx"]
        dy = y - z["cy"]
        if math.sqrt(dx*dx + dy*dy) <= z["r"]:
            return True, idx
    return False, None

def move_uav(u):
    # 🔥 حركة أسرع (بسبب MOVE_DT الكبير)
    x = u.x + u.speed * math.cos(u.direction) * MOVE_DT
    y = u.y + u.speed * math.sin(u.direction) * MOVE_DT

    # 🔥 Noise قوي حتى يصير Near / Collision واضح
    x += random.uniform(-0.0015, 0.0015)
    y += random.uniform(-0.0015, 0.0015)

    # 🔥 اتجاه ديناميكي – حتى لا تمشي بخط مستقيم
    u.direction += random.uniform(-0.1, 0.1)

    return x, y

def compute_future_prediction(uavs):
    future_set = set()
    for u in uavs:
        fx = u.x + u.speed * math.cos(u.direction) * 0.0008
        fy = u.y + u.speed * math.sin(u.direction) * 0.0008
        for v in uavs:
            if v.uav_id != u.uav_id:
                d = math.sqrt((fx - v.x)**2 + (fy - v.y)**2)
                if d < COLLISION_THRESHOLD:
                    future_set.add(u.uav_id)
                    future_set.add(v.uav_id)
    return future_set

def compute_risk(uavs, future_ids):
    risks = {}
    for u in uavs:
        min_d = 999
        for v in uavs:
            if v.uav_id != u.uav_id:
                min_d = min(min_d, distance(u, v))

        is_near   = min_d < COLLISION_THRESHOLD * NEAR_FACTOR
        is_future = u.uav_id in future_ids
        score = 0
        if min_d < COLLISION_THRESHOLD: score += 1.0
        if is_near:                     score += 0.5
        if is_future:                   score += 0.8
        risks[u.uav_id] = round(score, 3)
    return risks

# =====================================================
# PUT /uav
# =====================================================
@app.put("/uav")
async def put_uav(data: UAV):
    session = SessionLocal()
    try:
        existing = session.query(uav_table).filter_by(uav_id=data.uav_id).first()
        direction = random.uniform(0, 2*math.pi)

        values = {
            "x": data.x,
            "y": data.y,
            "altitude": data.altitude,
            "speed": data.speed,
            "system_case": data.system_case,
            "direction": direction if not existing else existing.direction,
        }

        if existing:
            stmt = uav_table.update().where(
                uav_table.c.uav_id == data.uav_id
            ).values(**values)
        else:
            stmt = uav_table.insert().values(uav_id=data.uav_id, **values)

        session.execute(stmt)
        session.commit()
        return {"status": "ok"}

    finally:
        session.close()

# =====================================================
# GET /uavs
# =====================================================
@app.get("/uavs")
async def get_uavs():
    session = SessionLocal()
    try:
        rows = session.query(uav_table).all()
        if not rows:
            return {"uavs": []}

        uavs = rows
        future_ids = compute_future_prediction(uavs)
        risks      = compute_risk(uavs, future_ids)

        result = []
        for u in uavs:
            in_nfz, _ = inside_nfz_xy(u.x, u.y)
            result.append({
                "uav_id": u.uav_id,
                "x": u.x,
                "y": u.y,
                "altitude": u.altitude,
                "speed": u.speed,
                "direction": u.direction,
                "system_case": u.system_case,
                "future_risk": (u.uav_id in future_ids),
                "risk_score": risks[u.uav_id],
                "in_nfz": in_nfz,
            })

        return {"uavs": result}

    finally:
        session.close()

# =====================================================
# DELETE /reset
# =====================================================
@app.delete("/reset")
async def reset():
    session = SessionLocal()
    try:
        session.query(uav_table).delete()
        session.commit()
        return {"status": "cleared"}
    finally:
        session.close()

# =====================================================
# POST /process
# =====================================================
@app.post("/process")
async def process_step():
    session = SessionLocal()
    try:
        uavs = session.query(uav_table).all()
        n = len(uavs)

        if n == 0:
            return {
                "processed": 0,
                "collisions_before_prediction": 0,
                "near_before_prediction": 0,
                "collisions_after_prediction": 0,
                "near_after_prediction": 0,
                "nfz_hits": 0,
                "future_risky": [],
                "risk_scores": {},
                "status": "ok"
            }

        # NFZ enforcement
        nfz_hits = 0
        for u in uavs:
            in_nfz, idx = inside_nfz_xy(u.x, u.y)
            if in_nfz:
                nfz_hits += 1
                z = NO_FLY_ZONES[idx]
                dx = u.x - z["cx"]
                dy = u.y - z["cy"]
                d  = math.sqrt(dx*dx + dy*dy) or 0.001
                u.x = z["cx"] + dx/d * (z["r"] + 0.01)
                u.y = z["cy"] + dy/d * (z["r"] + 0.01)
                u.altitude += AVOID_ALT_STEP
                u.speed = max(2, u.speed + AVOID_SPEED_STEP)

        # Before prediction
        coll_before = 0
        near_before = 0
        for i in range(n):
            for j in range(i+1, n):
                d = distance(uavs[i], uavs[j])
                if d < COLLISION_THRESHOLD:
                    coll_before += 1
                elif d < COLLISION_THRESHOLD * NEAR_FACTOR:
                    near_before += 1

        # Prediction
        future_ids = compute_future_prediction(uavs)

        # Apply avoidance
        for u in uavs:
            if u.uav_id in future_ids:
                u.altitude += AVOID_ALT_STEP
                u.speed = max(2, u.speed + AVOID_SPEED_STEP)
                u.direction += random.uniform(-0.3, 0.3)

        # After prediction
        coll_after = 0
        near_after = 0
        for i in range(n):
            for j in range(i+1, n):
                d = distance(uavs[i], uavs[j])
                if d < COLLISION_THRESHOLD:
                    coll_after += 1
                elif d < COLLISION_THRESHOLD * NEAR_FACTOR:
                    near_after += 1

        # MOVE all UAVs
        for u in uavs:
            new_x, new_y = move_uav(u)
            stmt = uav_table.update().where(
                uav_table.c.uav_id == u.uav_id
            ).values(
                x=new_x,
                y=new_y,
                altitude=u.altitude,
                speed=u.speed,
                direction=u.direction,
            )
            session.execute(stmt)

        session.commit()

        # Compute risk after move
        uavs_after = session.query(uav_table).all()
        future_ids_after = compute_future_prediction(uavs_after)
        risks = compute_risk(uavs_after, future_ids_after)

        return {
            "processed": n,
            "collisions_before_prediction": coll_before,
            "near_before_prediction": near_before,
            "collisions_after_prediction": coll_after,
            "near_after_prediction": near_after,
            "nfz_hits": nfz_hits,
            "future_risky": list(future_ids_after),
            "risk_scores": risks,
            "status": "ok"
        }

    finally:
        session.close()


# =====================================================
# LOCAL RUN
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
