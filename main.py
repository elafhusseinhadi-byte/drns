# =====================================================
# 🚀 UAV Single-City Simulation Server – Baghdad Only
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, MetaData, Table
from sqlalchemy.orm import sessionmaker
import time, random, math, asyncio

# =====================================================
# CITY CONFIG
# =====================================================
CITY_NAME = "Baghdad"
CITY_CENTER = (33.3, 44.4)

COLLISION_THRESHOLD = 0.05     # UI uses same
NEAR_FACTOR = 2.0              # near = distance < threshold * factor
AVOID_ALT_STEP = 5             # raise altitude when near collision
AVOID_SPEED_STEP = -3          # slow down when too near
MOVE_DT = 0.002                # movement step

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
# DATABASE SETUP
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
# FASTAPI
# =====================================================
app = FastAPI(title="UAV – Single-City Baghdad Simulation")

# ROOT ENDPOINT
@app.get("/")
async def home():
    return {"server": "running", "city": CITY_NAME}

# HEALTH
@app.get("/health")
async def health():
    return {"status": "ok"}

# =====================================================
# PUT /uav
# =====================================================
@app.put("/uav")
async def put_uav(data: UAV):
    session = SessionLocal()
    try:
        existing = session.query(uav_table).filter_by(uav_id=data.uav_id).first()

        # assign random direction if new UAV
        direction = random.uniform(0, 2 * math.pi)

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
        return {
            "uavs": [
                {
                    "uav_id": u.uav_id,
                    "x": u.x,
                    "y": u.y,
                    "altitude": u.altitude,
                    "speed": u.speed,
                    "direction": u.direction,
                    "system_case": u.system_case,
                }
                for u in rows
            ]
        }
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
# INTERNAL PHYSICS
# =====================================================
def move_uav(u):
    # basic movement
    x = u.x + u.speed * math.cos(u.direction) * MOVE_DT
    y = u.y + u.speed * math.sin(u.direction) * MOVE_DT

    # add natural noise
    x += random.uniform(-0.0005, 0.0005)
    y += random.uniform(-0.0005, 0.0005)

    return x, y

def distance(u1, u2):
    return math.sqrt((u1.x - u2.x)**2 + (u1.y - u2.y)**2)

# =====================================================
# POST /process – Apply movement + avoidance
# =====================================================
@app.post("/process")
async def process_step():
    session = SessionLocal()

    try:
        uavs = session.query(uav_table).all()
        n = len(uavs)

        collisions = 0

        # -------- collision avoidance --------
        for i in range(n):
            for j in range(i+1, n):
                d = distance(uavs[i], uavs[j])

                # collision
                if d < COLLISION_THRESHOLD:
                    collisions += 1

                # near → apply avoidance
                elif d < COLLISION_THRESHOLD * NEAR_FACTOR:
                    # raise altitude
                    uavs[i].altitude += AVOID_ALT_STEP
                    uavs[j].altitude += AVOID_ALT_STEP
                    # slow down
                    uavs[i].speed = max(2, uavs[i].speed + AVOID_SPEED_STEP)
                    uavs[j].speed = max(2, uavs[j].speed + AVOID_SPEED_STEP)
                    # small direction change
                    uavs[i].direction += random.uniform(-0.3, 0.3)
                    uavs[j].direction += random.uniform(-0.3, 0.3)

        # -------- movement --------
        for u in uavs:
            new_x, new_y = move_uav(u)
            stmt = uav_table.update().where(
                uav_table.c.uav_id == u.uav_id
            ).values(x=new_x, y=new_y, altitude=u.altitude,
                     speed=u.speed, direction=u.direction)
            session.execute(stmt)

        session.commit()

        await asyncio.sleep(0.001 * n)

        return {
            "processed": n,
            "collisions": collisions,
            "status": "ok"
        }

    finally:
        session.close()

# =====================================================
# LOCAL RUN (IGNORE ON RENDER)
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
