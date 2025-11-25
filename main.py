# =====================================================
# 🚀 Single-City UAV Simulation Server (Baghdad Only)
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, MetaData, Table
from sqlalchemy.orm import sessionmaker
from fastapi.responses import FileResponse
from math import sqrt
import time, random

# ==========================
# ⚙️ Settings
# ==========================
CITY_NAME = "Baghdad"
COLLISION_THRESHOLD = 0.05   # نفس مقياس واجهة الكولاب
NEAR_FACTOR = 2.0

# ==========================
# 🛰 Pydantic Model
# ==========================
class UAV(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    speed: float
    system_case: str = "normal"   # normal / avoidance

# ==========================
# 🗄 Database
# ==========================
engine = create_engine(
    "sqlite:///uav_single_city.sqlite",
    connect_args={"check_same_thread": False}
)
metadata = MetaData()

uav_table = Table(
    "uavs", metadata,
    Column("uav_id", Integer, primary_key=True),
    Column("city_name", String, index=True),
    Column("x", Float),
    Column("y", Float),
    Column("altitude", Float),
    Column("speed", Float),
    Column("system_case", String),
)

metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ==========================
# 🚀 FastAPI App
# ==========================
app = FastAPI(title="Single-City UAV Server")

# ==========================
# PUT /uav  → upsert
# ==========================
@app.put("/uav")
async def put_uav(data: UAV):
    session = SessionLocal()
    start = time.time()
    try:
        # هل موجود مسبقاً؟
        existing = session.execute(
            uav_table.select().where(uav_table.c.uav_id == data.uav_id)
        ).fetchone()

        values = {
            "city_name": CITY_NAME,
            "x": data.x,
            "y": data.y,
            "altitude": data.altitude,
            "speed": data.speed,
            "system_case": data.system_case,
        }

        if existing:
            stmt = (
                uav_table.update()
                .where(uav_table.c.uav_id == data.uav_id)
                .values(**values)
            )
        else:
            values["uav_id"] = data.uav_id
            stmt = uav_table.insert().values(**values)

        session.execute(stmt)
        session.commit()

        elapsed = (time.time() - start) * 1000
        return {"status": "ok", "put_time_ms": round(elapsed, 3)}
    finally:
        session.close()

# ==========================
# GET /uavs
# ==========================
@app.get("/uavs")
async def get_uavs():
    session = SessionLocal()
    start = time.time()
    try:
        rows = session.execute(
            uav_table.select().where(uav_table.c.city_name == CITY_NAME)
        ).fetchall()

        elapsed = (time.time() - start) * 1000

        return {
            "uavs": [
                {
                    "uav_id": u.uav_id,
                    "x": u.x,
                    "y": u.y,
                    "altitude": u.altitude,
                    "speed": u.speed,
                    "system_case": u.system_case,
                    "city_name": u.city_name,
                }
                for u in rows
            ],
            "get_time_ms": round(elapsed, 3),
            "count": len(rows),
        }
    finally:
        session.close()

# ==========================
# DELETE /reset  → clear all
# ==========================
@app.delete("/reset")
async def reset_city():
    session = SessionLocal()
    start = time.time()
    try:
        stmt = uav_table.delete().where(uav_table.c.city_name == CITY_NAME)
        result = session.execute(stmt)
        session.commit()
        elapsed = (time.time() - start) * 1000
        deleted = result.rowcount if hasattr(result, "rowcount") else None
        return {
            "status": "ok",
            "deleted_rows": deleted,
            "reset_time_ms": round(elapsed, 3),
        }
    finally:
        session.close()

# ==========================
# POST /process
# حركة + Collision Avoidance
# ==========================
@app.post("/process")
async def process_uavs():
    session = SessionLocal()
    start = time.time()
    try:
        rows = session.execute(
            uav_table.select().where(uav_table.c.city_name == CITY_NAME)
        ).fetchall()

        n = len(rows)
        if n == 0:
            return {"processed_uavs": 0, "moved": 0, "collisions": 0, "predicted_pairs": 0}

        # حرك الطائرات بشكل بسيط (jitter)
        moved = 0
        uavs_list = []
        for u in rows:
            new_x = u.x + random.uniform(-0.01, 0.01)
            new_y = u.y + random.uniform(-0.01, 0.01)

            # نحافظ على مجال تقريبي حول بغداد
            new_x = max(32.5, min(34.1, new_x))
            new_y = max(43.6, min(45.2, new_y))

            u_dict = {
                "uav_id": u.uav_id,
                "x": new_x,
                "y": new_y,
                "altitude": u.altitude,
                "speed": u.speed,
                "system_case": u.system_case,
            }
            uavs_list.append(u_dict)
            moved += 1

        # حساب التصادمات + near collisions
        collisions = []
        predicted = []

        for i in range(n):
            for j in range(i + 1, n):
                a = uavs_list[i]
                b = uavs_list[j]
                dxy = sqrt((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2)
                dalt = abs(a["altitude"] - b["altitude"]) / 100.0
                d = sqrt(dxy*dxy + dalt*dalt)

                if d < COLLISION_THRESHOLD:
                    collisions.append((a["uav_id"], b["uav_id"]))
                elif d < COLLISION_THRESHOLD * NEAR_FACTOR:
                    predicted.append((a["uav_id"], b["uav_id"]))

        # Collision Avoidance بسيط: رفع الارتفاع للطائرات القريبة
        avoid_set = set([uid for pair in predicted for uid in pair])
        new_rows_values = []
        for u in uavs_list:
            if u["uav_id"] in avoid_set:
                u["altitude"] += 10.0  # نرفع الارتفاع
                u["system_case"] = "avoidance"
            new_rows_values.append(u)

        # اكتب القيم الجديدة في DB
        for u in new_rows_values:
            stmt = (
                uav_table.update()
                .where(uav_table.c.uav_id == u["uav_id"])
                .values(
                    x=u["x"],
                    y=u["y"],
                    altitude=u["altitude"],
                    speed=u["speed"],
                    system_case=u["system_case"],
                )
            )
            session.execute(stmt)

        session.commit()
        elapsed = (time.time() - start) * 1000

        return {
            "processed_uavs": n,
            "moved": moved,
            "collisions": len(collisions),
            "predicted_pairs": len(predicted),
            "post_time_ms": round(elapsed, 3),
        }
    finally:
        session.close()

# ==========================
# DEBUG: download database
# ==========================
@app.get("/download/db")
async def download_db():
    return FileResponse("uav_single_city.sqlite")

# ==========================
# HEALTH
# ==========================
@app.get("/health")
async def health():
    return {"status": "ok"}

# ==========================
# LOCAL RUN
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
