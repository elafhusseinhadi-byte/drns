# =====================================================
# 🚀 UAV Simulation Server – Multi-City Transfer + Random Spread
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    MetaData, Table, and_
)
from sqlalchemy.orm import sessionmaker
import time, asyncio, random

# =====================================================
# 🌍 City Coordinates (You can add more cities)
# =====================================================
CITY_COORDS = {
    "Baghdad": (33.3, 44.4),
    "Basra":   (30.5, 47.8),
    "Najaf":   (31.99, 44.31),
}

# =====================================================
# 🛰️ UAV Model (received from client)
# =====================================================
class UAV(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    speed: float
    system_case: str       # normal, avoidance
    target_city: str | None = None
    progress: int = 0      # 0..100


class TransferRequest(BaseModel):
    uav_id: int
    from_city: str
    to_city: str


# =====================================================
# 🗄 Database Setup
# =====================================================
engine = create_engine("sqlite:///uav_db_full.sqlite",
                       connect_args={"check_same_thread": False})

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
    Column("target_city", String, nullable=True),
    Column("progress", Integer, default=0),
)

metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# =====================================================
# 🚀 FASTAPI App
# =====================================================
app = FastAPI(title="UAV Simulation Server – Multi-City + Spread Fix")

# =====================================================
# PUT /city/{city}/uav
# =====================================================
@app.put("/city/{city}/uav")
async def put_uav(city: str, data: UAV):
    session = SessionLocal()
    start = time.time()
    try:
        existing = session.query(uav_table).filter_by(
            city_name=city,
            uav_id=data.uav_id
        ).first()

        values = {
            "x": data.x,
            "y": data.y,
            "altitude": data.altitude,
            "speed": data.speed,
            "system_case": data.system_case,
            "city_name": city,
            "target_city": data.target_city,
            "progress": data.progress,
        }

        if existing:
            stmt = (
                uav_table.update()
                .where(
                    and_(
                        uav_table.c.city_name == city,
                        uav_table.c.uav_id == data.uav_id
                    )
                )
                .values(**values)
            )
            session.execute(stmt)
        else:
            values["uav_id"] = data.uav_id
            stmt = uav_table.insert().values(**values)
            session.execute(stmt)

        session.commit()
        elapsed_ms = (time.time() - start) * 1000
        return {"status": "ok", "put_time_ms": round(elapsed_ms, 3)}

    finally:
        session.close()


# =====================================================
# GET /city/{city}/uavs
# =====================================================
@app.get("/city/{city}/uavs")
async def get_uavs(city: str, system_case: str = None):
    session = SessionLocal()
    start = time.time()
    try:
        query = session.query(uav_table).filter_by(city_name=city)
        if system_case:
            query = query.filter_by(system_case=system_case)

        uavs = query.all()
        elapsed_ms = (time.time() - start) * 1000

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
                    "target_city": u.target_city,
                    "progress": u.progress,
                }
                for u in uavs
            ],
            "get_time_ms": round(elapsed_ms, 3),
            "db_size_kb": round(len(uavs) * 0.5, 2),
        }

    finally:
        session.close()


# =====================================================
# POST /transfer — Start moving UAV to another city
# =====================================================
@app.post("/transfer")
async def transfer_uav(req: TransferRequest):
    session = SessionLocal()
    try:
        uav = session.query(uav_table).filter_by(
            city_name=req.from_city,
            uav_id=req.uav_id
        ).first()

        if not uav:
            return {"status": "error", "message": "UAV not found in source city"}

        stmt = (
            uav_table.update()
            .where(and_(
                uav_table.c.city_name == req.from_city,
                uav_table.c.uav_id == req.uav_id
            ))
            .values(target_city=req.to_city, progress=0)
        )
        session.execute(stmt)
        session.commit()

        return {"status": "ok", "message": "Transfer Started"}

    finally:
        session.close()


# =====================================================
# Internal function to update travel progress
# =====================================================
def update_transfers(session, city):
    uavs = (
        session.query(uav_table)
        .filter_by(city_name=city)
        .filter(uav_table.c.target_city.isnot(None))
        .all()
    )

    moved = 0

    for u in uavs:
        if u.city_name not in CITY_COORDS or u.target_city not in CITY_COORDS:
            continue

        Ax, Ay = CITY_COORDS[u.city_name]
        Bx, By = CITY_COORDS[u.target_city]

        new_progress = min((u.progress or 0) + 10, 100)
        t = new_progress / 100

        new_x = Ax + t * (Bx - Ax)
        new_y = Ay + t * (By - Ay)

        stmt = (
            uav_table.update()
            .where(
                and_(
                    uav_table.c.city_name == u.city_name,
                    uav_table.c.uav_id == u.uav_id
                )
            )
            .values(x=new_x, y=new_y, progress=new_progress)
        )

        # =====================================================
        # ⭐ Spread Fix — Random distribution when reaching target city
        # =====================================================
        if new_progress >= 100:
            spread_x = random.uniform(-0.4, 0.4)
            spread_y = random.uniform(-0.4, 0.4)

            stmt = stmt.values(
                city_name=u.target_city,
                target_city=None,
                progress=100,
                x=Bx + spread_x,
                y=By + spread_y
            )

        session.execute(stmt)
        moved += 1

    return moved


# =====================================================
# POST /city/{city}/process
# =====================================================
@app.post("/city/{city}/process")
async def process_uavs(city: str, system_case: str = None):
    session = SessionLocal()
    start = time.time()

    try:
        moved = update_transfers(session, city)
        session.commit()

        query = session.query(uav_table).filter_by(city_name=city)
        if system_case:
            query = query.filter_by(system_case=system_case)

        uavs = query.all()
        n = len(uavs)

        collisions = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = uavs[i].x - uavs[j].x
                dy = uavs[i].y - uavs[j].y
                if (dx*dx + dy*dy) ** 0.5 < 5:
                    collisions += 1

        await asyncio.sleep(0.001 * n)
        elapsed = (time.time() - start) * 1000

        return {
            "processed_uavs": n,
            "moved": moved,
            "collisions": collisions,
            "post_time_ms": round(elapsed, 3),
        }

    finally:
        session.close()


# =====================================================
# Health Check
# =====================================================
@app.get("/health")
async def health():
    return {"status": "ok"}


# =====================================================
# Run Local
# =====================================================
if __name__ == "__main__":
    import uvicorn
    @app.get("/debug/db")
async def debug_db():
    session = SessionLocal()
    try:
        rows = session.query(uav_table).all()
        return {
            "rows": [
                {
                    "uav_id": u.uav_id,
                    "city_name": u.city_name,
                    "x": u.x,
                    "y": u.y,
                    "altitude": u.altitude,
                    "speed": u.speed,
                    "system_case": u.system_case,
                    "target_city": u.target_city,
                    "progress": u.progress,
                }
                for u in rows
            ]
        }
    finally:
        session.close()

    uvicorn.run(app, host="0.0.0.0", port=10000)
