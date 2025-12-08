from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone
import math, random

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ======================================
# DATABASE
# ======================================
DATABASE_URL = "sqlite:///./uav.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UAVState(Base):
    __tablename__ = "uav_state"
    id = Column(Integer, primary_key=True, index=True)
    uav_id = Column(Integer, unique=True, index=True, nullable=False)
    city = Column(String, default="Baghdad")
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class UAVHistory(Base):
    __tablename__ = "uav_history"
    id = Column(Integer, primary_key=True, index=True)
    uav_id = Column(Integer, index=True, nullable=False)
    city = Column(String, default="Baghdad")
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ======================================
# MODELS
# ======================================
class UAVIn(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    city: Optional[str] = "Baghdad"


class Prediction(BaseModel):
    t_seconds: float
    x: float
    y: float


class Avoidance(BaseModel):
    suggested_dx: float
    suggested_dy: float
    note: str


class UAVOut(BaseModel):
    uav_id: int
    city: str
    x: float
    y: float
    altitude: float
    timestamp: datetime
    status: str
    min_distance_km: float
    predicted: Optional[Prediction] = None
    avoidance: Optional[Avoidance] = None
    conflicts_with: List[int] = []


class UAVListOut(BaseModel):
    count: int
    uavs: List[UAVOut]
    collisions: int
    near: int
    safe: int


app = FastAPI()

# ======================================
# ROOT
# ======================================
@app.get("/")
def root():
    return {"status": "server online", "project": "UAV Cloud System"}



# ======================================
# RESET FULL DATABASE
# ======================================
@app.delete("/reset")
def reset_uavs(db: Session = Depends(get_db)):
    db.query(UAVState).delete()
    db.query(UAVHistory).delete()
    db.commit()
    return {"status": "database_cleared"}



# ======================================
# INIT — Sequential UAVs
# ======================================
@app.post("/init")
def init_uavs(n: int, db: Session = Depends(get_db)):
    db.query(UAVState).delete()
    db.query(UAVHistory).delete()
    db.commit()

    for i in range(1, n+1):
        u = UAVState(
            uav_id=i,
            city="Baghdad",
            x=30.80,
            y=47.50,
            altitude=200,
            timestamp=datetime.utcnow()
        )
        db.add(u)

    db.commit()
    return {"status": "initialized", "count": n}



# ======================================
# INIT RANDOM — UAV Random Spread
# ======================================
@app.post("/init_random")
def init_uavs_random(n: int, db: Session = Depends(get_db)):
    if n <= 0:
        raise HTTPException(status_code=400, detail="n must be > 0")

    # clear all first
    db.query(UAVState).delete()
    db.query(UAVHistory).delete()
    db.commit()

    # Baghdad bounding box
    min_x, max_x = 30.55, 30.95
    min_y, max_y = 47.25, 47.75

    for i in range(1, n+1):
        u = UAVState(
            uav_id=i,
            city="Baghdad",
            x=random.uniform(min_x, max_x),
            y=random.uniform(min_y, max_y),
            altitude=random.uniform(180, 230),
            timestamp=datetime.utcnow()
        )
        db.add(u)

    db.commit()
    return {"status": "random_initialized", "count": n}




# ======================================
# DISTANCE CONSTANTS
# ======================================
THR_COLLISION_KM = 1.0
INNER_NEAR_KM = 1.5
THR_NEAR_KM = 3.0



# ======================================
# HAVERSINE
# ======================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c



# ======================================
# VELOCITY COMPUTATION
# ======================================
def compute_velocities(db: Session, history_window: int = 3):
    result = {}
    for row in db.query(UAVState).all():
        hist = (
            db.query(UAVHistory)
            .filter(UAVHistory.uav_id == row.uav_id)
            .order_by(UAVHistory.timestamp.desc())
            .limit(history_window)
            .all()
        )
        if len(hist) < 2:
            result[row.uav_id] = {"vx": 0, "vy": 0}
            continue

        newest = hist[0]
        oldest = hist[-1]
        dt = (newest.timestamp - oldest.timestamp).total_seconds()

        if dt <= 0:
            result[row.uav_id] = {"vx": 0, "vy": 0}
            continue

        dx = newest.x - oldest.x
        dy = newest.y - oldest.y
        result[row.uav_id] = {"vx": dx / dt, "vy": dy / dt}

    return result



# ======================================
# PREDICTION
# ======================================
def predict_position(uav: UAVState, vel, t: float = 5.0):
    return Prediction(
        t_seconds=t,
        x=uav.x + vel["vx"] * t,
        y=uav.y + vel["vy"] * t
    )



# ======================================
# CONFLICT DETECTION
# ======================================
def compute_conflicts(uavs):
    info = {
        u.uav_id: {"min_dist": float("inf"), "status": "safe", "conflicts": set()}
        for u in uavs
    }

    for i in range(len(uavs)):
        ui = uavs[i]
        for j in range(i + 1, len(uavs)):
            uj = uavs[j]

            d = haversine_km(ui.y, ui.x, uj.y, uj.x)

            # min distance
            info[ui.uav_id]["min_dist"] = min(info[ui.uav_id]["min_dist"], d)
            info[uj.uav_id]["min_dist"] = min(info[uj.uav_id]["min_dist"], d)

            # conflict
            if d < THR_NEAR_KM:
                info[ui.uav_id]["conflicts"].add(uj.uav_id)
                info[uj.uav_id]["conflicts"].add(ui.uav_id)

            # status
            if d < THR_COLLISION_KM:
                info[ui.uav_id]["status"] = "collision"
                info[uj.uav_id]["status"] = "collision"
            elif d < INNER_NEAR_KM:
                if info[ui.uav_id]["status"] != "collision":
                    info[ui.uav_id]["status"] = "inner_near"
                if info[uj.uav_id]["status"] != "collision":
                    info[uj.uav_id]["status"] = "inner_near"
            elif d < THR_NEAR_KM:
                if info[ui.uav_id]["status"] not in ("collision", "inner_near"):
                    info[ui.uav_id]["status"] = "outer_near"
                if info[uj.uav_id]["status"] not in ("collision", "inner_near"):
                    info[uj.uav_id]["status"] = "outer_near"

    for u in uavs:
        if info[u.uav_id]["min_dist"] == float("inf"):
            info[u.uav_id]["min_dist"] = 9999

    return info



# ======================================
# SERVER AVOIDANCE
# ======================================
def compute_server_avoidance(uavs, conflict_info):
    id_to_uav = {u.uav_id: u for u in uavs}
    result = {}

    for u in uavs:
        info = conflict_info[u.uav_id]
        dmin = info["min_dist"]
        neighbors = list(info["conflicts"])

        if dmin >= THR_NEAR_KM or not neighbors:
            result[u.uav_id] = None
            continue

        ax = ay = 0
        for nid in neighbors:
            o = id_to_uav[nid]
            dx = u.x - o.x
            dy = u.y - o.y
            dist = math.hypot(dx, dy) + 1e-6
            ax += dx / dist
            ay += dy / dist

        ax /= len(neighbors)
        ay /= len(neighbors)

        if dmin < THR_COLLISION_KM:
            scale = 0.015
            note = "collision_zone"
        elif dmin < INNER_NEAR_KM:
            scale = 0.008
            note = "inner_near_zone"
        else:
            scale = 0.003
            note = "outer_near_zone"

        result[u.uav_id] = Avoidance(
            suggested_dx=ax * scale,
            suggested_dy=ay * scale,
            note=note
        )

    return result



# ======================================
# PUT — UPDATE ONLY
# ======================================
@app.put("/uav")
def put_uav(uav: UAVIn, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)

    state = db.query(UAVState).filter(UAVState.uav_id == uav.uav_id).first()

    if state is None:
        raise HTTPException(
            status_code=400,
            detail=f"uav_id {uav.uav_id} not initialized. Use /init or /init_random"
        )

    # UPDATE
    state.x = uav.x
    state.y = uav.y
    state.altitude = uav.altitude
    state.city = uav.city
    state.timestamp = now

    # HISTORY
    hist = UAVHistory(
        uav_id=uav.uav_id,
        city=uav.city,
        x=uav.x,
        y=uav.y,
        altitude=uav.altitude,
        timestamp=now
    )
    db.add(hist)

    db.commit()
    return {"status": "updated", "uav_id": uav.uav_id}



# ======================================
# GET UAV LIST
# ======================================
@app.get("/uavs", response_model=UAVListOut)
def get_uavs(process: bool = False, db: Session = Depends(get_db)):
    uavs = db.query(UAVState).order_by(UAVState.uav_id).all()
    if not uavs:
        return UAVListOut(count=0, uavs=[], collisions=0, near=0, safe=0)

    conflict_info = compute_conflicts(uavs)

    if process:
        avoidance = compute_server_avoidance(uavs, conflict_info)
        now = datetime.now(timezone.utc)

        for u in uavs:
            a = avoidance.get(u.uav_id)
            if a:
                u.x += a.suggested_dx
                u.y += a.suggested_dy
                u.timestamp = now
                db.add(
                    UAVHistory(
                        uav_id=u.uav_id,
                        city=u.city,
                        x=u.x,
                        y=u.y,
                        altitude=u.altitude,
                        timestamp=now
                    )
                )

        db.commit()
        uavs = db.query(UAVState).order_by(UAVState.uav_id).all()
        conflict_info = compute_conflicts(uavs)
    else:
        avoidance = {u.uav_id: None for u in uavs}

    velocities = compute_velocities(db)

    counts = {"collision": 0, "inner_near": 0, "outer_near": 0, "safe": 0}
    out_list = []

    for u in uavs:
        info = conflict_info[u.uav_id]
        status = info["status"]

        counts[status] += 1

        vel = velocities.get(u.uav_id, {"vx": 0, "vy": 0})
        pred = predict_position(u, vel)

        out_list.append(
            UAVOut(
                uav_id=u.uav_id,
                city=u.city,
                x=u.x,
                y=u.y,
                altitude=u.altitude,
                timestamp=u.timestamp,
                status=status,
                min_distance_km=round(info["min_dist"], 3),
                predicted=pred,
                avoidance=avoidance.get(u.uav_id),
                conflicts_with=list(info["conflicts"])
            )
        )

    near_total = counts["inner_near"] + counts["outer_near"]

    return UAVListOut(
        count=len(out_list),
        uavs=out_list,
        collisions=counts["collision"],
        near=near_total,
        safe=counts["safe"]
    )



@app.get("/logs")
def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    logs = (
        db.query(UAVHistory)
        .order_by(UAVHistory.timestamp.desc())
        .limit(min(limit, 1000))
        .all()
    )

    return [
        {
            "id": row.id,
            "uav_id": row.uav_id,
            "city": row.city,
            "x": row.x,
            "y": row.y,
            "altitude": row.altitude,
            "timestamp": row.timestamp,
        }
        for row in logs
    ]


@app.get("/health")
def health():
    return {"status": "ok"}
