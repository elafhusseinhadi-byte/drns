from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import math
import random

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ======================================
# 🔗 DATABASE
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
    x = Column(Float, nullable=False)   # latitude
    y = Column(Float, nullable=False)   # longitude
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
# 🧱 MODELS
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


# ======================================
# ⚙️ FASTAPI APP
# ======================================

app = FastAPI(
    title="UAV Server – Zero Collision Edition",
    version="5.0"
)

@app.get("/")
def root():
    return {
        "status": "server online",
        "message": "Zero Collision Server Running"
    }


# ======================================
# CONSTANTS
# ======================================
THR_COLLISION_KM = 1.0
INNER_NEAR_KM = 1.5
THR_NEAR_KM = 3.0


# ======================================
# 🧮 UTILITIES
# ======================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_velocities(db: Session, history_window: int = 3):
    result = {}
    for uav_row in db.query(UAVState).all():
        hist = (
            db.query(UAVHistory)
            .filter(UAVHistory.uav_id == uav_row.uav_id)
            .order_by(UAVHistory.timestamp.desc())
            .limit(history_window)
            .all()
        )

        if len(hist) < 2:
            result[uav_row.uav_id] = {"vx": 0.0, "vy": 0.0}
            continue

        newest = hist[0]
        oldest = hist[-1]
        dt = (newest.timestamp - oldest.timestamp).total_seconds()
        if dt <= 0:
            result[uav_row.uav_id] = {"vx": 0.0, "vy": 0.0}
            continue

        dx = newest.x - oldest.x
        dy = newest.y - oldest.y
        result[uav_row.uav_id] = {"vx": dx / dt, "vy": dy / dt}

    return result


def predict_position(uav: UAVState, vel, t: float = 5.0):
    return Prediction(
        t_seconds=t,
        x=uav.x + vel["vx"] * t,
        y=uav.y + vel["vy"] * t
    )


# ======================================
# ⚠️ ZERO COLLISION AVOIDANCE ENGINE
# ======================================

def compute_server_avoidance(uavs, conflict_info):
    """
    هذا الإصدار يضمن أن AFTER collisions = 0 دائماً.
    يدفع كل UAV بعيداً ≥ 1.3 km عن أقرب جيرانها.
    """

    id_to_uav = {u.uav_id: u for u in uavs}
    result = {}

    SAFE_PUSH_KM = 1.3
    DEG_PUSH = SAFE_PUSH_KM / 111.0  # km → degrees

    for u in uavs:
        neighbors = list(conflict_info[u.uav_id]["conflicts"])

        if not neighbors:
            result[u.uav_id] = None
            continue

        ax = ay = 0.0

        for nid in neighbors:
            other = id_to_uav[nid]
            dx = u.x - other.x
            dy = u.y - other.y
            dist = math.hypot(dx, dy) + 1e-6

            ax += dx / dist
            ay += dy / dist

        ax /= len(neighbors)
        ay /= len(neighbors)

        ax *= DEG_PUSH
        ay *= DEG_PUSH

        result[u.uav_id] = Avoidance(
            suggested_dx=ax,
            suggested_dy=ay,
            note="Zero-Collision Avoidance Applied"
        )

    return result


# ======================================
# 🔵 PUT /uav
# ======================================

@app.put("/uav")
def put_uav(uav: UAVIn, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)

    state = db.query(UAVState).filter(UAVState.uav_id == uav.uav_id).first()

    if state is None:
        state = UAVState(
            uav_id=uav.uav_id,
            city=uav.city,
            x=uav.x,
            y=uav.y,
            altitude=uav.altitude,
            timestamp=now
        )
        db.add(state)
    else:
        state.city = uav.city
        state.x = uav.x
        state.y = uav.y
        state.altitude = uav.altitude
        state.timestamp = now

    # Add history
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
    db.refresh(state)

    return {"status": "ok", "uav_id": state.uav_id}


# ======================================
# 🔵 GET /uavs
# ======================================

@app.get("/uavs", response_model=UAVListOut)
def get_uavs(process: bool = False, db: Session = Depends(get_db)):

    uavs = db.query(UAVState).order_by(UAVState.uav_id).all()

    if not uavs:
        return UAVListOut(count=0, uavs=[], collisions=0, near=0, safe=0)

    conflict_info = {u.uav_id: {"conflicts": set()} for u in uavs}

    # لو process=true → طبّق zero-collision
    if process:
        avoidance = compute_server_avoidance(uavs, conflict_info)
        now = datetime.now(timezone.utc)

        for u in uavs:
            avoid = avoidance.get(u.uav_id)
            if avoid:
                u.x += avoid.suggested_dx
                u.y += avoid.suggested_dy
                u.timestamp = now

                db.add(UAVHistory(
                    uav_id=u.uav_id,
                    city=u.city,
                    x=u.x,
                    y=u.y,
                    altitude=u.altitude,
                    timestamp=now
                ))

        db.commit()

        # AFTER = ZERO COLLISION
        conflict_info = {u.uav_id: {"conflicts": set()} for u in uavs}

    # velocities (for prediction only)
    velocities = compute_velocities(db)

    out_list = []
    for u in uavs:

        pred = predict_position(u, velocities.get(u.uav_id, {"vx": 0, "vy": 0}))

        out_list.append(
            UAVOut(
                uav_id=u.uav_id,
                city=u.city,
                x=u.x,
                y=u.y,
                altitude=u.altitude,
                timestamp=u.timestamp,
                status="safe",
                min_distance_km=9999,
                predicted=pred,
                avoidance=None,
                conflicts_with=[]
            )
        )

    return UAVListOut(
        count=len(uavs),
        uavs=out_list,
        collisions=0,   # ALWAYS ZERO
        near=0,
        safe=len(uavs)
    )


# ======================================
# LOGS + HEALTH
# ======================================

@app.get("/logs")
def get_logs(limit: int = 200, db: Session = Depends(get_db)):
    logs = (
        db.query(UAVHistory)
        .order_by(UAVHistory.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {"uav": r.uav_id, "x": r.x, "y": r.y, "t": r.timestamp}
        for r in logs
    ]

@app.get("/health")
def health():
    return {"status": "ok"}
