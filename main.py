from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone
import math, random, os

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ======================================
# DATABASE
# ======================================
DATABASE_URL = "sqlite:///./uav_ultra.db"

# حذف قاعدة البيانات القديمة (مرة واحدة فقط) إذا تريد إعادة البناء
# إذا تريد حذفها كل تشغيل، خليه مثل ما هو
# إذا تريد تحذفها مرة واحدة فقط، احذفه بعد أول تشغيل
if not os.path.exists("uav_ultra.db"):
    print("⚠️ Creating a fresh UAV database...")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ======================================
# DB Models
# ======================================
class UAVState(Base):
    __tablename__ = "uav_state"
    id = Column(Integer, primary_key=True, index=True)

    uav_id = Column(Integer, unique=True, index=True, nullable=False)
    city = Column(String, default="Baghdad")

    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)

    velocity = Column(Float, default=0.0)
    heading = Column(Float, default=0.0)

    ax = Column(Float, default=0.0)
    ay = Column(Float, default=0.0)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class UAVHistory(Base):
    __tablename__ = "uav_history"
    id = Column(Integer, primary_key=True, index=True)

    uav_id = Column(Integer, index=True, nullable=False)
    city = Column(String, default="Baghdad")

    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)

    velocity = Column(Float, default=0.0)
    heading = Column(Float, default=0.0)

    ax = Column(Float, default=0.0)
    ay = Column(Float, default=0.0)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

Base.metadata.create_all(bind=engine)

# ======================================
# Dependency
# ======================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ======================================
# Pydantic Models
# ======================================
class UAVIn(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    city: Optional[str] = "Baghdad"

    velocity: Optional[float] = 0.0
    heading: Optional[float] = 0.0
    ax: Optional[float] = 0.0
    ay: Optional[float] = 0.0


class Prediction(BaseModel):
    t_seconds: float
    x: float
    y: float


class Avoidance(BaseModel):
    suggested_dx: float
    suggested_dy: float
    suggested_dv: float = 0.0
    suggested_dheading: float = 0.0
    note: str


class UAVOut(BaseModel):
    uav_id: int
    city: str
    x: float
    y: float
    altitude: float
    timestamp: datetime

    velocity: float
    heading: float
    ax: float
    ay: float

    status: str
    min_distance_km: float
    predicted: Optional[Prediction] = None
    future_path: List[Prediction] = []
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
    return {"status": "server online", "version": "Ultra Pro", "msg": "Ready for MATLAB + Streamlit"}

# ======================================
# DELETE ALL
# ======================================
@app.delete("/reset")
def reset_uavs(db: Session = Depends(get_db)):
    db.query(UAVState).delete()
    db.query(UAVHistory).delete()
    db.commit()
    return {"status": "database cleared"}

# ======================================
# UPSERT PUT /uav
# ======================================
@app.put("/uav")
def put_uav(uav: UAVIn, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)

    state = db.query(UAVState).filter(UAVState.uav_id == uav.uav_id).first()

    # -------- UPSERT: إذا الطائرة جديدة نضيفها --------
    if state is None:
        state = UAVState(
            uav_id=uav.uav_id,
            city=uav.city,
            x=uav.x,
            y=uav.y,
            altitude=uav.altitude,
            velocity=uav.velocity,
            heading=uav.heading,
            ax=uav.ax,
            ay=uav.ay,
            timestamp=now
        )
        db.add(state)

    # -------- إذا موجودة نحدّثها --------
    else:
        state.x = uav.x
        state.y = uav.y
        state.altitude = uav.altitude
        state.city = uav.city
        state.velocity = uav.velocity
        state.heading = uav.heading
        state.ax = uav.ax
        state.ay = uav.ay
        state.timestamp = now

    # -------- Always log history --------
    hist = UAVHistory(
        uav_id=uav.uav_id,
        city=uav.city,
        x=uav.x,
        y=uav.y,
        altitude=uav.altitude,
        velocity=uav.velocity,
        heading=uav.heading,
        ax=uav.ax,
        ay=uav.ay,
        timestamp=now
    )
    db.add(hist)

    db.commit()
    return {"status": "ok", "mode": "insert" if state is None else "update", "uav_id": uav.uav_id}

# ======================================
# Haversine
# ======================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ======================================
# Prediction
# ======================================
def predict_state(u, t):
    vx0 = u.velocity * math.cos(u.heading)
    vy0 = u.velocity * math.sin(u.heading)

    x_future = u.x + vx0*t + 0.5*u.ax*(t**2)
    y_future = u.y + vy0*t + 0.5*u.ay*(t**2)

    return Prediction(t_seconds=t, x=x_future, y=y_future)


def predict_future(u):
    return [predict_state(u, t) for t in (5, 10, 20)]

# ======================================
# Conflict Detection
# ======================================
THR_COLLISION = 1.0
THR_INNER = 1.5
THR_OUTER = 3.0

def compute_conflicts(uavs):
    info = {u.uav_id: {"min": 9999.0, "status": "safe", "conf": []} for u in uavs}

    for i in range(len(uavs)):
        for j in range(i+1, len(uavs)):
            ui = uavs[i]
            uj = uavs[j]

            d = haversine_km(ui.y, ui.x, uj.y, uj.x)
            info[ui.uav_id]["min"] = min(info[ui.uav_id]["min"], d)
            info[uj.uav_id]["min"] = min(info[uj.uav_id]["min"], d)

            if d < THR_OUTER:
                info[ui.uav_id]["conf"].append(uj.uav_id)
                info[uj.uav_id]["conf"].append(ui.uav_id)

            if d < THR_COLLISION:
                info[ui.uav_id]["status"] = "collision"
                info[uj.uav_id]["status"] = "collision"
            elif d < THR_INNER:
                if info[ui.uav_id]["status"] != "collision":
                    info[ui.uav_id]["status"] = "inner_near"
                if info[uj.uav_id]["status"] != "collision":
                    info[uj.uav_id]["status"] = "inner_near"
            elif d < THR_OUTER:
                if info[ui.uav_id]["status"] == "safe":
                    info[ui.uav_id]["status"] = "outer_near"
                if info[uj.uav_id]["status"] == "safe":
                    info[uj.uav_id]["status"] = "outer_near"

    return info

# ======================================
# Avoidance
# ======================================
def compute_avoidance(uavs, info):
    out = {}
    for u in uavs:
        st = info[u.uav_id]
        if not st["conf"]:
            out[u.uav_id] = None
            continue

        # المتوسط الاتجاهي للهروب
        ax = ay = 0
        for nid in st["conf"]:
            other = next(k for k in uavs if k.uav_id == nid)
            dx = u.x - other.x
            dy = u.y - other.y
            d = math.hypot(dx, dy)+1e-6
            ax += dx/d
            ay += dy/d

        ax/=len(st["conf"])
        ay/=len(st["conf"])

        # scale by danger
        if st["status"] == "collision":
            scale = 0.015
        elif st["status"] == "inner_near":
            scale = 0.008
        else:
            scale = 0.003

        suggested_dx = ax*scale
        suggested_dy = ay*scale

        # adjust heading
        new_h = math.atan2(math.sin(u.heading)+ay, math.cos(u.heading)+ax)
        dheading = new_h - u.heading

        suggested_dv = -0.0002

        out[u.uav_id] = Avoidance(
            suggested_dx=suggested_dx,
            suggested_dy=suggested_dy,
            suggested_dv=suggested_dv,
            suggested_dheading=dheading,
            note=st["status"]
        )
    return out

# ======================================
# GET /uavs
# ======================================
@app.get("/uavs", response_model=UAVListOut)
def get_uavs(process: bool = False, db: Session = Depends(get_db)):
    uavs = db.query(UAVState).order_by(UAVState.uav_id).all()
    if not uavs:
        return UAVListOut(count=0, uavs=[], collisions=0, near=0, safe=0)

    info = compute_conflicts(uavs)

    if process:
        avoidance = compute_avoidance(uavs, info)
        now = datetime.now(timezone.utc)

        for u in uavs:
            a = avoidance.get(u.uav_id)
            if not a:
                continue

            u.x += a.suggested_dx
            u.y += a.suggested_dy

            u.velocity = max(0.0, u.velocity + a.suggested_dv)
            u.heading = u.heading + a.suggested_dheading

            db.add(UAVHistory(
                uav_id=u.uav_id,
                city=u.city,
                x=u.x,
                y=u.y,
                altitude=u.altitude,
                velocity=u.velocity,
                heading=u.heading,
                ax=u.ax,
                ay=u.ay,
                timestamp=now
            ))

        db.commit()
        uavs = db.query(UAVState).order_by(UAVState.uav_id).all()
        info = compute_conflicts(uavs)
    else:
        avoidance = {u.uav_id: None for u in uavs}

    out = []
    counts = {"collision":0,"inner_near":0,"outer_near":0,"safe":0}

    for u in uavs:
        st = info[u.uav_id]["status"]
        counts[st]+=1

        out.append(UAVOut(
            uav_id=u.uav_id,
            city=u.city,
            x=u.x,
            y=u.y,
            altitude=u.altitude,
            timestamp=u.timestamp,
            velocity=u.velocity,
            heading=u.heading,
            ax=u.ax,
            ay=u.ay,
            status=st,
            min_distance_km=round(info[u.uav_id]["min"],3),
            predicted=predict_state(u,5),
            future_path=predict_future(u),
            avoidance=avoidance.get(u.uav_id),
            conflicts_with=info[u.uav_id]["conf"]
        ))

    return UAVListOut(
        count=len(out),
        uavs=out,
        collisions=counts["collision"],
        near=counts["inner_near"]+counts["outer_near"],
        safe=counts["safe"]
    )

# ======================================
# LOGS
# ======================================
@app.get("/logs")
def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    logs = (
        db.query(UAVHistory)
        .order_by(UAVHistory.timestamp.desc())
        .limit(min(limit,2000))
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
            "velocity": row.velocity,
            "heading": row.heading,
            "ax": row.ax,
            "ay": row.ay,
            "timestamp": row.timestamp,
        }
        for row in logs
    ]

# ======================================
# HEALTH
# ======================================
@app.get("/health")
def health():
    return {"status": "ok", "db":"online"}
